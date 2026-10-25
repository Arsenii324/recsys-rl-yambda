#!/usr/bin/env python3
"""Matched CE, history-baseline REINFORCE, REINFORCE-8, and RLOO-8 on one frozen 100k policy.

The grouped arm uses one policy forward per context and samples eight actions
from that same categorical distribution. All arms use the exact full-catalog
KL(pi_theta || pi_0) at temperature one. The target remains the frozen logged-
listen completion proxy; no causal/long-horizon claim is intended.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from recsys_tt2.batching import batches, make_batch, open_event_arrays
from recsys_tt2.live_monitor import LiveMonitor
from recsys_tt2.model import ConditionalCompletionRM
from recsys_tt2.rl_context_baseline import frozen_history_baseline, load_frozen_history_rm
from run_experiments import device_from, eval_policy, seed_all
from run_rl import data_provenance, eval_proxy, load_policy, make_item_durations, sha256

DIAGNOSTICS = Path(__file__).resolve().parents[1] / "research" / "diagnostics"
sys.path.insert(0, str(DIAGNOSTICS))
from exact_reward_tilt import history_embedding, score_catalog_cached  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_sampled_actions import sample_action_classes  # noqa: E402


def load_combined(path: Path, device: torch.device, n_items: int):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("variant") != "combined" or int(payload["config"]["n_items"]) != n_items:
        raise ValueError("combined reward checkpoint variant/catalog mismatch")
    model = ConditionalCompletionRM(n_items, width=int(payload["config"]["width"]), variant="combined").to(device)
    model.load_state_dict(payload["state_dict"]); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model, payload


def artifact(monitor: LiveMonitor, path: Path, name: str, metadata: dict) -> bool:
    run = getattr(monitor, "_wandb_run", None)
    wandb = getattr(monitor, "_wandb", None)
    if run is None or wandb is None:
        return False
    item = wandb.Artifact(name=name, type="model", metadata=metadata)
    item.add_file(str(path), name=path.name)
    try:
        handle = run.log_artifact(item)
        if hasattr(handle, "wait"):
            handle.wait()
        return True
    except Exception as exc:
        print(json.dumps({"event": "artifact_upload_failed", "artifact": name,
                          "reason_type": type(exc).__name__}), flush=True)
        return False


@torch.inference_mode()
def exact_q_gaps(seed: int, cfg: dict, data: Path, base_path: Path, reward_path: Path, reward,
                 train_idx, valid_all, arrays, durations, output: Path, device) -> dict:
    """Reuse one exact full-catalog q computation across all three final actors."""
    from torch.nn import functional as F
    from exact_reward_tilt import tilt_distribution
    fixed = np.random.default_rng(int(cfg["validation_seed"])).choice(
        valid_all, size=int(cfg["validation_event_count"]), replace=False)
    query = np.random.default_rng(int(cfg["exact_q_seed"])).choice(
        fixed, size=int(cfg["exact_q_event_count"]), replace=False)
    q_sha = hashlib.sha256(np.asarray(query, dtype=np.uint32).tobytes()).hexdigest()
    if q_sha != "27eebe3388f87343bd8989f1eb0158854e193d59b688de2a84d2aecc64c63e20":
        raise ValueError("exact-q query sample does not match archived H1 query cohort")
    n_items = int(cfg["n_items"])
    duration = torch.as_tensor(durations, dtype=torch.float32, device=device)
    ranks = torch.as_tensor(arrays["candidate_popularity_rank"].astype(np.int64), device=device)
    base, _, _ = load_policy(base_path, device)
    base.eval()
    all_logp0, all_reward, all_logz, q_expected_reward, q_kl = [], [], [], [], []
    for ids in batches(query, 2):
        b = make_batch(ids, arrays, n_items=n_items, max_context=base.max_context, device=device)
        logp0 = F.log_softmax(base(b["tokens"], b["padding_mask"]).float(), dim=-1)
        pooled = history_embedding(reward, b["rm_history"])
        reward_all = score_catalog_cached(reward, pooled, duration, ranks, candidate_chunk=2048)
        logq, q, logz = tilt_distribution(logp0, reward_all, float(cfg["beta"]))
        q_expected_reward.append((q * reward_all).sum(dim=-1))
        q_kl.append((q * (logq - logp0)).sum(dim=-1))
        all_logp0.append(logp0); all_reward.append(reward_all); all_logz.append(logz)
    logp0_all = torch.cat(all_logp0); reward_all = torch.cat(all_reward); logz_all = torch.cat(all_logz)
    q_obj = torch.cat(q_expected_reward) - float(cfg["beta"]) * torch.cat(q_kl)
    beta = float(cfg["beta"])
    p0_expected_reward = []
    p0_kl_to_q = []
    # Keep the full-catalog reference comparison on every registered query,
    # while limiting temporary probability tensors to eight queries at a time.
    for start in range(0, len(query), 8):
        sl = slice(start, min(start + 8, len(query)))
        lp0 = logp0_all[sl]
        p0 = lp0.exp()
        reward_slice = reward_all[sl]
        logq = lp0 + reward_slice / beta - logz_all[sl, None]
        p0_expected_reward.append((p0 * reward_slice).sum(dim=-1))
        p0_kl_to_q.append((p0 * (lp0 - logq)).sum(dim=-1))
    p0_expected_reward = torch.cat(p0_expected_reward)
    p0_kl_to_q = torch.cat(p0_kl_to_q)
    p0_gap = q_obj - p0_expected_reward
    per_actor = {}
    per_query = {
        "event_index": np.asarray(query, dtype=np.uint32),
        "p0_expected_frozen_rm": p0_expected_reward.cpu().numpy().astype(np.float32),
        "p0_exact_q_log_normalizer": logz_all.cpu().numpy().astype(np.float32),
        "p0_kl_to_exact_q": p0_kl_to_q.cpu().numpy().astype(np.float32),
        "p0_exact_q_objective_gap": p0_gap.cpu().numpy().astype(np.float32),
    }
    arms = tuple(cfg.get("arms", ("ce", "context_reinforce", "rloo8")))
    for arm in arms:
        checkpoint = output / arm / f"step-{int(cfg['steps'])}.pt"
        actor, _, _ = load_policy(checkpoint, device)
        actor.eval()
        exp_reward, kl_base, kl_q = [], [], []
        for start in range(0, len(query), 8):
            ids = query[start:start + 8]
            b = make_batch(ids, arrays, n_items=n_items, max_context=actor.max_context, device=device)
            logp = F.log_softmax(actor(b["tokens"], b["padding_mask"]).float(), dim=-1)
            sl = slice(start, start + len(ids))
            p = logp.exp(); lp0 = logp0_all[sl]; reward_slice = reward_all[sl]
            logq = lp0 + reward_slice / float(cfg["beta"]) - logz_all[sl, None]
            exp_reward.append((p * reward_slice).sum(dim=-1))
            kl_base.append((p * (logp - lp0)).sum(dim=-1))
            kl_q.append((p * (logp - logq)).sum(dim=-1))
        er = torch.cat(exp_reward); kb = torch.cat(kl_base); kq = torch.cat(kl_q)
        gap = q_obj - (er - float(cfg["beta"]) * kb)
        per_actor[arm] = {"expected_frozen_rm": float(er.mean()), "kl_policy_to_base": float(kb.mean()),
                          "kl_policy_to_exact_q": float(kq.mean()), "q_objective_gap": float(gap.mean()),
                          "identity_error_abs_mean": float((gap - float(cfg["beta"]) * kq).abs().mean())}
        per_query[f"{arm}_kl_policy_to_q"] = kq.cpu().numpy().astype(np.float32)
        per_query[f"{arm}_q_objective_gap"] = gap.cpu().numpy().astype(np.float32)
        del actor
    np.savez_compressed(output / "exact-q-gaps-per-query.npz", **per_query)
    result = {"scope": "exact optimizer comparator for same one-step frozen RM/reference/beta only",
              "beta": float(cfg["beta"]), "catalog_size": n_items, "query_count": len(query),
              "supported_query_count": int(np.count_nonzero(arrays["item_token"][query] > 0)),
              "query_index_sha256": q_sha, "base_checkpoint_sha256": sha256(base_path),
              "combined_rm_sha256": sha256(reward_path),
              "exact_q_objective_mean": float(q_obj.mean()),
              "p0_expected_frozen_rm_mean": float(p0_expected_reward.mean()),
              "p0_kl_to_exact_q_mean": float(p0_kl_to_q.mean()),
              "p0_exact_q_objective_gap_mean": float(p0_gap.mean()),
              "p0_gap_identity_error_abs_mean": float((p0_gap - beta * p0_kl_to_q).abs().mean()),
              "actors": per_actor}
    (output / "exact-q-gaps.json").write_text(json.dumps(result, indent=2) + "\n")
    del base, logp0_all, reward_all, logz_all
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def behavior_audit(seed: int, cfg: dict, data: Path, base_path: Path, output: Path,
                   arrays, valid_idx, device) -> dict:
    """Use the existing common-uniform sampler; save IDs and aggregate behavior only."""
    n_items = int(cfg["n_items"])
    raw_ids = arrays["candidate_raw_item_id"]
    uniforms = np.random.default_rng(1957).random((len(valid_idx), 16), dtype=np.float32)
    arms = tuple(cfg.get("arms", ("ce", "context_reinforce", "rloo8")))
    out_arrays: dict[str, np.ndarray] = {}
    records = {}
    policies = [("parent", base_path)] + [
        (arm, output / arm / f"step-{int(cfg['steps'])}.pt")
        for arm in arms]
    for name, checkpoint in policies:
        model, _, _ = load_policy(checkpoint, device); model.eval()
        top_ids, sample_ids = [], []
        entropies, top_ranks, sample_ranks, top_repeats, sample_repeats = [], [], [], [], []
        top_counts = np.zeros(n_items, dtype=np.int64); sample_counts = np.zeros(n_items, dtype=np.int64)
        offset = 0
        for ids in batches(valid_idx, 128):
            b = make_batch(ids, arrays, n_items=n_items, max_context=model.max_context, device=device)
            probs = torch.softmax(model(b["tokens"], b["padding_mask"]).float(), dim=-1)
            u = torch.as_tensor(uniforms[offset:offset + len(ids)], device=device)
            actions = sample_action_classes(probs, u)
            offset += len(ids)
            top = probs.argmax(dim=-1)
            top_np = top.cpu().numpy().astype(np.uint32)
            sampled_np = actions.cpu().numpy().astype(np.uint32)
            top_ids.append(raw_ids[top_np].astype(np.uint32)); sample_ids.append(raw_ids[sampled_np].astype(np.uint32))
            top_counts += np.bincount(top_np, minlength=n_items)
            sample_counts += np.bincount(sampled_np.reshape(-1), minlength=n_items)
            hist = b["tokens"]
            top_repeats.extend((hist.eq(top[:, None] + 1).any(dim=1)).cpu().numpy().tolist())
            sample_repeats.extend(hist[:, None, :].eq(actions[:, :, None] + 1).any(dim=2).cpu().numpy().reshape(-1).tolist())
            entropies.extend((-(probs * torch.log(probs.clamp_min(1e-30))).sum(dim=-1)).cpu().numpy().tolist())
            top_ranks.extend((arrays["candidate_popularity_rank"][top_np] + 1).tolist())
            sample_ranks.extend((arrays["candidate_popularity_rank"][sampled_np] + 1).reshape(-1).tolist())
        top_arr = np.concatenate(top_ids); sample_arr = np.concatenate(sample_ids)
        out_arrays[f"{name}_top1_raw_item_id"] = top_arr
        out_arrays[f"{name}_sampled_raw_item_ids"] = sample_arr
        records[name] = {"entropy_nats_mean": float(np.mean(entropies)),
                         "top1_unique_items": int(np.count_nonzero(top_counts)),
                         "top1_max_item_share": float(top_counts.max() / max(1, top_counts.sum())),
                         "sample_unique_items": int(np.count_nonzero(sample_counts)),
                         "sample_max_item_share": float(sample_counts.max() / max(1, sample_counts.sum())),
                         "top1_history_repeat_fraction": float(np.mean(top_repeats)),
                         "sample_history_repeat_fraction": float(np.mean(sample_repeats)),
                         "top1_mean_catalog_rank": float(np.mean(top_ranks)),
                         "sample_mean_catalog_rank": float(np.mean(sample_ranks))}
        del model
    np.savez_compressed(output / "behavior-action-ids.npz", **out_arrays)
    result = {"scope": "same fixed seed-991 validation rows; common-uniform model action samples, not actual exposure/response",
              "seed": seed, "query_count": len(valid_idx), "actions_per_query": 16,
              "action_uniform_seed": 1957, "models": records}
    (output / "behavior-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def train_arm(arm: str, seed: int, args, cfg, arrays, train_idx, valid_idx,
              base_path: Path, reward, history_rm, durations, device,
              monitor: LiveMonitor, run_dir: Path, *, stop_at: int | None = None,
              resume_state: Path | None = None) -> dict:
    seed_all(seed)
    policy, model_cfg, base_payload = load_policy(base_path, device)
    policy.eval()
    reference, _, _ = load_policy(base_path, device)
    reference.eval()
    for p in reference.parameters(): p.requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    permutation = np.random.default_rng(seed).permutation(train_idx)
    cursor = 0
    batch_size = int(cfg["batch_size"])
    beta = float(cfg["beta"])
    arm_dir = run_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=resume_state is not None)
    metrics_path = arm_dir / "metrics.jsonl"
    state = torch.load(resume_state, map_location="cpu", weights_only=False) if resume_state else None
    if state is not None:
        policy.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        cursor = int(state["cursor"])
        train_seconds = float(state["train_seconds"])
        validation_seconds = float(state["validation_seconds"])
        prior_elapsed_seconds = float(state["elapsed_seconds"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(state["cuda_rng"], device=device)
        log_mode = "a"
    else:
        start_step = 0
        train_seconds = validation_seconds = prior_elapsed_seconds = 0.0
        log_mode = "x"
    best_ndcg = -math.inf
    started = time.monotonic()
    step_wall_seconds: list[float] = []
    monitor.log(stage=f"arm_start_{arm}", step=0, metrics={"step": 0})

    target_step = int(stop_at if stop_at is not None else cfg["steps"])
    with metrics_path.open(log_mode, encoding="utf-8") as stream:
        for step in range(start_step + 1, target_step + 1):
            if cursor + batch_size > len(permutation):
                permutation = np.random.default_rng(seed + step).permutation(train_idx)
                cursor = 0
            ids = permutation[cursor:cursor + batch_size]; cursor += batch_size
            batch = make_batch(ids, arrays, n_items=policy.n_items,
                               max_context=policy.max_context, device=device)
            step_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            logits = policy(batch["tokens"], batch["padding_mask"]) / float(cfg["temperature"])

            if arm == "ce":
                loss = F.cross_entropy(logits[batch["supported"]], batch["target"][batch["supported"]])
                train_loss = float(loss.detach())
                reward_mean = float("nan"); adv_mean = float("nan"); kl_value = float("nan")
            else:
                with torch.no_grad():
                    ref_logits = reference(batch["tokens"], batch["padding_mask"]) / float(cfg["temperature"])
                logp = torch.log_softmax(logits, dim=-1)
                ref_logp = torch.log_softmax(ref_logits, dim=-1)
                dist = Categorical(logits=logits)
                if arm == "context_reinforce":
                    action = dist.sample()
                    action_logp = dist.log_prob(action)
                    action_duration = torch.as_tensor(durations, device=device)[action]
                    rank = torch.as_tensor(arrays["candidate_popularity_rank"][action.detach().cpu().numpy()].astype(np.int64), device=device)
                    with torch.no_grad():
                        reward_value = reward(batch["rm_history"], action, action_duration, rank)
                        baseline = frozen_history_baseline(history_rm, batch["rm_history"])
                    advantage = (reward_value - baseline).detach()
                    reinforce = -(advantage * action_logp).mean()
                    action_draws_per_context = 1
                    reward_mean = float(reward_value.mean()); adv_mean = float(advantage.mean())
                elif arm in {"rloo8", "context_reinforce8"}:
                    group_size = int(cfg["rloo_group_size"])
                    actions = dist.sample((group_size,)).transpose(0, 1).contiguous()
                    action_logp = dist.log_prob(actions.transpose(0, 1)).transpose(0, 1)
                    flat_actions = actions.reshape(-1)
                    repeated_hist = batch["rm_history"].repeat_interleave(group_size, dim=0)
                    action_duration = torch.as_tensor(durations, device=device)[flat_actions]
                    rank = torch.as_tensor(arrays["candidate_popularity_rank"][flat_actions.detach().cpu().numpy()].astype(np.int64), device=device)
                    with torch.no_grad():
                        reward_value = reward(repeated_hist, flat_actions, action_duration, rank).reshape(batch_size, group_size)
                    if arm == "rloo8":
                        # Leave-one-out baseline, exactly G/(G-1)*(r_i-mean(r)); no reward normalization.
                        advantage = (group_size / (group_size - 1.0) *
                                     (reward_value - reward_value.mean(dim=1, keepdim=True))).detach()
                    else:
                        # Action-independent frozen history score; average eight iid score-function draws.
                        with torch.no_grad():
                            baseline = frozen_history_baseline(history_rm, batch["rm_history"])
                        if baseline.shape != (batch_size,):
                            raise ValueError("context baseline must provide one scalar per context")
                        advantage = (reward_value - baseline[:, None]).detach()
                    reinforce = -(advantage * action_logp).mean()
                    action_draws_per_context = group_size
                    reward_mean = float(reward_value.mean()); adv_mean = float(advantage.mean())
                else:
                    raise ValueError(f"unknown arm {arm}")
                kl_per_query = (logp.exp() * (logp - ref_logp)).sum(dim=-1)
                kl_value = float(kl_per_query.mean().detach())
                loss = reinforce + beta * kl_per_query.mean()
                train_loss = float(loss.detach())

            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss arm={arm} step={step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float(cfg["gradient_clip_norm"]))
            if not torch.isfinite(grad_norm): raise FloatingPointError(f"non-finite gradient arm={arm} step={step}")
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_elapsed = time.monotonic() - step_started
            train_seconds += step_elapsed
            if stop_at is not None:
                step_wall_seconds.append(step_elapsed)
            if step in set(map(int, cfg["progress_steps"])):
                row = {"event": "train_progress", "seed": seed, "arm": arm, "step": step,
                       "action_draws_per_context": action_draws_per_context if arm != "ce" else 0,
                       "sampled_action_count": batch_size * action_draws_per_context if arm != "ce" else 0,
                       "train_loss": train_loss, "sampled_proxy_reward": reward_mean,
                       "advantage_mean": adv_mean, "kl_policy_to_base": kl_value,
                       "elapsed_seconds": time.monotonic() - started}
                stream.write(json.dumps(row, allow_nan=True) + "\n"); stream.flush()
                monitor.log(stage=f"train_{arm}", step=step, metrics={"step": step, "train_loss": train_loss,
                    "loss": train_loss, "stage_elapsed_seconds": row["elapsed_seconds"]})
                print(json.dumps(row, allow_nan=True), flush=True)

            if step in set(map(int, cfg["evaluation_steps"])):
                eval_started = time.monotonic()
                policy.eval()
                val = eval_policy(policy, valid_idx, arrays, batch_size=batch_size, device=device)
                proxy = eval_proxy(policy, reference, reward, valid_idx, arrays, durations,
                                   batch_size=batch_size, max_context=policy.max_context,
                                   temperature=float(cfg["temperature"]), device=device)
                ndcg = float(val["all"]["ndcg@100"])
                row = {"event": "validation", "seed": seed, "arm": arm, "step": step,
                       "validation": val, "policy_diagnostics": proxy,
                       "elapsed_seconds": time.monotonic() - started}
                stream.write(json.dumps(row) + "\n"); stream.flush()
                monitor.log(stage=f"validation_{arm}", step=step, metrics={"step": step,
                    "valid_ndcg_at_100": ndcg, "valid_recall_at_100": float(val["all"]["recall@100"]),
                    "valid_organic_ndcg_at_100": float(val["organic"]["ndcg@100"]),
                    "valid_recommended_ndcg_at_100": float(val["recommended"]["ndcg@100"]),
                    "stage_elapsed_seconds": row["elapsed_seconds"]})
                print(json.dumps(row), flush=True)
                validation_seconds += time.monotonic() - eval_started
                best_ndcg = max(best_ndcg, ndcg)
                if step in set(map(int, cfg["checkpoint_steps"])):
                    path = arm_dir / f"step-{step}.pt"
                    torch.save({"state_dict": policy.state_dict(), "config": model_cfg,
                                "step": step, "arm": arm, "seed": seed, "beta": beta,
                                "parent_sha256": sha256(base_path)}, path)
                    if bool(cfg.get("wandb_enabled", True)):
                        uploaded = artifact(monitor, path, f"rl-grouped-{arm}-seed{seed}-step{step}",
                                            {"arm": arm, "seed": seed, "step": step,
                                             "parent_sha256": sha256(base_path)})
                        if not uploaded:
                            raise RuntimeError(f"private W&B checkpoint upload failed at {arm}/{step}")
                    if step < int(cfg["steps"]):
                        path.unlink()
                policy.train(False)
        elapsed = prior_elapsed_seconds + time.monotonic() - started
        if target_step < int(cfg["steps"]):
            state_path = arm_dir / "profile-state.pt"
            torch.save({"model": policy.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": target_step, "cursor": cursor,
                        "train_seconds": train_seconds, "validation_seconds": validation_seconds,
                        "elapsed_seconds": elapsed, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}, state_path)
            return {"arm": arm, "seed": seed, "profile_steps": target_step,
                    "train_seconds": train_seconds, "validation_seconds": validation_seconds,
                    "profile_step_duration_p75_seconds": float(np.percentile(step_wall_seconds, 75)),
                    "gpu_peak_allocated_mib": (torch.cuda.max_memory_allocated(device) / 2**20
                                                if device.type == "cuda" else None),
                    "gpu_peak_reserved_mib": (torch.cuda.max_memory_reserved(device) / 2**20
                                               if device.type == "cuda" else None),
                    "elapsed_seconds": elapsed, "resume_state": str(state_path)}
    final_path = arm_dir / f"step-{int(cfg['steps'])}.pt"
    return {"arm": arm, "seed": seed, "steps": int(cfg["steps"]),
            "training_examples_seen": int(cfg["steps"]) * batch_size,
            "elapsed_seconds": elapsed, "train_seconds": train_seconds,
            "validation_seconds": validation_seconds, "best_validation_ndcg100_descriptive": best_ndcg,
            "final_checkpoint": str(final_path) if final_path.is_file() else None,
            "final_checkpoint_sha256": sha256(final_path) if final_path.is_file() else None}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--aux", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True, choices=(17, 29))
    p.add_argument("--device", default="auto")
    p.add_argument("--preflight", action="store_true", help="one real CPU update per arm; no checkpoint/test access")
    p.add_argument("--dose-decision", type=Path, default=None,
                   help="shared wrapper decision written after both seeds complete measured profiles")
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    device = torch.device("cpu") if args.preflight else device_from(args.device)
    if not args.preflight and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("each seed worker requires exactly one visible CUDA device")
    if args.seed not in cfg["seeds"]:
        raise ValueError("unregistered initialization seed")
    data_manifest = args.data / "manifest.json"
    aux_manifest = args.aux / "manifest.json"
    if sha256(data_manifest) != cfg["feature_manifest_sha256"] or sha256(aux_manifest) != cfg["aux_manifest_sha256"]:
        raise ValueError("input manifest hash mismatch")
    base_path = args.aux / str(cfg.get("base_checkpoint_filename", "checkpoint_gpt_seed43_subset.pt"))
    reward_path = args.aux / "checkpoint_combined_seed17_primary.pt"
    history_path = args.aux / "checkpoint_history_seed17.pt"
    for path, key in ((base_path, "base_checkpoint_sha256"), (reward_path, "combined_rm_checkpoint_sha256"),
                      (history_path, "history_rm_checkpoint_sha256")):
        if sha256(path) != cfg[key]: raise ValueError(f"checkpoint SHA mismatch: {key}")
    arrays = open_event_arrays(args.data)
    train_idx = np.load(args.data / "train_indices.npy", mmap_mode="r")
    if len(train_idx) != 4_110_754 or not np.all(arrays["split"][train_idx] == 0):
        raise ValueError("training anchor count/split mismatch")
    valid_all = np.load(args.data / "valid_indices.npy", mmap_mode="r")
    valid_idx = (np.random.default_rng(int(cfg["validation_seed"])).choice(
        valid_all, size=int(cfg["validation_event_count"]), replace=False)
        if len(valid_all) > int(cfg["validation_event_count"]) else valid_all)
    output = args.output / f"seed-{args.seed}"
    output.mkdir(parents=True, exist_ok=False)
    job_id = str(cfg.get("job_id", "rl-grouped-20261025"))
    run_prefix = str(cfg.get("run_id_prefix", "tt2-grouped-rl"))
    wandb_enabled = bool(cfg.get("wandb_enabled", True)) and not args.preflight
    monitor = LiveMonitor(output_dir=output, run_id=f"{run_prefix}-seed{args.seed}",
                          project="recsys-tt2", enable_wandb=wandb_enabled,
                          config={"job_id": job_id, "stage": "one_step_rl",
                                  "seed": args.seed, "dataset_id": cfg["feature_dataset_id"],
                                  "dataset_manifest_sha256": cfg["feature_manifest_sha256"],
                                  "parent_checkpoint_sha256": cfg["base_checkpoint_sha256"],
                                  "checkpoint_sha256": cfg["combined_rm_checkpoint_sha256"],
                                  "optimizer_steps": int(cfg["steps"]), "batch_size": int(cfg["batch_size"]),
                                  "anchor_count": int(len(train_idx)), "validation_sample_count": len(valid_idx),
                                  "accelerator": "CPU preflight" if args.preflight else "Kaggle T4",
                                  "visible_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0})
    try:
        if wandb_enabled and getattr(monitor, "_wandb_run", None) is None:
            raise RuntimeError("W&B must be online before production training starts")
        reward, reward_payload = load_combined(reward_path, device, int(cfg["n_items"]))
        history_rm = load_frozen_history_rm(history_path, device=device, expected_n_items=int(cfg["n_items"]), expected_seed=17)
        durations = make_item_durations(train_idx, arrays, int(cfg["n_items"]))
        arms = tuple(cfg.get("arms", ("ce", "context_reinforce", "rloo8")))
        if not arms or len(set(arms)) != len(arms):
            raise ValueError("config arms must be a nonempty unique sequence")
        if args.preflight:
            cfg = {**cfg, "steps": 1, "progress_steps": [1], "evaluation_steps": [], "checkpoint_steps": []}
            results = [train_arm(arm, args.seed, args, cfg, arrays, train_idx, valid_idx,
                                 base_path, reward, history_rm, durations, device, monitor, output)
                       for arm in arms]
            selected_dose = 1
        elif cfg.get("dose_mode") == "fixed":
            selected_dose = int(cfg["steps"])
            if selected_dose <= 0:
                raise ValueError("fixed dose must be positive")
            results = [train_arm(arm, args.seed, args, cfg, arrays, train_idx, valid_idx,
                                 base_path, reward, history_rm, durations, device, monitor, output)
                       for arm in arms]
        else:
            profile_results = [train_arm(arm, args.seed, args, cfg, arrays, train_idx, valid_idx,
                                         base_path, reward, history_rm, durations, device, monitor, output,
                                         stop_at=int(cfg["profile_steps"]))
                               for arm in arms]
            profile_path = output / "profile-summary.json"
            profile_path.write_text(json.dumps({"seed": args.seed, "profile_steps": int(cfg["profile_steps"]),
                                                "arms": profile_results}, indent=2) + "\n")
            if args.dose_decision is None:
                raise ValueError("production run requires a shared two-seed dose decision path")
            wait_started = time.monotonic()
            while not args.dose_decision.is_file() and time.monotonic() - wait_started < 600:
                if device.type == "cuda": torch.cuda.empty_cache()
                time.sleep(2.0)
            if not args.dose_decision.is_file():
                raise TimeoutError("supervisor did not publish measured-dose decision within 600s")
            decision = json.loads(args.dose_decision.read_text())
            selected_dose = decision.get("selected_dose")
            if selected_dose is None:
                (output / "run-result.json").write_text(json.dumps({
                    "mode": "grouped-one-step-rl", "status": "profile_only_no_common_dose_fits",
                    "seed": args.seed, "profile_steps": int(cfg["profile_steps"]),
                    "profile_results": profile_results, "dose_decision": decision,
                    "feature_manifest_sha256": sha256(data_manifest), "aux_manifest_sha256": sha256(aux_manifest),
                }, indent=2) + "\n")
                monitor.log(stage="profile_only_stop", step=int(cfg["profile_steps"]), metrics={
                    "step": int(cfg["profile_steps"]), "stage_elapsed_seconds": time.monotonic() - wait_started})
                monitor.finish(exit_code=0)
                return
            if selected_dose not in cfg["candidate_doses"]:
                raise ValueError("supervisor selected an unregistered update dose")
            cfg = {**cfg, "steps": int(selected_dose),
                   "progress_steps": [int(x) for x in cfg["progress_steps"] if int(x) <= selected_dose],
                   "evaluation_steps": [int(x) for x in cfg["evaluation_steps"] if int(x) <= selected_dose],
                   "checkpoint_steps": [int(selected_dose)]}
            monitor.log(stage="common_dose_selected", step=0, metrics={"steps_total": int(selected_dose)})
            results = []
            profile_by_arm = {row["arm"]: row for row in profile_results}
            for arm in arms:
                state_path = Path(profile_by_arm[arm]["resume_state"])
                result = train_arm(arm, args.seed, args, cfg, arrays, train_idx, valid_idx,
                                   base_path, reward, history_rm, durations, device, monitor, output,
                                   resume_state=state_path)
                state_path.unlink(missing_ok=True)
                results.append(result)

        if not args.preflight and selected_dose > 1:
            monitor.log(stage="exact_q_start", step=selected_dose, metrics={"step": selected_dose})
            q_result = exact_q_gaps(args.seed, cfg, args.data, base_path, reward_path, reward, train_idx,
                                    valid_all, arrays, durations, output, device)
            monitor.log(stage="exact_q_complete", step=selected_dose, metrics={
                "step": selected_dose, "stage_elapsed_seconds": float(q_result["query_count"])})
            behavior_result = behavior_audit(args.seed, cfg, args.data, base_path, output,
                                             arrays, valid_idx, device)
        else:
            q_result = None; behavior_result = None
        (output / "run-result.json").write_text(json.dumps({
            "mode": str(cfg.get("job_id", "grouped-one-step-rl")), "status": "complete",
            "interpretation": "frozen one-step logged-listen proxy; not causal/OPE/long-horizon",
            "seed": args.seed, "preflight": bool(args.preflight), "config": cfg,
            "selected_common_dose": selected_dose,
            "feature_manifest_sha256": sha256(data_manifest), "aux_manifest_sha256": sha256(aux_manifest),
            "base_checkpoint_sha256": sha256(base_path), "combined_rm_checkpoint_sha256": sha256(reward_path),
            "history_rm_checkpoint_sha256": sha256(history_path), "reward_variant": reward_payload.get("variant"),
            "data_provenance": data_provenance(args.data, train_idx, valid_idx), "results": results,
            "exact_q": q_result, "behavior_audit": behavior_result,
        }, indent=2) + "\n")
        monitor.log(stage="complete", step=int(cfg["steps"]), metrics={"step": int(cfg["steps"]),
                     "steps_total": int(cfg["steps"]), "stage_elapsed_seconds": sum(r["elapsed_seconds"] for r in results)})
        monitor.finish(exit_code=0)
    except BaseException:
        monitor.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
