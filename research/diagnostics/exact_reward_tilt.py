#!/usr/bin/env python3
"""Exact full-catalog tilt of a frozen next-item policy by a frozen RM proxy.

For each fixed validation context this computes
    q_beta(i|h) ∝ pi_0(i|h) exp(r_hat(h,i) / beta)
over every class in the selected train-frequency catalog. This is an exact
comparator for the constructed proxy objective, not an estimate of user
satisfaction or a claim about counterfactual outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from recsys_tt2.batching import batches, make_batch, open_event_arrays
from recsys_tt2.model import ConditionalCompletionRM, ItemGPT


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def index_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.uint32).tobytes()).hexdigest()


def load_policy(path: Path, device: torch.device) -> tuple[ItemGPT, dict, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload.get("model_config", payload["config"])
    model = ItemGPT(
        n_items=int(cfg["n_items"]), width=int(cfg["width"]), layers=int(cfg["layers"]),
        heads=int(cfg["heads"]), max_context=int(cfg["max_context"]), dropout=float(cfg["dropout"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, cfg, payload


def load_reward_model(path: Path, device: torch.device, n_items: int) -> tuple[ConditionalCompletionRM, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if int(cfg["n_items"]) != n_items or payload.get("variant") != "combined":
        raise ValueError("reward checkpoint must be the combined RM for the same catalog")
    model = ConditionalCompletionRM(n_items, width=int(cfg["width"]), variant="combined").to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def train_item_durations(train_indices: np.ndarray, arrays: dict[str, np.ndarray], n_items: int) -> np.ndarray:
    """Per-class mean duration from supported train anchors only."""
    tokens = np.asarray(arrays["item_token"][train_indices], dtype=np.int64)
    duration = np.asarray(arrays["track_length_seconds"][train_indices], dtype=np.float64)
    keep = tokens > 0
    classes = tokens[keep] - 1
    sums = np.bincount(classes, weights=duration[keep], minlength=n_items)
    counts = np.bincount(classes, minlength=n_items)
    return np.divide(sums, counts, out=np.full(n_items, 180.0), where=counts > 0).astype(np.float32)


def history_embedding(rm: ConditionalCompletionRM, history: torch.Tensor) -> torch.Tensor:
    """Cache the combined RM's pooled history term once per query."""
    valid = history.ne(-1)
    embedded = rm.item_embedding(history.clamp(min=0)).masked_fill(~valid.unsqueeze(-1), 0.0)
    return embedded.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp(min=1)


def score_catalog_cached(
    rm: ConditionalCompletionRM,
    pooled_history: torch.Tensor,
    durations: torch.Tensor,
    popularity_rank: torch.Tensor,
    *,
    candidate_chunk: int = 4096,
) -> torch.Tensor:
    """Score [query, all catalog items], without recomputing history embeddings."""
    if rm.variant != "combined":
        raise ValueError("cached scoring currently implements only the combined RM")
    batch = len(pooled_history)
    n_items = rm.n_items
    result = torch.empty((batch, n_items), dtype=torch.float32, device=pooled_history.device)
    scale = math.log1p(2500)
    rank_scale = max(1, n_items - 1)
    for start in range(0, n_items, candidate_chunk):
        stop = min(start + candidate_chunk, n_items)
        candidate = torch.arange(start, stop, device=pooled_history.device)
        width = stop - start
        item_feature = rm.item_embedding(candidate + 1).unsqueeze(0).expand(batch, -1, -1)
        history_feature = pooled_history.unsqueeze(1).expand(-1, width, -1)
        duration_feature = (torch.log1p(durations[start:stop].float()) / scale).view(1, width, 1).expand(batch, -1, -1)
        rank_feature = (popularity_rank[start:stop].float() / rank_scale).view(1, width, 1).expand(batch, -1, -1)
        features = torch.cat((history_feature, item_feature, duration_feature, rank_feature), dim=-1)
        result[:, start:stop] = torch.sigmoid(rm.mlp(features.reshape(batch * width, -1))).reshape(batch, width)
    return result


def tilt_distribution(log_pi0: torch.Tensor, reward: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized log q, q, and log Z for reference-relative reward tilt."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    if log_pi0.shape != reward.shape or log_pi0.ndim != 2:
        raise ValueError("log_pi0 and reward must share shape [queries, catalog]")
    unnormalized = log_pi0.float() + reward.float() / float(beta)
    log_z = torch.logsumexp(unnormalized, dim=-1, keepdim=True)
    log_q = unnormalized - log_z
    return log_q, log_q.exp(), log_z.squeeze(-1)


def self_test() -> None:
    """Small CPU checks against direct formulas and ordinary model scoring."""
    torch.manual_seed(11)
    for beta in (0.01, 0.1):
        base = torch.tensor([[0.0, -0.2, -1.0, -2.0]], dtype=torch.float32)
        log_pi0 = F.log_softmax(base, dim=-1)
        reward = torch.tensor([[0.2, 0.7, 0.1, 0.9]], dtype=torch.float32)
        log_q, q, log_z = tilt_distribution(log_pi0, reward, beta)
        brute = torch.softmax(log_pi0.double() + reward.double() / beta, dim=-1).float()
        assert torch.allclose(q, brute, atol=2e-6, rtol=2e-6)
        assert torch.allclose(q.sum(-1), torch.ones(1), atol=1e-6)
        assert torch.allclose(torch.logsumexp(log_q, -1), torch.zeros(1), atol=1e-6)
        assert torch.isfinite(log_z).all()
        # A history-only score is constant across candidate actions, so its
        # tilted policy must equal the reference exactly (up to roundoff).
        constant_reward = torch.full_like(reward, 0.37)
        _, q_history_only, _ = tilt_distribution(log_pi0, constant_reward, beta)
        assert torch.allclose(q_history_only, log_pi0.exp(), atol=1e-6, rtol=1e-6)

    # Confirm the cached-feature path exactly matches the public RM forward.
    rm = ConditionalCompletionRM(n_items=7, width=4, variant="combined").eval()
    history = torch.tensor([[1, 4, -1], [0, -1, -1]], dtype=torch.long)
    durations = torch.tensor([31, 100, 200, 61, 90, 40, 80], dtype=torch.float32)
    ranks = torch.tensor([3, 0, 5, 1, 4, 2, 6], dtype=torch.long)
    pooled = history_embedding(rm, history)
    cached = score_catalog_cached(rm, pooled, durations, ranks, candidate_chunk=3)
    candidates = torch.arange(7).repeat(2)
    expanded_history = history.repeat_interleave(7, dim=0)
    direct = rm(expanded_history, candidates, durations.repeat(2), ranks.repeat(2)).reshape(2, 7)
    assert torch.allclose(cached, direct, atol=1e-7, rtol=1e-6)
    print("exact_reward_tilt self-test passed: brute-force tilt, normalization, cached RM scoring")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--rm-checkpoint", type=Path)
    parser.add_argument("--policy-checkpoint", type=Path, help="optional learned policy for KL/q and objective-gap audit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--query-count", type=int, default=1024)
    parser.add_argument("--eval-event-count", type=int, default=10000)
    parser.add_argument("--selection-seed", type=int, default=991)
    parser.add_argument("--query-seed", type=int, default=20261020)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.01, 0.1])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--candidate-chunk", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.data, args.base_checkpoint, args.rm_checkpoint, args.output)):
        parser.error("--data, --base-checkpoint, --rm-checkpoint, and --output are required")
    if args.temperature <= 0 or args.query_count < 1 or args.eval_event_count < args.query_count:
        parser.error("temperature/query counts must be positive and eval-event-count >= query-count")
    if any(beta <= 0 for beta in args.betas):
        parser.error("all beta values must be positive")

    device = torch.device(args.device)
    arrays = open_event_arrays(args.data)
    valid = np.load(args.data / "valid_indices.npy", mmap_mode="r")
    if args.eval_event_count > len(valid):
        raise ValueError("requested fixed validation sample exceeds available validation rows")
    fixed_validation = np.random.default_rng(args.selection_seed).choice(
        valid, size=args.eval_event_count, replace=False
    ) if args.eval_event_count and len(valid) > args.eval_event_count else np.asarray(valid)
    # Subsample the label-independent fixed query set; labels are used only after q is computed.
    query_indices = np.random.default_rng(args.query_seed).choice(
        fixed_validation, size=args.query_count, replace=False
    )
    n_items = len(np.load(args.data / "candidate_item_ids.npy", mmap_mode="r"))
    base, base_cfg, base_payload = load_policy(args.base_checkpoint, device)
    if base.n_items != n_items:
        raise ValueError("base policy catalog size differs from dataset catalog")
    rm, rm_payload = load_reward_model(args.rm_checkpoint, device, n_items)
    learned = None
    learned_payload = None
    if args.policy_checkpoint:
        learned, _, learned_payload = load_policy(args.policy_checkpoint, device)
        if learned.n_items != n_items:
            raise ValueError("learned policy catalog size differs from dataset catalog")

    train_indices = np.load(args.data / "train_indices.npy", mmap_mode="r")
    if not np.all(arrays["split"][train_indices] == 0):
        raise ValueError("train_indices include non-train events")
    durations_np = train_item_durations(train_indices, arrays, n_items)
    durations = torch.as_tensor(durations_np, dtype=torch.float32, device=device)
    ranks = torch.as_tensor(np.asarray(arrays["candidate_popularity_rank"], dtype=np.int64), device=device)
    raw_ids = np.asarray(arrays["candidate_raw_item_id"], dtype=np.int64)

    fields: dict[str, list[np.ndarray]] = {k: [] for k in (
        "uid", "event_index", "organic", "supported", "target_class", "target_token", "label",
        "base_target_logp", "rm_target", "base_rm_expected",
    )}
    for beta in args.betas:
        key = f"beta_{beta:g}"
        fields.update({f"{key}_{k}": [] for k in (
            "target_logq", "target_prob", "reciprocal_rank", "hit10", "hit50", "hit100", "ndcg10", "ndcg50", "ndcg100",
            "expected_rm", "kl_q_to_base", "entropy", "expected_popularity_rank", "expected_duration",
            "repeat_mass", "top1_class", "top1_raw_item_id", "top1_rm",
        )})
        if learned is not None:
            fields.update({f"{key}_{k}": [] for k in ("kl_policy_to_q", "policy_expected_rm", "policy_objective", "q_objective_gap")})

    processed_count = 0
    with torch.inference_mode():
        for id_batch in batches(query_indices, args.batch_size):
            processed_count += len(id_batch)
            b = make_batch(id_batch, arrays, n_items=n_items, max_context=base.max_context, device=device)
            base_logits = base(b["tokens"], b["padding_mask"]) / args.temperature
            log_pi0 = F.log_softmax(base_logits.float(), dim=-1)
            learned_logits = None
            if learned is not None:
                learned_logits = F.log_softmax(learned(b["tokens"], b["padding_mask"]).float() / args.temperature, dim=-1)
            pooled = history_embedding(rm, b["rm_history"])
            reward = score_catalog_cached(rm, pooled, durations, ranks, candidate_chunk=args.candidate_chunk)
            if not torch.isfinite(reward).all():
                raise FloatingPointError("nonfinite reward prediction over catalog")

            support = b["supported"].cpu().numpy().astype(bool)
            target = b["target"].cpu().numpy().astype(np.int64)
            labels = b["reward"].cpu().numpy().astype(np.float32)
            uid = b["uid"].cpu().numpy().astype(np.int64)
            organic = b["organic"].cpu().numpy().astype(bool)
            token = np.asarray(arrays["item_token"][id_batch], dtype=np.uint32)
            event_id = np.asarray(id_batch, dtype=np.uint64)
            target_safe = np.maximum(target, 0)
            row = torch.arange(len(id_batch), device=device)
            target_logp = log_pi0[row, torch.as_tensor(target_safe, device=device)].cpu().numpy()
            target_reward = reward[row, torch.as_tensor(target_safe, device=device)].cpu().numpy()
            base_probs = log_pi0.exp()
            base_expected_reward = (base_probs * reward).sum(dim=-1).cpu().numpy()
            repeat_mask = torch.zeros((len(id_batch), n_items), dtype=torch.bool, device=device)
            hist_np = b["rm_history"].cpu().numpy()
            for q, hist in enumerate(hist_np):
                classes = hist[(hist >= 1) & (hist <= n_items)] - 1
                if len(classes):
                    repeat_mask[q, torch.as_tensor(classes, dtype=torch.long, device=device)] = True
            fields["uid"].append(uid)
            fields["event_index"].append(event_id)
            fields["organic"].append(organic)
            fields["supported"].append(support)
            fields["target_class"].append(np.where(support, target, -1).astype(np.int32))
            fields["target_token"].append(token)
            fields["label"].append(labels)
            fields["base_target_logp"].append(np.where(support, target_logp, np.nan).astype(np.float32))
            fields["rm_target"].append(np.where(support, target_reward, np.nan).astype(np.float32))
            fields["base_rm_expected"].append(base_expected_reward.astype(np.float32))

            for beta in args.betas:
                name = f"beta_{beta:g}"
                log_q, q, _ = tilt_distribution(log_pi0, reward, beta)
                entropy = -(q * log_q).sum(dim=-1)
                kl = (q * (log_q - log_pi0)).sum(dim=-1)
                expected_rm = (q * reward).sum(dim=-1)
                expected_rank = (q * ranks.float().view(1, -1)).sum(dim=-1) + 1.0
                expected_duration = (q * durations.view(1, -1)).sum(dim=-1)
                repeat_mass = (q * repeat_mask.float()).sum(dim=-1)
                topk = torch.topk(log_q, k=min(100, n_items), dim=-1).indices.cpu().numpy()
                log_q_cpu = log_q.cpu().numpy()
                target_logq = log_q[row, torch.as_tensor(target_safe, device=device)].cpu().numpy()
                rrank = np.zeros(len(id_batch), dtype=np.float32)
                hit_ndcg: dict[int, tuple[np.ndarray, np.ndarray]] = {}
                for qid in range(len(id_batch)):
                    if support[qid]:
                        rank_pos = int(np.flatnonzero(topk[qid] == target[qid])[0]) + 1 if target[qid] in topk[qid] else int(np.sum(log_q_cpu[qid] > log_q_cpu[qid, target[qid]])) + 1
                        rrank[qid] = 1.0 / rank_pos
                for k in (10, 50, 100):
                    hit = np.zeros(len(id_batch), dtype=np.float32)
                    hit[support] = np.any(topk[support, :k] == target[support, None], axis=1).astype(np.float32)
                    ndcg = np.zeros(len(id_batch), dtype=np.float32)
                    ndcg[support] = hit[support] / np.log2(rrank[support] ** -1 + 1.0)
                    hit_ndcg[k] = (hit, ndcg)
                top1 = topk[:, 0]
                fields[f"{name}_target_logq"].append(np.where(support, target_logq, np.nan).astype(np.float32))
                fields[f"{name}_target_prob"].append(np.where(support, np.exp(target_logq), np.nan).astype(np.float32))
                fields[f"{name}_reciprocal_rank"].append(rrank)
                for k, (hit, ndcg) in hit_ndcg.items():
                    fields[f"{name}_hit{k}"].append(hit)
                    fields[f"{name}_ndcg{k}"].append(ndcg)
                fields[f"{name}_expected_rm"].append(expected_rm.cpu().numpy().astype(np.float32))
                fields[f"{name}_kl_q_to_base"].append(kl.cpu().numpy().astype(np.float32))
                fields[f"{name}_entropy"].append(entropy.cpu().numpy().astype(np.float32))
                fields[f"{name}_expected_popularity_rank"].append(expected_rank.cpu().numpy().astype(np.float32))
                fields[f"{name}_expected_duration"].append(expected_duration.cpu().numpy().astype(np.float32))
                fields[f"{name}_repeat_mass"].append(repeat_mass.cpu().numpy().astype(np.float32))
                fields[f"{name}_top1_class"].append(top1.astype(np.int32))
                fields[f"{name}_top1_raw_item_id"].append(raw_ids[top1].astype(np.int64))
                fields[f"{name}_top1_rm"].append(reward[row, torch.as_tensor(top1, device=device)].cpu().numpy().astype(np.float32))
                if learned_logits is not None:
                    policy_p = learned_logits.exp()
                    policy_kl = (policy_p * (learned_logits - log_q)).sum(dim=-1)
                    policy_reward = (policy_p * reward).sum(dim=-1)
                    policy_objective = policy_reward - beta * (policy_p * (learned_logits - log_pi0)).sum(dim=-1)
                    q_objective = expected_rm - beta * kl
                    for suffix, value in (
                        ("kl_policy_to_q", policy_kl), ("policy_expected_rm", policy_reward),
                        ("policy_objective", policy_objective), ("q_objective_gap", q_objective - policy_objective),
                    ):
                        fields[f"{name}_{suffix}"].append(value.cpu().numpy().astype(np.float32))
            print(f"processed {processed_count}/{args.query_count} queries", flush=True)

    output = {key: np.concatenate(chunks) for key, chunks in fields.items()}
    if len(np.unique(output["event_index"])) != len(output["event_index"]):
        raise ValueError("duplicate sampled validation events")
    if len(output["uid"]) != args.query_count:
        raise ValueError("query output count differs from requested count")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)

    aggregates = {}
    for beta in args.betas:
        name = f"beta_{beta:g}"
        aggregates[name] = {}
        for k in (10, 50, 100):
            aggregates[name][f"recall@{k}"] = float(output[f"{name}_hit{k}"].mean())
            aggregates[name][f"ndcg@{k}"] = float(output[f"{name}_ndcg{k}"].mean())
        for metric in ("expected_rm", "kl_q_to_base", "entropy", "expected_popularity_rank", "expected_duration", "repeat_mass", "top1_rm"):
            aggregates[name][metric] = float(output[f"{name}_{metric}"].mean())
        if learned is not None:
            for metric in ("kl_policy_to_q", "policy_expected_rm", "policy_objective", "q_objective_gap"):
                aggregates[name][metric] = float(output[f"{name}_{metric}"].mean())
    manifest = {
        "mode": "exact-full-catalog-reference-relative-RM-tilt",
        "interpretation": "exact optimizer comparator for frozen proxy objective only; not user utility or counterfactual evaluation",
        "data_path": str(args.data.resolve()), "data_manifest_sha256": sha256(args.data / "manifest.json"),
        "base_checkpoint": str(args.base_checkpoint), "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "base_step": base_payload.get("step"), "base_config": base_cfg,
        "rm_checkpoint": str(args.rm_checkpoint), "rm_checkpoint_sha256": sha256(args.rm_checkpoint),
        "rm_step": rm_payload.get("step"), "catalog_size": n_items,
        "validation_sample_size": int(len(fixed_validation)), "validation_selection_seed": args.selection_seed,
        "validation_index_sha256": index_hash(fixed_validation), "query_count": int(len(query_indices)),
        "query_seed": args.query_seed, "query_index_sha256": index_hash(query_indices),
        "temperature": args.temperature, "betas": args.betas, "batch_size": args.batch_size,
        "candidate_chunk": args.candidate_chunk, "device": str(device),
        "train_item_duration_source": "mean duration by class from train_indices only",
        "policy_checkpoint": str(args.policy_checkpoint) if args.policy_checkpoint else None,
        "policy_checkpoint_sha256": sha256(args.policy_checkpoint) if args.policy_checkpoint else None,
        "policy_step": learned_payload.get("step") if learned_payload else None,
        "supported_query_count": int(output["supported"].sum()), "aggregate": aggregates,
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
