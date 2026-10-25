#!/usr/bin/env python3
"""Paired fixed-uniform action-value diagnostics on held-out validation contexts.

All scores are learned/associative conditional-listen proxies. This is not OPE:
there are no logged action propensities or no-listen exposure outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from recsys_tt2.batching import make_batch, open_event_arrays
from recsys_tt2.catalog import build_candidate_encoding
from recsys_tt2.model import ConditionalCompletionRM, ItemGPT


def mix_distributions(gpt_probs: torch.Tensor, history_tokens: torch.Tensor, n_items: int,
                      history_weight: float = 0.5) -> torch.Tensor:
    """Mix GPT probabilities with last-64-known-item frequency distribution."""
    if not 0.0 <= history_weight <= 1.0:
        raise ValueError("history_weight must be in [0, 1]")
    history_tokens = history_tokens[:, -64:]
    known = history_tokens.ge(1) & history_tokens.le(n_items)
    classes = (history_tokens - 1).clamp(min=0, max=n_items - 1)
    counts = torch.zeros_like(gpt_probs)
    counts.scatter_add_(1, classes, known.to(gpt_probs.dtype))
    totals = counts.sum(dim=1, keepdim=True)
    recent = counts / totals.clamp_min(1.0)
    mixed = (1.0 - history_weight) * gpt_probs + history_weight * recent
    return torch.where(totals > 0, mixed, gpt_probs)


def sample_action_classes(probs: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    """Inverse-CDF draws using supplied common random numbers."""
    return torch.searchsorted(probs.cumsum(dim=-1), uniforms, right=False).clamp_(max=probs.shape[-1] - 1)


def export_action_ids(actions: torch.Tensor, class_to_raw_item_id: np.ndarray | torch.Tensor,
                      top1_classes: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return compact catalog class IDs and mapped item IDs for samples and top-1."""
    action_classes = actions.detach().cpu().numpy().astype(np.uint32, copy=False)
    top_classes = top1_classes.detach().cpu().numpy().astype(np.uint32, copy=False)
    item_map = np.asarray(class_to_raw_item_id)
    if action_classes.size and int(action_classes.max()) >= len(item_map):
        raise ValueError("sampled class ID is outside the class-to-raw-item map")
    if top_classes.size and int(top_classes.max()) >= len(item_map):
        raise ValueError("top-1 class ID is outside the class-to-raw-item map")
    return action_classes, item_map[action_classes], top_classes, item_map[top_classes]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_policy(path: Path, device: torch.device) -> ItemGPT:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload.get("model_config", payload["config"])
    model = ItemGPT(n_items=int(config["n_items"]), width=int(config["width"]),
                    layers=int(config["layers"]), heads=int(config["heads"]),
                    max_context=int(config["max_context"]), dropout=float(config["dropout"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def load_rm(path: Path, device: torch.device) -> ConditionalCompletionRM:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload["config"]
    model = ConditionalCompletionRM(int(config["n_items"]), width=int(config["width"]),
                                    variant=payload["variant"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def train_controls(data: Path, arrays: dict[str, np.ndarray], n_items: int):
    train_idx = np.load(data / "train_indices.npy", mmap_mode="r")
    if not np.all(arrays["split"][train_idx] == 0):
        raise ValueError("train_indices contains a non-train event")
    token = arrays["item_token"][train_idx].astype(np.int64)
    if np.any(token <= 0):
        raise ValueError("train_indices must contain supported targets only")
    cls = token - 1
    labels = np.minimum(arrays["played_ratio_pct"][train_idx].astype(np.float64) / 100.0, 1.0)
    event_duration = arrays["track_length_seconds"][train_idx].astype(np.float64)
    item_counts = np.bincount(cls, minlength=n_items)
    item_means = np.divide(np.bincount(cls, weights=labels, minlength=n_items), item_counts,
                           out=np.full(n_items, float(labels.mean())), where=item_counts > 0).astype(np.float32)
    item_durations = np.divide(np.bincount(cls, weights=event_duration, minlength=n_items), item_counts,
                               out=np.full(n_items, 180.0), where=item_counts > 0).astype(np.float32)
    duration_edges = np.asarray([0, 60, 120, 180, 240, 300, 420, 600, 900, 1500, 2501], dtype=np.float64)
    duration_bin = np.clip(np.digitize(event_duration, duration_edges[1:-1]), 0, len(duration_edges) - 2)
    duration_sums = np.bincount(duration_bin, weights=labels, minlength=len(duration_edges) - 1)
    duration_counts = np.bincount(duration_bin, minlength=len(duration_edges) - 1)
    duration_means = np.divide(duration_sums, duration_counts, out=np.full(len(duration_counts), float(labels.mean())),
                               where=duration_counts > 0).astype(np.float32)
    return item_means, item_durations, duration_edges, duration_means


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--policy", action="append", required=True, metavar="NAME=CHECKPOINT",
                        help="repeat for checkpoint policies; NAME=fixed-history-mixture adds the fixed 50/50 baseline")
    parser.add_argument("--history-rm", type=Path, required=True)
    parser.add_argument("--item-rm", type=Path, required=True)
    parser.add_argument("--combined-rm", type=Path, required=True)
    parser.add_argument("--audit-combined-rm", type=Path, default=None,
                        help="optional independent-initialization combined RM, used only as another scorer")
    parser.add_argument("--history-mixture-base", type=Path, default=None,
                        help="explicit raw seed43 base checkpoint for the fixed-history-mixture policy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-event-count", type=int, default=10_000)
    parser.add_argument("--selection-seed", type=int, default=991)
    parser.add_argument("--action-seed", type=int, default=1957)
    parser.add_argument("--n-actions", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-action-ids", action="store_true",
                        help="include sampled catalog IDs and top-1 item IDs in the NPZ output")
    args = parser.parse_args()
    if args.n_actions < 2:
        raise ValueError("n-actions must be at least two")
    device = torch.device(args.device)
    arrays = open_event_arrays(args.data)
    valid_indices = np.load(args.data / "valid_indices.npy", mmap_mode="r")
    indices = np.asarray(valid_indices)
    if args.eval_event_count and len(indices) > args.eval_event_count:
        indices = np.random.default_rng(args.selection_seed).choice(indices, size=args.eval_event_count, replace=False)
    n_items = len(np.load(args.data / "candidate_item_ids.npy", mmap_mode="r"))
    candidate_item_ids = np.load(args.data / "candidate_item_ids.npy", mmap_mode="r")
    class_to_raw_item_id = build_candidate_encoding(candidate_item_ids)[2]
    item_means, item_durations, duration_edges, duration_means = train_controls(args.data, arrays, n_items)
    reward_models = {
        "history": load_rm(args.history_rm, device),
        "item": load_rm(args.item_rm, device),
        "combined": load_rm(args.combined_rm, device),
    }
    policy_specs = [arg.split("=", 1) for arg in args.policy]
    if any(len(parts) != 2 for parts in policy_specs):
        raise ValueError("--policy must be NAME=CHECKPOINT or NAME=fixed-history-mixture")
    policy_names = [parts[0] for parts in policy_specs]
    if len(set(policy_names)) != len(policy_names):
        raise ValueError("policy names must be unique")
    has_mixture = any(path_text == "fixed-history-mixture" for _, path_text in policy_specs)
    if has_mixture and args.history_mixture_base is None:
        raise ValueError("--history-mixture-base must identify the raw seed43 checkpoint for the mixture")
    uniforms = np.random.default_rng(args.action_seed).random((len(indices), args.n_actions), dtype=np.float32)
    all_results: dict[str, list[np.ndarray]] = {key: [] for key in (
        "uid", "event_index", "organic", "target_supported", "logged_label", "logged_target_class"
    )}
    names = ["history", "item", "combined"]
    if args.audit_combined_rm is not None:
        reward_models["combined_audit29"] = load_rm(args.audit_combined_rm, device)
        names.append("combined_audit29")
    names.extend(["train_item_mean", "train_duration_mean"])
    for policy_arg in args.policy:
        name, path_text = policy_arg.split("=", 1)
        if not name or name in {"uid", "event_index"}:
            raise ValueError(f"invalid or duplicate policy name {name!r}")
        is_history_mixture = path_text == "fixed-history-mixture"
        path = args.history_mixture_base if is_history_mixture else Path(path_text)
        model = load_policy(path, device)
        if model.n_items != n_items:
            raise ValueError(f"policy {name} item count differs from catalog")
        policy_columns = {f"{name}_{proxy}_{stat}": [] for proxy in names for stat in ("sample_mean", "sample_sd", "top1")}
        history_action_ranges = []
        sample_offset = 0
        with torch.inference_mode():
            for start in range(0, len(indices), args.batch_size):
                batch_indices = indices[start : start + args.batch_size]
                b = make_batch(batch_indices, arrays, n_items=n_items, max_context=model.max_context, device=device)
                logits = model(b["tokens"], b["padding_mask"])
                probs = torch.softmax(logits.float(), dim=-1)
                if is_history_mixture:
                    probs = mix_distributions(probs, b["tokens"], n_items)
                cdf = probs.cumsum(dim=-1)
                u = torch.as_tensor(uniforms[sample_offset : sample_offset + len(batch_indices)], device=device)
                actions = sample_action_classes(probs, u)
                sample_offset += len(batch_indices)
                actions_flat = actions.reshape(-1)
                history_flat = b["rm_history"].repeat_interleave(args.n_actions, dim=0)
                duration_np = item_durations[actions_flat.cpu().numpy()]
                rank_np = arrays["candidate_popularity_rank"][actions_flat.cpu().numpy()].astype(np.int64)
                duration = torch.as_tensor(duration_np, device=device)
                rank = torch.as_tensor(rank_np, device=device)
                flat_scores: dict[str, torch.Tensor] = {}
                for proxy, rm in reward_models.items():
                    flat_scores[proxy] = rm(history_flat, actions_flat, duration, rank).reshape(-1, args.n_actions)
                item_scores = torch.as_tensor(item_means, device=device)[actions]
                duration_bin = np.clip(np.digitize(duration_np, duration_edges[1:-1]), 0, len(duration_means) - 1)
                duration_scores = torch.as_tensor(duration_means[duration_bin], device=device).reshape(-1, args.n_actions)
                flat_scores["train_item_mean"] = item_scores
                flat_scores["train_duration_mean"] = duration_scores
                top1 = torch.argmax(logits, dim=-1)
                if args.save_action_ids:
                    action_classes, action_item_ids, top1_classes, top1_item_ids = export_action_ids(
                        actions, class_to_raw_item_id, top1)
                    all_results.setdefault(f"{name}_sampled_action_classes", []).append(action_classes)
                    all_results.setdefault(f"{name}_sampled_item_ids", []).append(action_item_ids)
                    all_results.setdefault(f"{name}_top1_class", []).append(top1_classes)
                    all_results.setdefault(f"{name}_top1_item_id", []).append(top1_item_ids)
                top1_duration = torch.as_tensor(item_durations, device=device)[top1]
                top1_rank = torch.as_tensor(arrays["candidate_popularity_rank"][top1.cpu().numpy()].astype(np.int64), device=device)
                top1_scores = {
                    proxy: rm(b["rm_history"], top1, top1_duration, top1_rank)
                    for proxy, rm in reward_models.items()
                }
                top1_scores["train_item_mean"] = torch.as_tensor(item_means, device=device)[top1]
                top1_bin = np.clip(np.digitize(top1_duration.cpu().numpy(), duration_edges[1:-1]), 0, len(duration_means)-1)
                top1_scores["train_duration_mean"] = torch.as_tensor(duration_means[top1_bin], device=device)
                for proxy, values in flat_scores.items():
                    values_np = values.cpu().numpy().astype(np.float32)
                    policy_columns[f"{name}_{proxy}_sample_mean"].append(values_np.mean(axis=1))
                    policy_columns[f"{name}_{proxy}_sample_sd"].append(values_np.std(axis=1))
                    policy_columns[f"{name}_{proxy}_top1"].append(top1_scores[proxy].cpu().numpy().astype(np.float32))
                history_action_ranges.append((flat_scores["history"].max(dim=1).values - flat_scores["history"].min(dim=1).values).max().item())
                if name == args.policy[0].split("=", 1)[0]:
                    if "uid" not in all_results:
                        all_results["uid"] = []
                    all_results["uid"].append(b["uid"].cpu().numpy().astype(np.int64))
                    all_results["event_index"].append(np.asarray(batch_indices, dtype=np.uint64))
                    all_results["organic"].append(b["organic"].cpu().numpy().astype(bool))
                    all_results["target_supported"].append(b["supported"].cpu().numpy().astype(bool))
                    all_results["logged_label"].append(b["reward"].cpu().numpy().astype(np.float32))
                    all_results["logged_target_class"].append(b["target"].cpu().numpy().astype(np.int64))
        model_hash = sha256(path)
        for key, values in policy_columns.items():
            all_results[key] = [np.concatenate(values)]
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
        # Keep per-policy run provenance without exposing action-level logits or samples.
        all_results[f"meta_{name}"] = [np.frombuffer(json.dumps({
            "policy_type": "fixed 50/50 GPT and recent-history-frequency mixture" if is_history_mixture else "checkpoint policy",
            "checkpoint": str(path), "sha256": model_hash,
            "history_mixture_base_sha256": model_hash if is_history_mixture else None,
            "history_proxy_max_action_range": max(history_action_ranges, default=0.0)}).encode(), dtype=np.uint8)]

    result = {key: np.concatenate(value) if key not in {k for k in all_results if k.startswith("meta_")} else value[0]
              for key, value in all_results.items()}
    group_masks = {
        "all": np.ones(len(result["uid"]), dtype=bool),
        "organic": result["organic"],
        "recommended": ~result["organic"],
    }
    summary = {}
    for group, mask in group_masks.items():
        summary[group] = {}
        for policy_name in policy_names:
            summary[group][policy_name] = {}
            for proxy in names:
                summary[group][policy_name][proxy] = {
                    "queries": int(mask.sum()),
                    "mc16_mean": float(result[f"{policy_name}_{proxy}_sample_mean"][mask].mean()),
                    "mc16_mean_within_query_sd": float(result[f"{policy_name}_{proxy}_sample_sd"][mask].mean()),
                    "top1_mean": float(result[f"{policy_name}_{proxy}_top1"][mask].mean()),
                }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    reward_model_paths = {"history": args.history_rm, "item": args.item_rm, "combined": args.combined_rm}
    if args.audit_combined_rm is not None:
        reward_model_paths["combined_audit29"] = args.audit_combined_rm
    meta = {
        "data_source_manifest_sha256": sha256(args.data / "manifest.json"),
        "validation_selection_seed": args.selection_seed,
        "validation_event_count_selected": int(len(indices)),
        "validation_event_indices_sha256": hashlib.sha256(np.asarray(indices, dtype=np.uint32).tobytes()).hexdigest(),
        "action_uniform_seed": args.action_seed,
        "actions_per_query": int(args.n_actions),
        "query_and_action_sampling": "same fixed validation index order and inverse-CDF uniform matrix for every policy; policy-specific logits only determine the CDF",
        "proxy_scope": "learned conditional completion on observed listens; not OPE or causal utility",
        "train_controls": "item means and duration bins computed from supported train_indices only",
        "reward_models": {name: {"path": str(path), "sha256": sha256(path)}
                          for name, path in reward_model_paths.items()},
        "policies": [{"name": name, "type": "fixed 50/50 GPT and recent-history-frequency mixture"
                      if path_text == "fixed-history-mixture" else "checkpoint policy"}
                     for name, path_text in policy_specs],
        "history_mixture_base": ({"path": str(args.history_mixture_base),
                                  "sha256": sha256(args.history_mixture_base)} if has_mixture else None),
        "proxy_summary_by_policy_and_target_group": summary,
        "output_arrays": ("per-query proxy sample means, within-query sample SDs, top-1 proxy scores and group labels" +
                          ("; optional sampled zero-based catalog class IDs, mapped item IDs and top-1 IDs are included"
                           if args.save_action_ids else "; sampled action/item IDs and logits are not saved")),
        "action_id_export": ({"enabled": True, "sampled_class_array": "<policy>_sampled_action_classes",
                              "sampled_item_array": "<policy>_sampled_item_ids",
                              "top1_class_array": "<policy>_top1_class",
                              "top1_item_array": "<policy>_top1_item_id",
                              "class_ids_are_zero_based": True,
                              "item_ids_from": "production build_candidate_encoding(candidate_item_ids)[2] class-to-raw map"}
                             if args.save_action_ids else None),
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
