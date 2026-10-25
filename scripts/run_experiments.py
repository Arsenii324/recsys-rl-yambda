#!/usr/bin/env python3
"""Train/evaluate the supervised anchor and conditional listened-completion models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from recsys_tt2.batching import batches, make_batch, open_event_arrays
from recsys_tt2.model import ConditionalCompletionRM, ItemGPT


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_provenance(data_root: Path, train_indices: np.ndarray, valid_indices: np.ndarray) -> dict:
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return {
        "data_runtime_path": str(data_root.resolve()),
        "data_source": manifest.get("source"),
        "data_revision": manifest.get("revision"),
        "data_source_sha256": manifest.get("source_sha256"),
        "data_manifest_sha256": sha256(manifest_path),
        "train_indices_file_sha256": sha256(data_root / "train_indices.npy"),
        "train_anchor_count_used": int(len(train_indices)),
        "train_anchor_indices_used_sha256": hashlib.sha256(np.asarray(train_indices, dtype=np.uint32).tobytes()).hexdigest(),
        "validation_event_count_used": int(len(valid_indices)),
        "validation_event_indices_used_sha256": hashlib.sha256(np.asarray(valid_indices, dtype=np.uint32).tobytes()).hexdigest(),
    }


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def device_from(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def eval_policy(model: ItemGPT, indices: np.ndarray, arrays: dict, *, batch_size: int, device: torch.device):
    model.eval()
    total = {"all": {"n": 0, "supported": 0}, "organic": {"n": 0, "supported": 0}, "recommended": {"n": 0, "supported": 0}}
    for group in total.values():
        group["repeat_in_history"] = 0
        group["repeat_supported"] = 0
        for k in (10, 50, 100):
            group[f"recall@{k}"] = 0.0
            group[f"ndcg@{k}"] = 0.0
            for baseline in ("popularity", "last_item"):
                group[f"{baseline}_recall@{k}"] = 0.0
                group[f"{baseline}_ndcg@{k}"] = 0.0
    catalog_counts = np.zeros(model.n_items, dtype=np.int64)
    with torch.inference_mode():
        for id_batch in batches(indices, batch_size):
            b = make_batch(id_batch, arrays, n_items=model.n_items, max_context=model.max_context, device=device)
            logits = model(b["tokens"], b["padding_mask"])
            top = torch.topk(logits, k=100, dim=1, sorted=True).indices
            supported = b["supported"]
            target = b["target"]
            organic = b["organic"]
            for name, mask in (("all", torch.ones_like(supported)), ("organic", organic), ("recommended", ~organic)):
                n = int(mask.sum())
                if not n:
                    continue
                g = total[name]
                g["n"] += n
                g["supported"] += int((supported & mask).sum())
                match = top.eq(target[:, None]) & supported[:, None] & mask[:, None]
                found = match.any(dim=1)
                ranks = match.float().argmax(dim=1) + 1
                # Candidate classes are assigned in descending train frequency, so their
                # class rank is the train-only popularity baseline. Last-item predicts the
                # most recent observed in-catalog item at rank one.
                pop_rank = b["target_popularity_rank"] + 1
                history = b["tokens"]
                in_catalog_history = history.ge(1) & history.le(model.n_items)
                positions = torch.arange(history.shape[1], device=history.device).expand_as(history)
                last_position = torch.where(in_catalog_history, positions, -1).max(dim=1).values
                last_token = history.gather(1, last_position.clamp(min=0)[:, None]).squeeze(1)
                last_token = torch.where(last_position >= 0, last_token, 0)
                last_hit = supported & (last_token > 0) & (last_token - 1 == target)
                repeated = supported & history.eq(target[:, None] + 1).any(dim=1)
                g["repeat_in_history"] += int((repeated & mask).sum())
                g["repeat_supported"] += int((supported & mask).sum())
                for k in (10, 50, 100):
                    hit = found & (ranks <= k)
                    g[f"recall@{k}"] += float(hit.sum())
                    gain = torch.where(hit, 1.0 / torch.log2(ranks.float() + 1.0), 0.0)
                    g[f"ndcg@{k}"] += float(gain.sum())
                    pop_hit = supported & (pop_rank <= k) & mask
                    g[f"popularity_recall@{k}"] += float(pop_hit.sum())
                    g[f"popularity_ndcg@{k}"] += float(torch.where(pop_hit, 1.0 / torch.log2(pop_rank.float() + 1.0), 0.0).sum())
                    g[f"last_item_recall@{k}"] += float((last_hit & mask).sum())
                    g[f"last_item_ndcg@{k}"] += float((last_hit & mask).sum())
            catalog_counts += np.bincount(top[:, :100].cpu().numpy().reshape(-1), minlength=model.n_items)
    for g in total.values():
        n = max(1, g["n"])
        for metric in ("recall@10", "recall@50", "recall@100", "ndcg@10", "ndcg@50", "ndcg@100",
                       "popularity_recall@10", "popularity_recall@50", "popularity_recall@100",
                       "popularity_ndcg@10", "popularity_ndcg@50", "popularity_ndcg@100",
                       "last_item_recall@10", "last_item_recall@50", "last_item_recall@100",
                       "last_item_ndcg@10", "last_item_ndcg@50", "last_item_ndcg@100"):
            g[metric] /= n
        g["support_fraction"] = g["supported"] / n
        g["repeat_in_history_fraction"] = g.pop("repeat_in_history") / n
        g["repeat_in_history_supported_fraction"] = g.pop("repeat_supported") / max(1, g["supported"])
    total["catalog"] = {
        "top100_coverage_fraction": float(np.count_nonzero(catalog_counts) / model.n_items),
        "top1_item_share": float(catalog_counts.max() / max(1, catalog_counts.sum())),
        "recommended_items": int(np.count_nonzero(catalog_counts)),
    }
    return total


def run_base(args) -> None:
    config = load_config(args.config)
    seed_all(int(config["seed"]))
    device = device_from(args.device)
    arrays = open_event_arrays(args.data)
    train_indices = np.load(Path(args.data) / "train_indices.npy", mmap_mode="r")
    valid_indices = np.load(Path(args.data) / "valid_indices.npy", mmap_mode="r")
    if args.max_train_events:
        train_indices = train_indices[: args.max_train_events]
    if args.max_eval_events:
        valid_indices = valid_indices[: args.max_eval_events]
    elif int(config.get("eval_event_count", 0)) and len(valid_indices) > int(config["eval_event_count"]):
        # Shared, deterministic event-uniform validation subset across seeds/models.
        valid_indices = np.random.default_rng(991).choice(valid_indices, size=int(config["eval_event_count"]), replace=False)
    model = ItemGPT(
        n_items=int(config["n_items"]), width=int(config["width"]), layers=int(config["layers"]),
        heads=int(config["heads"]), max_context=int(config["max_context"]), dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    if config.get("single_pass"):
        expected_steps = math.ceil(len(train_indices) / int(config["batch_size"]))
        if int(config["steps"]) != expected_steps or not config.get("partial_last_batch", False):
            raise ValueError(f"single-pass config must use {expected_steps} steps with partial_last_batch=true")
    run_dir = Path(args.output)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_validation = -math.inf
    best_step = None
    with (run_dir / "metrics.jsonl").open("a") as log:
        model.train()
        permutation = np.random.default_rng(int(config["seed"])).permutation(train_indices)
        cursor = 0
        train_examples_seen = 0
        start = time.time()
        for step in range(1, int(config["steps"]) + 1):
            bs = int(config["batch_size"])
            if cursor + bs > len(permutation):
                if config.get("partial_last_batch", False) and cursor < len(permutation):
                    ids = permutation[cursor:]
                    cursor = len(permutation)
                else:
                    if config.get("single_pass"):
                        raise RuntimeError("single-pass training attempted to wrap the permutation")
                    permutation = np.random.default_rng(int(config["seed"]) + step).permutation(train_indices)
                    cursor = 0
                    ids = permutation[cursor : cursor + bs]
                    cursor += len(ids)
            else:
                ids = permutation[cursor : cursor + bs]
                cursor += len(ids)
            train_examples_seen += len(ids)
            b = make_batch(ids, arrays, n_items=model.n_items, max_context=model.max_context, device=device)
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(b["tokens"], b["padding_mask"])
                    loss = F.cross_entropy(logits[b["supported"]], b["target"][b["supported"]])
            else:
                logits = model(b["tokens"], b["padding_mask"])
                loss = F.cross_entropy(logits[b["supported"]], b["target"][b["supported"]])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite CE at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if step % 100 == 0 or step == 1:
                record = {"step": step, "train_ce": float(loss.detach()), "elapsed_seconds": time.time() - start}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
            if step % int(config["eval_every"]) == 0 or step == int(config["steps"]):
                validation = eval_policy(model, valid_indices, arrays, batch_size=bs, device=device)
                record = {"step": step, "validation": validation, "elapsed_seconds": time.time() - start}
                log.write(json.dumps(record) + "\n")
                log.flush()
                score = validation["all"]["ndcg@100"]
                if score > best_validation:
                    best_validation, best_step = score, step
                    torch.save({"state_dict": model.state_dict(), "config": config, "step": step,
                                "selection_metric": "validation all-event NDCG@100", "selection_score": score}, run_dir / "base-best.pt")
                if step == int(config["steps"]):
                    torch.save({"state_dict": model.state_dict(), "config": config, "step": step}, run_dir / "base-final.pt")
                print(json.dumps(record), flush=True)
                model.train()
    provenance = data_provenance(Path(args.data), train_indices, valid_indices)
    (run_dir / "run.json").write_text(json.dumps({
        "mode": "base", "config": config, **provenance,
        "device": str(device), "torch": torch.__version__, "code_sha256": sha256(Path(__file__)),
        "candidate_mapping_version": "legacy-encode-v1-derived-class-to-raw-and-frequency-rank",
        "training_examples_seen": int(train_examples_seen),
        "train_anchor_fraction_seen": min(1.0, train_examples_seen / max(1, len(train_indices))),
        "selection_metric": "validation all-event NDCG@100", "best_step": best_step, "best_score": best_validation,
    }, indent=2) + "\n")


def _pearson(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 2 or np.std(pred) == 0 or np.std(target) == 0:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


def eval_reward(model: ConditionalCompletionRM, indices: np.ndarray, arrays: dict, *, batch_size: int, device: torch.device,
                global_mean: float, item_means: np.ndarray):
    model.eval()
    predictions, labels, orgs, durations, classes, popularity_ranks = [], [], [], [], [], []
    with torch.inference_mode():
        for id_batch in batches(indices, batch_size):
            b = make_batch(id_batch, arrays, n_items=model.n_items, max_context=64, device=device)
            keep = b["supported"]
            if not bool(keep.any()):
                continue
            pred = model(b["rm_history"][keep], b["target"][keep], b["duration"][keep], b["target_popularity_rank"][keep])
            predictions.extend(pred.cpu().numpy().tolist())
            labels.extend(b["reward"][keep].cpu().numpy().tolist())
            orgs.extend(b["organic"][keep].cpu().numpy().tolist())
            durations.extend(b["duration"][keep].cpu().numpy().tolist())
            classes.extend(b["target"][keep].cpu().numpy().tolist())
            popularity_ranks.extend(b["target_popularity_rank"][keep].cpu().numpy().tolist())
    pred = np.asarray(predictions, dtype=np.float32)
    target = np.asarray(labels, dtype=np.float32)
    organic = np.asarray(orgs, dtype=bool)
    duration = np.asarray(durations, dtype=np.float32)
    target_class = np.asarray(classes, dtype=np.int64)
    target_rank = np.asarray(popularity_ranks, dtype=np.int64)
    output = {}
    for name, mask in (("all", np.ones(len(target), dtype=bool)), ("organic", organic), ("recommended", ~organic)):
        if not mask.any():
            continue
        error = pred[mask] - target[mask]
        output[name] = {"n": int(mask.sum()), "mse": float(np.mean(error ** 2)),
                        "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
                        "pearson": _pearson(pred[mask], target[mask]), "mean_label": float(target[mask].mean()),
                        "mean_prediction": float(pred[mask].mean())}
        global_error = global_mean - target[mask]
        item_error = item_means[target_class[mask]] - target[mask]
        output[name]["train_mean_control"] = {"mse": float(np.mean(global_error ** 2)),
                                                "mae": float(np.abs(global_error).mean()),
                                                "rmse": float(np.sqrt(np.mean(global_error ** 2)))}
        output[name]["train_item_mean_control"] = {"mse": float(np.mean(item_error ** 2)),
                                                    "mae": float(np.abs(item_error).mean()),
                                                    "rmse": float(np.sqrt(np.mean(item_error ** 2)))}
        edges = [0, 30, 60, 120, 240, 600, 2_500]
        duration_bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            bmask = mask & (duration >= lo) & (duration < hi)
            if bmask.any():
                duration_bins.append({"duration_range": [lo, hi], "n": int(bmask.sum()),
                                      "mean_label": float(target[bmask].mean()), "mean_prediction": float(pred[bmask].mean())})
        output[name]["duration_calibration"] = duration_bins
        rank_edges = [0, 1_000, 10_000, 50_000, 100_000]
        rank_bins = []
        for lo, hi in zip(rank_edges[:-1], rank_edges[1:]):
            rmask = mask & (target_rank >= lo) & (target_rank < hi)
            if rmask.any():
                rank_bins.append({"train_popularity_rank_range": [lo + 1, hi], "n": int(rmask.sum()),
                                  "mean_label": float(target[rmask].mean()), "mean_prediction": float(pred[rmask].mean())})
        output[name]["train_popularity_rank_calibration"] = rank_bins
    output["support_fraction"] = len(target) / max(1, len(indices))
    return output


def run_reward(args) -> None:
    config = load_config(args.config)
    seed_all(int(config["seed"]))
    device = device_from(args.device)
    arrays = open_event_arrays(args.data)
    train_indices = np.load(Path(args.data) / "train_indices.npy", mmap_mode="r")
    valid_indices = np.load(Path(args.data) / "valid_indices.npy", mmap_mode="r")
    if args.max_train_events:
        train_indices = train_indices[: args.max_train_events]
    if args.max_eval_events:
        valid_indices = valid_indices[: args.max_eval_events]
    elif int(config.get("eval_event_count", 0)) and len(valid_indices) > int(config["eval_event_count"]):
        valid_indices = np.random.default_rng(991).choice(valid_indices, size=int(config["eval_event_count"]), replace=False)
    # Simple training-only controls on the same supported event sample used by the RM.
    item_sum = np.zeros(int(config["n_items"]), dtype=np.float64)
    item_count = np.zeros(int(config["n_items"]), dtype=np.int64)
    target_tokens = arrays["item_token"][train_indices].astype(np.int64)
    train_labels = np.minimum(arrays["played_ratio_pct"][train_indices].astype(np.float64) / 100.0, 1.0)
    target_classes = target_tokens - 1
    np.add.at(item_sum, target_classes, train_labels)
    np.add.at(item_count, target_classes, 1)
    global_mean = float(train_labels.mean()) if len(train_labels) else 0.0
    item_means = np.divide(item_sum, item_count, out=np.full_like(item_sum, global_mean), where=item_count > 0)
    run_dir = Path(args.output)
    run_dir.mkdir(parents=True, exist_ok=True)
    variant_exposure = {}
    for variant in config["variants"]:
        model = ConditionalCompletionRM(int(config["n_items"]), width=int(config["width"]), variant=variant).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
        permutation = np.random.default_rng(int(config["seed"])).permutation(train_indices)
        cursor = 0
        started = time.time()
        model.train()
        metrics_path = run_dir / f"rm-{variant}-metrics.jsonl"
        metrics_path.write_text("")
        best_validation_mse = math.inf
        best_step = None
        train_examples_seen = 0
        for step in range(1, int(config["steps"]) + 1):
            bs = int(config["batch_size"])
            if cursor + bs > len(permutation):
                if config.get("partial_last_batch", False) and cursor < len(permutation):
                    batch_indices = permutation[cursor:]
                    cursor = len(permutation)
                else:
                    permutation = np.random.default_rng(int(config["seed"]) + step).permutation(train_indices)
                    cursor = 0
                    batch_indices = permutation[cursor : cursor + bs]
                    cursor += len(batch_indices)
            else:
                batch_indices = permutation[cursor : cursor + bs]
                cursor += len(batch_indices)
            train_examples_seen += len(batch_indices)
            b = make_batch(batch_indices, arrays, n_items=model.n_items, max_context=64, device=device)
            keep = b["supported"]
            pred = model(b["rm_history"][keep], b["target"][keep], b["duration"][keep], b["target_popularity_rank"][keep])
            loss = F.mse_loss(pred, b["reward"][keep])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % int(config.get("log_every", 100)) == 0 or step == int(config["steps"]):
                train_record = {"variant": variant, "step": step, "train_mse": float(loss.detach()),
                                "elapsed_seconds": time.time() - started}
                with metrics_path.open("a") as metrics_log:
                    metrics_log.write(json.dumps(train_record) + "\n")
                print(json.dumps(train_record), flush=True)
            if step % int(config["eval_every"]) == 0 or step == int(config["steps"]):
                validation = eval_reward(model, valid_indices, arrays, batch_size=bs, device=device,
                                         global_mean=global_mean, item_means=item_means)
                score = float(validation["all"]["mse"])
                if score < best_validation_mse:
                    best_validation_mse = score
                    best_step = step
                    torch.save({"state_dict": model.state_dict(), "config": config, "variant": variant,
                                "step": step, "selection_metric": "validation supported all-event MSE",
                                "selection_score": score}, run_dir / f"rm-{variant}-best.pt")
                if step == int(config["steps"]):
                    torch.save({"state_dict": model.state_dict(), "config": config, "variant": variant,
                                "step": step}, run_dir / f"rm-{variant}-final.pt")
                (run_dir / f"rm-{variant}-validation.json").write_text(json.dumps(validation, indent=2) + "\n")
                validation_record = {"variant": variant, "step": step, "validation": validation,
                                     "elapsed_seconds": time.time() - started}
                with metrics_path.open("a") as metrics_log:
                    metrics_log.write(json.dumps(validation_record) + "\n")
                print(json.dumps(validation_record), flush=True)
                model.train()
        variant_exposure[variant] = {
            "steps": int(config["steps"]),
            "examples_seen": int(train_examples_seen),
            "available_train_anchors": int(len(train_indices)),
            "anchor_fraction_seen": min(1.0, train_examples_seen / max(1, len(train_indices))),
            "best_validation_mse": best_validation_mse,
            "best_step": best_step,
            "final_step": int(config["steps"]),
        }
    validation_supported = int(np.count_nonzero(arrays["item_token"][valid_indices] > 0))
    (run_dir / "run.json").write_text(json.dumps({
        "mode": "conditional_completion_reward_model", "config": config,
        **data_provenance(Path(args.data), train_indices, valid_indices),
        "code_sha256": sha256(Path(__file__)), "device": str(device), "torch": torch.__version__,
        "candidate_mapping_version": "legacy-encode-v1-derived-class-to-raw-and-frequency-rank",
        "train_supported_anchor_count_used": int(len(train_indices)),
        "validation_supported_target_count": validation_supported,
        "validation_support_fraction": validation_supported / max(1, len(valid_indices)),
        "training_exposure_by_variant": variant_exposure,
        "selection_metric": "best checkpoint by supported all-event validation MSE; final checkpoint also retained",
        "fit_scope": "supported training listen events only; target is clipped completion conditional on observed listen"
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["base", "reward"])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-events", type=int, default=0, help="optional deterministic prefix limit for smoke runs")
    parser.add_argument("--max-eval-events", type=int, default=0, help="optional deterministic prefix limit for smoke runs")
    args = parser.parse_args()
    if args.mode == "base":
        run_base(args)
    else:
        run_reward(args)


if __name__ == "__main__":
    main()
