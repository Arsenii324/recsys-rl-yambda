#!/usr/bin/env python3
"""Explicitly exploratory scoring of saved history gates on the already-viewed R9 order.

This is separate from the frozen-confirmatory exporter. It performs no selection or
training and labels the output as reused-test descriptive evidence.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "infra/attempts/history-gate-on-ce-20261025/stage"
sys.path[:0] = [str(STAGE / "src"), str(STAGE / "scripts")]
from run_history_gate import HistoryGate, evaluate, load_base, sha256  # noqa: E402
from recsys_tt2.batching import open_event_arrays  # noqa: E402

DATA = ROOT / "data/processed/yambda100k-eventuniform"
R9 = ROOT / "artifacts/core-controls-cpu-fallback-20261020-r9-20261020T105818Z/frozen-test-rankings.npz"
CURRENT = ROOT / "artifacts/rl-full43-final-actors-test-rank-export-20261025/per-query-test-ranks.npz"
GATE_ROOT = ROOT / "infra/attempts/history-gate-on-ce-20261025/results"
OUT = ROOT / "artifacts/history-gate-ce-r9-reused-exploratory-20261025"
DATA_SHA = "ecc0aa56ac64c3d2a0351f4e02c60977035855a41b83ed8d88ad947c00d949b6"
TEST_SHA = "40dec33232ce369bb3e0b9f67dcafc1afd298899543de065b216501944297083"
CE_SHA = {
    17: "352a67f210af1c8b260e3ceeb491c63879d86f5e37672fd375956ac62809a311",
    29: "f1c0292f66d2a9c766ac4adda1f558a129672813f886e49de7a5229d53de2f57",
}


def boot(uid: np.ndarray, delta: np.ndarray, *, seed: int) -> dict:
    users, inv = np.unique(uid, return_inverse=True)
    sums = np.bincount(inv, weights=delta, minlength=len(users))
    counts = np.bincount(inv, minlength=len(users)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(users), size=(1000, len(users)))
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {"point": float(delta.mean()),
            "ci95_uid_cluster_percentile": [float(x) for x in np.quantile(estimates, [.025, .975])],
            "users": int(len(users)), "events": int(len(uid)), "bootstrap_replicates": 1000,
            "estimand": "event-weighted paired contrast with UID-cluster resampling"}


def ndcg(rank: np.ndarray, supported: np.ndarray) -> np.ndarray:
    hit = supported & (rank <= 100)
    out = np.zeros(len(rank), dtype=np.float64)
    out[hit] = 1.0 / np.log2(rank[hit].astype(np.float64) + 1.0)
    return out


def log_event(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if OUT.exists():
        raise FileExistsError(OUT)
    if sha256(DATA / "manifest.json") != DATA_SHA or sha256(DATA / "test_indices.npy") != TEST_SHA:
        raise RuntimeError("pinned eventuniform/test-index identity mismatch")
    if sha256(STAGE / "scripts/run_history_gate.py") != "9e1768f4e55d3beefa8e1ebe1cbe0c007a775f95d088224105b6c9482db68de6":
        raise RuntimeError("staged gate evaluator source changed")
    r9, current = np.load(R9), np.load(CURRENT)
    for key in ("event_index", "uid", "target_class", "supported", "organic"):
        if not np.array_equal(r9[key], current[key]):
            raise RuntimeError(f"canonical R9/current export identity mismatch: {key}")
    arrays = open_event_arrays(DATA)
    test_idx = np.load(DATA / "test_indices.npy", mmap_mode="r")
    if len(test_idx) != 232049 or not np.all(arrays["split"][test_idx] == 4):
        raise RuntimeError("test cohort size/split mismatch")
    if not np.array_equal(np.asarray(test_idx), current["event_index"]):
        raise RuntimeError("test index order differs from frozen R9 query order")

    device = torch.device(args.device)
    torch.set_num_threads(2)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(0.05, device=0)
    ranks: dict[str, np.ndarray] = {}
    sources = {}
    t0 = time.monotonic()
    OUT.mkdir(parents=True)
    for seed in (17, 29):
        side = GATE_ROOT / f"seed-{seed}"
        base_path, gate_path = (ROOT / "infra/attempts/rl-full43-ce-rloo8-20261025/gpu/output/"
                                / "rl-full43-ce-rloo8-t4-20261025" / f"seed-{seed}" / "ce" / "step-32000.pt"), side / "gate-final.pt"
        if sha256(base_path) != CE_SHA[seed]:
            raise RuntimeError(f"CE final checkpoint changed for seed {seed}")
        gate_payload = torch.load(gate_path, map_location=device, weights_only=False)
        if gate_payload.get("base_checkpoint_sha256") != CE_SHA[seed] or int(gate_payload.get("step", -1)) != 8029:
            raise RuntimeError(f"gate final checkpoint identity mismatch for seed {seed}")
        base, _, cfg = load_base(base_path, device)
        if (base.n_items != 100000 or base.width != 128 or len(base.blocks) != 2
                or base.blocks[0].attn.heads != 4 or base.max_context != 64):
            raise RuntimeError("unexpected CE base architecture")
        gate = HistoryGate(base.width, hidden=int(gate_payload["config"]["hidden_width"])).to(device)
        gate.load_state_dict(gate_payload["state_dict"], strict=True)
        if seed == 17 and device.type == "cuda":
            parity_indices = np.asarray(test_idx[:512], dtype=np.int64)
            base.cpu()
            gate.cpu()
            _, cpu_columns = evaluate(base, gate, parity_indices, arrays, n_items=base.n_items,
                                      batch_size=128, device=torch.device("cpu"), collect=True)
            base.to(device)
            gate.to(device)
            _, gpu_columns = evaluate(base, gate, parity_indices, arrays, n_items=base.n_items,
                                      batch_size=128, device=device, collect=True)
            if not np.array_equal(cpu_columns["rank_history_gate"], gpu_columns["rank_history_gate"]):
                raise RuntimeError("512-query CPU/CUDA history-gate rank parity failed")
            print(json.dumps({"event": "cpu_cuda_rank_parity", "queries": 512,
                              "seed": seed, "matched": True}), flush=True)
            del cpu_columns, gpu_columns
        report, columns = evaluate(base, gate, test_idx, arrays, n_items=base.n_items,
                                   batch_size=128, device=device, collect=True)
        key = f"ce_seed{seed}_plus_gate"
        ranks[key] = columns["rank_history_gate"].astype(np.uint16)
        sources[key] = {"base_checkpoint": str(base_path), "base_sha256": sha256(base_path),
                        "gate_checkpoint": str(gate_path), "gate_sha256": sha256(gate_path),
                        "gate_training_step": int(gate_payload["step"]),
                        "base_model_config": cfg, "source_metrics": report["all"]["history_gate"]}
        partial_path = OUT / f"seed-{seed}-partial-ranks.npz"
        partial_tmp = OUT / f"seed-{seed}-partial-ranks.tmp.npz"
        with partial_tmp.open("wb") as f:
            np.savez_compressed(f, event_index=test_idx, uid=current["uid"],
                                supported=current["supported"], organic=current["organic"],
                                rank_history_gate=ranks[key])
            f.flush()
            os.fsync(f.fileno())
        os.replace(partial_tmp, partial_path)
        log_event({"event": "scored_gate", "seed": seed,
                   "elapsed_seconds": round(time.monotonic() - t0, 1),
                   "partial_ranks": str(partial_path)})
        del base, gate, columns

    uid = current["uid"]
    supported = current["supported"].astype(bool)
    organic = current["organic"].astype(bool)
    metrics = {}
    for name, r in ranks.items():
        metrics[name] = {}
        for group, mask in {"all": np.ones(len(uid), bool), "supported": supported,
                            "organic": organic, "recommended": ~organic}.items():
            z = r[mask]
            metrics[name][group] = {"n": int(mask.sum()),
                                    "ndcg@100": float(ndcg(z, supported[mask]).mean()),
                                    "recall@100": float((supported[mask] & (z <= 100)).mean())}
    comparisons = {}
    for seed in (17, 29):
        name = f"ce_seed{seed}_plus_gate"
        group_mask = np.ones(len(uid), bool)
        actor = current[f"rank_ce_seed{seed}"]
        delta = ndcg(ranks[name], supported) - ndcg(actor, supported)
        comparisons[f"gate_seed{seed}_minus_matched_CE_seed{seed}"] = boot(uid[group_mask], delta[group_mask], seed=20261025 + seed)
        old = r9["rank_learned_history_gate_seed43_fullsupported"]
        old_delta = ndcg(ranks[name], supported) - ndcg(old, supported)
        comparisons[f"gate_seed{seed}_minus_archived_full43_gate"] = boot(uid[group_mask], old_delta[group_mask], seed=20261050 + seed)
        for label, ref_key in (("recent_frequency", "rank_recent_frequency"),
                               ("global_popularity", "rank_global_popularity")):
            ref_delta = ndcg(ranks[name], supported) - ndcg(r9[ref_key], supported)
            comparisons[f"gate_seed{seed}_minus_{label}"] = boot(uid, ref_delta, seed=20261060 + seed)
    new_mean = np.mean([ndcg(ranks[f"ce_seed{s}_plus_gate"], supported) for s in (17, 29)], axis=0)
    ce_mean = np.mean([ndcg(current[f"rank_ce_seed{s}"], supported) for s in (17, 29)], axis=0)
    old_full = ndcg(r9["rank_learned_history_gate_seed43_fullsupported"], supported)
    comparisons["two_seed_gate_mean_minus_two_seed_CE_mean"] = boot(uid, new_mean - ce_mean, seed=20261077)
    comparisons["two_seed_gate_mean_minus_archived_full43_gate"] = boot(uid, new_mean - old_full, seed=20261078)
    for label, ref_key in (("recent_frequency", "rank_recent_frequency"),
                           ("global_popularity", "rank_global_popularity")):
        comparisons[f"two_seed_gate_mean_minus_{label}"] = boot(
            uid, new_mean - ndcg(r9[ref_key], supported), seed=20261080 + len(label))

    npz = OUT / "per-query-test-ranks.npz"
    payload = {key: current[key] for key in ("event_index", "uid", "target_class", "supported", "organic")}
    payload.update({f"rank_{k}": v for k, v in ranks.items()})
    np.savez_compressed(npz, **payload)
    result = {"schema": "history-gate-r9-reused-exploratory-v1", "status": "complete",
              "interpretation": "Exploratory descriptive scoring on the previously viewed canonical R9 test cohort; no selection or fresh confirmation.",
              "events": len(uid), "users": int(np.unique(uid).size), "supported_events": int(supported.sum()),
              "oov_events": int((~supported).sum()), "feature_manifest_sha256": DATA_SHA,
              "test_indices_sha256": TEST_SHA, "r9_identity_npz_sha256": sha256(R9),
              "current_parent_actor_export_sha256": sha256(CURRENT), "scoring_seconds": round(time.monotonic() - t0, 2),
              "models": sources, "metrics": metrics, "paired_uid_cluster_bootstrap": comparisons,
              "cached_comparison_metrics": {
                  "parent_full43": {"all_ndcg@100": float(ndcg(current["rank_parent_full43"], supported).mean())},
                  "ce_seed17": {"all_ndcg@100": float(ndcg(current["rank_ce_seed17"], supported).mean())},
                  "ce_seed29": {"all_ndcg@100": float(ndcg(current["rank_ce_seed29"], supported).mean())},
                  "archived_full43_gate": {"all_ndcg@100": float(ndcg(r9["rank_learned_history_gate_seed43_fullsupported"], supported).mean())},
                  "recent_frequency": {"all_ndcg@100": float(ndcg(r9["rank_recent_frequency"], supported).mean())},
                  "global_popularity": {"all_ndcg@100": float(ndcg(r9["rank_global_popularity"], supported).mean())}},
              "baseline_rank_sources": {"matched_CE": sha256(CURRENT),
                                        "global_popularity_and_archived_full43_gate": sha256(R9)},
              "per_query_npz_sha256": sha256(npz)}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    log_event({"event": "complete", "summary": str(OUT / "summary.json"),
               "elapsed_seconds": round(time.monotonic() - t0, 1)})


if __name__ == "__main__":
    main()
