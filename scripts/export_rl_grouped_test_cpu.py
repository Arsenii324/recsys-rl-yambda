#!/usr/bin/env python3
"""Exploratory CPU ranks for frozen grouped-RL endpoints on the reused R9 test order.

This is a separate post-hoc export. It does not read or alter frozen selection
manifests and does not select checkpoints. Test metrics were already viewed in
R9; this is exploratory reuse, not fresh confirmatory evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from recsys_tt2.batching import make_batch, open_event_arrays  # noqa: E402
from recsys_tt2.model import ItemGPT  # noqa: E402

DATA_SHA = "ecc0aa56ac64c3d2a0351f4e02c60977035855a41b83ed8d88ad947c00d949b6"
BASE_SHA = "4076631ea76eca428fb3747f4f112b33633d7c835048cf7e1b2a2f807533674d"
TEST_SHA = "40dec33232ce369bb3e0b9f67dcafc1afd298899543de065b216501944297083"
R9_NPZ = ROOT / "artifacts/core-controls-cpu-fallback-20261020-r9-20261020T105818Z/frozen-test-rankings.npz"
R9_RUN = ROOT / "artifacts/core-controls-cpu-fallback-20261020-r9-20261020T105818Z/frozen-test-rankings.run.json"
ARCH = ("n_items", "width", "layers", "heads", "max_context", "dropout")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def contributions(rank: np.ndarray, supported: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    hit = supported & (rank <= k)
    ndcg = np.zeros(rank.shape, dtype=np.float64)
    ndcg[hit] = 1.0 / np.log2(rank[hit].astype(np.float64) + 1.0)
    return hit.astype(np.float64), ndcg


def aggregate(rank: np.ndarray, supported: np.ndarray) -> dict:
    result = {"n": int(len(rank)), "supported": int(supported.sum())}
    for k in (10, 100):
        hit, ndcg = contributions(rank, supported, k)
        result[f"recall@{k}"] = float(hit.mean())
        result[f"ndcg@{k}"] = float(ndcg.mean())
    return result


def uid_bootstrap(uid: np.ndarray, delta: np.ndarray, reps: int = 1000, seed: int = 20261025) -> dict:
    active, inv = np.unique(uid, return_inverse=True)
    sums = np.bincount(inv, weights=delta, minlength=len(active))
    counts = np.bincount(inv, minlength=len(active)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(active), size=(reps, len(active)))
    est = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    return {"point": float(delta.mean()), "ci95_percentile": [float(np.quantile(est, .025)),
            float(np.quantile(est, .975))], "users": int(len(active)), "events": int(len(delta)),
            "bootstrap_replicates": reps, "estimand": "event-weighted paired contrast; UID-cluster resampling"}


def load_checkpoint(path: Path, arm: str, seed: int, expected_sha: str, base_arch: dict):
    observed_sha = sha256(path)
    if observed_sha != expected_sha:
        raise RuntimeError(f"checkpoint SHA mismatch for {arm}/seed{seed}: {observed_sha}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    identity = {"arm": arm, "seed": seed, "step": 32000, "parent_sha256": BASE_SHA}
    if {k: payload.get(k) for k in identity} != identity or float(payload.get("beta", -1)) != .1:
        raise RuntimeError(f"checkpoint payload identity mismatch: {path}")
    arch = {k: payload["config"][k] for k in ARCH}
    if arch != base_arch or int(arch["n_items"]) != 100000:
        raise RuntimeError(f"checkpoint architecture differs from pinned SFT: {path}")
    model = ItemGPT(**arch)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model, {"path": str(path.resolve()), "sha256": observed_sha,
                   "bytes": path.stat().st_size, "arm": arm, "seed": seed,
                   "step": int(payload["step"]), "beta": float(payload["beta"]),
                   "parent_checkpoint_sha256": payload["parent_sha256"], "architecture": arch}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=ROOT / "data/processed/yambda100k-eventuniform")
    p.add_argument("--results-root", type=Path, default=ROOT / "infra/attempts/reward-frontier-grouped-rl-recovery-v3/gpu-recovery/output/reward-frontier-recovery-v2-20261025")
    p.add_argument("--baseline", type=Path, default=ROOT / "infra/kaggle/core-controls-20261019/aux_upload/checkpoint_gpt_seed43_subset.pt")
    p.add_argument("--r9-npz", type=Path, default=R9_NPZ)
    p.add_argument("--output-npz", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--preflight-only", action="store_true", help="score first 512 SFT rows and compare R9 ranks")
    args = p.parse_args()
    if args.batch_size not in (128, 256):
        raise ValueError("batch-size must be 128 or 256")
    if args.output_npz.exists() or args.output_json.exists():
        raise FileExistsError("refusing to overwrite a prior exploratory export")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    manifest = args.data_root / "manifest.json"
    test_file = args.data_root / "test_indices.npy"
    if sha256(manifest) != DATA_SHA or sha256(test_file) != TEST_SHA:
        raise RuntimeError("feature manifest or canonical test-index SHA mismatch")
    indices = np.load(test_file, mmap_mode="r")
    if len(indices) != 232049:
        raise RuntimeError("canonical test event count mismatch")
    r9run = json.loads(R9_RUN.read_text())
    if (r9run.get("test_event_count") != len(indices)
            or r9run.get("test_indices_sha256") != TEST_SHA
            or r9run.get("data_manifest_sha256") != DATA_SHA
            or r9run.get("checkpoint_sha256", {}).get("gpt_seed43_subset") != BASE_SHA):
        raise RuntimeError("R9 cache is not pinned to this full test order and SFT reference")
    ref = np.load(args.r9_npz)
    if not np.array_equal(ref["event_index"], indices.astype(np.int64)):
        raise RuntimeError("R9 event order differs from canonical test index order")
    arrays = open_event_arrays(args.data_root)
    split = np.asarray(arrays["split"][indices], dtype=np.uint8)
    if not np.all(split == 4):
        raise RuntimeError("test indices contain rows outside the frozen test split")
    base_payload = torch.load(args.baseline, map_location="cpu", weights_only=False)
    if sha256(args.baseline) != BASE_SHA:
        raise RuntimeError("SFT baseline is not the exact pinned 407 checkpoint")
    base_arch = {k: base_payload["config"][k] for k in ARCH}
    del base_payload

    n = 512 if args.preflight_only else len(indices)
    rows = indices[:n]
    base_model = ItemGPT(**base_arch)
    bp = torch.load(args.baseline, map_location="cpu", weights_only=False)
    base_model.load_state_dict(bp["state_dict"], strict=True)
    del bp
    base_model.eval().requires_grad_(False)
    probe_n = min(512, len(indices))
    parity_rank = np.full(probe_n, 101, np.uint16)
    probe_rows = indices[:probe_n]
    probe_uid = np.empty(probe_n, np.int64)
    probe_target = np.empty(probe_n, np.int32)
    probe_supported = np.empty(probe_n, bool)
    probe_organic = np.empty(probe_n, bool)
    progress_every = max(1, 512 // args.batch_size)
    with torch.inference_mode():
        for start in range(0, probe_n, args.batch_size):
            ix = probe_rows[start:start + args.batch_size]
            b = make_batch(ix, arrays, n_items=base_arch["n_items"],
                           max_context=base_arch["max_context"], device=torch.device("cpu"))
            logits = base_model(b["tokens"], b["padding_mask"]).float()
            top = torch.topk(logits, k=100, dim=1, sorted=True).indices.numpy()
            tgt, sup = b["target"].numpy(), b["supported"].numpy()
            rr = np.full(len(ix), 101, np.uint16)
            r, c = np.nonzero((top == tgt[:, None]) & sup[:, None])
            rr[r] = c.astype(np.uint16) + 1
            parity_rank[start:start + len(ix)] = rr
            probe_uid[start:start + len(ix)] = b["uid"].numpy()
            probe_target[start:start + len(ix)] = np.where(sup, tgt, -1)
            probe_supported[start:start + len(ix)] = sup
            probe_organic[start:start + len(ix)] = b["organic"].numpy()
            if (start // args.batch_size) % progress_every == 0:
                print(json.dumps({"phase": "sft-protocol-probe", "done": min(start + len(ix), n),
                                  "total": n, "elapsed": round(time.monotonic() - started, 2)}), flush=True)
    del base_model
    cache_rank = ref["rank_gpt_seed43"][:probe_n].astype(np.uint16)
    mismatches = int(np.count_nonzero(parity_rank != cache_rank))
    if (not np.array_equal(probe_uid, ref["uid"][:probe_n])
            or not np.array_equal(probe_target, ref["target_class"][:probe_n])
            or not np.array_equal(probe_supported, ref["supported"][:probe_n])
            or not np.array_equal(probe_organic, ref["organic"][:probe_n])):
        raise RuntimeError("probe row labels/UID do not join R9 cache on event order")
    if args.preflight_only:
        print(json.dumps({"status": "preflight_complete", "rows": probe_n, "sft_rank_mismatches_vs_r9": mismatches,
                          "exact_rank_match": mismatches == 0,
                          "elapsed_seconds": round(time.monotonic() - started, 3),
                          "baseline_sha256": BASE_SHA, "test_indices_sha256": TEST_SHA}, sort_keys=True), flush=True)
        return
    if mismatches:
        raise RuntimeError(f"SFT top100 protocol differs from R9 rank cache in {mismatches}/{probe_n} probe rows")

    # Join every cached test label back to the event arrays in canonical order.
    n = len(indices)
    rows = indices
    uid = np.asarray(arrays["uid"][indices], dtype=np.int64)
    item_token = np.asarray(arrays["item_token"][indices], dtype=np.int64)
    supported = item_token > 0
    target = np.where(supported, item_token - 1, -1).astype(np.int32)
    organic = np.asarray(arrays["is_organic"][indices], dtype=np.bool_)
    ratio = np.asarray(arrays["played_ratio_pct"][indices], dtype=np.float32)
    duration = np.asarray(arrays["track_length_seconds"][indices], dtype=np.float32)
    if (not np.array_equal(uid, ref["uid"]) or not np.array_equal(target, ref["target_class"])
            or not np.array_equal(supported, ref["supported"]) or not np.array_equal(organic, ref["organic"])):
        raise RuntimeError("full test labels/UID do not join R9 cache on canonical event order")
    cache_rank = ref["rank_gpt_seed43"].astype(np.uint16)

    ranks = {}
    model_info = {}
    for seed in (17, 29):
        run_path = args.results_root / f"seed-{seed}/run-result.json"
        run = json.loads(run_path.read_text())
        cfg = run.get("config", {})
        if (run.get("status") != "complete" or int(run.get("seed", -1)) != seed
                or int(run.get("selected_common_dose", -1)) != 32000
                or int(cfg.get("steps", -1)) != 32000 or int(cfg.get("batch_size", -1)) != 128
                or run.get("base_checkpoint_sha256") != BASE_SHA
                or run.get("feature_manifest_sha256") != DATA_SHA):
            raise RuntimeError(f"provider run result provenance mismatch: {run_path}")
        for row in run.get("results", []):
            arm = row.get("arm")
            if arm not in ("ce", "context_reinforce", "rloo8"):
                continue
            if int(row.get("steps", -1)) != 32000 or int(row.get("training_examples_seen", -1)) != 4096000:
                raise RuntimeError(f"unexpected dose for {arm}/seed{seed}")
            path = args.results_root / f"seed-{seed}/{arm}/step-32000.pt"
            model, info = load_checkpoint(path, arm, seed, row["final_checkpoint_sha256"], base_arch)
            name = f"{arm}_seed{seed}"
            rank = np.full(n, 101, np.uint16)
            with torch.inference_mode():
                for start in range(0, n, args.batch_size):
                    ix = rows[start:start + args.batch_size]
                    b = make_batch(ix, arrays, n_items=base_arch["n_items"],
                                   max_context=base_arch["max_context"], device=torch.device("cpu"))
                    logits = model(b["tokens"], b["padding_mask"]).float()
                    top = torch.topk(logits, k=100, dim=1, sorted=True).indices.numpy()
                    tgt, sup = b["target"].numpy(), b["supported"].numpy()
                    rr = np.full(len(ix), 101, np.uint16)
                    r, c = np.nonzero((top == tgt[:, None]) & sup[:, None])
                    rr[r] = c.astype(np.uint16) + 1
                    rank[start:start + len(ix)] = rr
                    done = start + len(ix)
                    if done == n or done % (args.batch_size * 64) == 0:
                        print(json.dumps({"phase": name, "done": done, "total": n,
                                          "elapsed": round(time.monotonic() - started, 2)}), flush=True)
            del model
            ranks[name] = rank
            model_info[name] = info
    # Cache identity and data-derived labels are retained directly to make all joins auditable.
    ratio_stratum = np.where(ratio == 0, 0, np.where(ratio < 50, 1, 2)).astype(np.uint8)
    metadata = {"event_index": rows.astype(np.uint32), "uid": uid, "target_class": target,
                "supported": supported, "organic": organic, "played_ratio_pct": ratio,
                "duration_seconds": duration, "completion_stratum": ratio_stratum,
                "rank100_sft_reference": cache_rank}
    for name, value in ranks.items():
        metadata[f"rank100_{name}"] = value
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **metadata)
    groups = {"all": np.ones(n, bool), "organic": organic, "recommended": ~organic,
              "zero_completion": ratio == 0, "below_50_completion": (ratio > 0) & (ratio < 50),
              "at_least_50_completion": ratio >= 50}
    names = ["sft_reference"] + sorted(ranks)
    all_ranks = {"sft_reference": cache_rank, **ranks}
    summary = {}
    for name in names:
        summary[name] = {group: aggregate(all_ranks[name][mask], supported[mask])
                         for group, mask in groups.items()}
    contrasts = {}
    for arm in ("ce", "context_reinforce", "rloo8"):
        seed_values = [ranks[f"{arm}_seed{seed}"] for seed in (17, 29)]
        m_ndcg = []
        m_recall = []
        for rr in seed_values:
            hit, ndcg = contributions(rr, supported, 100)
            m_recall.append(hit); m_ndcg.append(ndcg)
        for comparator, comp_rank in (("sft_reference", cache_rank),
                                      ("selected_ce_r9", ref["rank_selected_ce"].astype(np.uint16)),
                                      ("recent_frequency_r9", ref["rank_recent_frequency"].astype(np.uint16))):
            comp_hit, comp_ndcg = contributions(comp_rank, supported, 100)
            contrasts[f"{arm}_two_seed_mean_minus_{comparator}"] = {
                "ndcg@100": uid_bootstrap(uid, np.mean(m_ndcg, axis=0) - comp_ndcg),
                "recall@100": uid_bootstrap(uid, np.mean(m_recall, axis=0) - comp_hit)}
    cached_r9 = {}
    cached_names = ("global_popularity", "recent_frequency", "gpt_seed43_fullsupported",
                    "fixed_history_mixture_0p5", "learned_history_gate", "selected_ce",
                    "rl_beta001_constant_2k", "rl_beta01_constant_2k",
                    "fixed_history_mixture_selected_seed43_subset",
                    "fixed_history_mixture_selected_seed43_fullsupported",
                    "fixed_history_mixture_0p5_seed43_fullsupported",
                    "learned_history_gate_seed43_fullsupported")
    for base_name in cached_names:
        base_rank = ref[f"rank_{base_name}"].astype(np.uint16)
        cached_r9[base_name] = {group: aggregate(base_rank[mask], supported[mask])
                                for group, mask in groups.items()}
    summary_doc = {"schema": "rl-grouped-reused-test-exploratory-v1", "status": "complete",
        "interpretation": "Exploratory rescore on the same 232,049-event R9 test order whose aggregate outcomes were previously viewed; not fresh confirmation and not a checkpoint/seed selection basis.",
        "test_event_count": n, "test_indices_sha256": TEST_SHA, "data_manifest_sha256": DATA_SHA,
        "r9_rank_cache": str(args.r9_npz.resolve()), "r9_rank_cache_sha256": sha256(args.r9_npz),
        "r9_run_metadata_sha256": sha256(R9_RUN), "r9_sft_policy": "gpt_seed43_subset",
        "sft_checkpoint_sha256": BASE_SHA, "first_512_sft_r9_rank_mismatches": mismatches,
        "test_outcomes_opened": True, "rank_cap": 100, "miss_value": 101,
        "optimizer_exposure": {"updates_per_fit": 32000, "examples_per_fit": 4096000,
                                "seeds": [17, 29], "arms": ["ce", "context_reinforce", "rloo8"]},
        "models": model_info, "group_counts": {k: int(v.sum()) for k, v in groups.items()},
        "aggregates": summary, "cached_r9_baselines": cached_r9,
        "cached_baselines_note": "These ranks are exact-order reused R9 outputs and retain their distinct exposure/initialization regimes; they are descriptive comparisons, not matched-dose arms.",
        "paired_uid_bootstrap_contrasts": contrasts,
        "conditioning": "Two-fit actor mean is per-query mean over seeds 17/29; UID-cluster bootstrap CIs are conditional on these trained fits, not seed uncertainty.",
        "target_label_note": "played_ratio_pct strata use the observed next event after the strict-prior context; descriptive association only, not counterfactual response.",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "exporter_sha256": sha256(Path(__file__)), "npz_sha256": sha256(args.output_npz)}
    args.output_json.write_text(json.dumps(summary_doc, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "npz": str(args.output_npz), "summary": str(args.output_json),
                      "elapsed_seconds": summary_doc["elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
