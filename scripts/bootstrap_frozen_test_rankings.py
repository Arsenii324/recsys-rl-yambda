#!/usr/bin/env python3
"""Paired user-cluster intervals for a frozen ranking export.

Contrasts are read from the exact frozen selection manifest recorded by the
exporter. The bootstrap resamples users and pools their events, preserving an
event-weighted estimand without materializing row-level bootstrap samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_EVENTS = 232_049
METRICS = ("hit10", "ndcg10", "hit50", "ndcg50", "hit100", "ndcg100")
SUBSETS = ("all", "organic", "recommended")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_frozen_contract(path: Path) -> tuple[dict, dict, tuple[str, ...], tuple[dict, ...]]:
    run_path = path.with_suffix(".run.json")
    if not run_path.is_file():
        raise ValueError(f"missing export run metadata: {run_path}")
    run = json.loads(run_path.read_text())
    selection_path = Path(run.get("selection_manifest", "")).expanduser()
    if not selection_path.is_file():
        raise ValueError("export metadata does not resolve to a frozen selection manifest")
    if sha256(selection_path) != run.get("selection_manifest_sha256"):
        raise ValueError("selection manifest SHA does not match export metadata")
    selection = json.loads(selection_path.read_text())
    frozen_at = selection.get("frozen_at_utc")
    if (selection.get("selection_frozen") is not True or not isinstance(frozen_at, str)
            or not frozen_at or frozen_at.startswith("FILL")):
        raise ValueError("selection manifest is not frozen")
    if selection.get("test_metric_or_per_query_results_previously_viewed") is not False:
        raise ValueError("selection manifest must disclose that model test metrics were not previously viewed")
    if selection.get("existing_test_aggregate_qa_previously_viewed") is not True:
        raise ValueError("selection manifest must disclose prior aggregate test QA")
    if selection.get("expected_test_events") != EXPECTED_EVENTS or run.get("test_event_count") != EXPECTED_EVENTS:
        raise ValueError("export does not contain the registered full-test event count")
    if run.get("test_indices_sha256") != selection.get("test_indices_sha256"):
        raise ValueError("export test-index SHA differs from the frozen selection")

    records = selection.get("policy_registry")
    if not isinstance(records, list) or not records:
        raise ValueError("frozen policy_registry is missing")
    names = []
    for record in records:
        if not isinstance(record, dict) or record.get("status") not in {"include", "omitted"}:
            raise ValueError("every policy must have an explicit include/omitted status")
        if record["status"] == "include":
            name = record.get("name")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("included policy names must be nonempty and unique")
            names.append(name)
        elif not isinstance(record.get("omission_reason"), str) or not record["omission_reason"].strip():
            raise ValueError(f"omitted policy {record.get('name')} lacks an omission reason")
    exported = tuple(run.get("policies", ()))
    if exported != tuple(names):
        raise ValueError("export policy list differs from the frozen included registry rows")

    declarations = selection.get("bootstrap_comparisons")
    if not isinstance(declarations, list) or not declarations:
        raise ValueError("frozen selection must contain explicit bootstrap_comparisons")
    ids = set()
    resolved = []
    for row in declarations:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("each bootstrap contrast requires a concrete id")
        if row["id"] in ids:
            raise ValueError("bootstrap contrast IDs must be unique")
        ids.add(row["id"])
        if any(not isinstance(row.get(key), str) or not row[key].strip()
               for key in ("estimand", "notes")):
            raise ValueError(f"contrast {row['id']} requires a nonempty estimand and notes")
        status = row.get("status")
        if status == "omitted":
            if not isinstance(row.get("omission_reason"), str) or not row["omission_reason"].strip():
                raise ValueError(f"omitted contrast {row['id']} lacks an omission reason")
            resolved.append({**row, "resolved": False})
            continue
        if status != "include":
            raise ValueError(f"contrast {row['id']} requires status include or omitted")
        policy, reference = row.get("policy"), row.get("reference")
        if policy not in names or reference not in names or policy == reference:
            raise ValueError(f"included contrast {row['id']} references an absent or identical policy pair")
        resolved.append({**row, "resolved": True})
    if not any(row["resolved"] for row in resolved):
        raise ValueError("frozen registry has no included bootstrap contrasts")
    return run, selection, exported, tuple(resolved)


def load_export(path: Path, policies: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        required = {"uid", "event_index", "supported", "organic"}
        required |= {f"{policy}_{metric}" for policy in policies for metric in METRICS}
        missing = required - set(z.files)
        if missing:
            raise ValueError(f"ranking export is missing required arrays: {sorted(missing)}")
        data = {name: z[name] for name in required}
    lengths = {len(x) for x in data.values()}
    if lengths != {EXPECTED_EVENTS}:
        raise ValueError(f"ranking export arrays must all contain {EXPECTED_EVENTS} rows")
    if len(np.unique(data["event_index"])) != EXPECTED_EVENTS:
        raise ValueError("ranking export has duplicate event indices")
    if not np.isfinite(data["uid"]).all():
        raise ValueError("uid contains non-finite values")
    return data


def paired_user_cluster_ci(uid: np.ndarray, a: np.ndarray, b: np.ndarray, *,
                           replicates: int, seed: int, chunk_size: int) -> dict:
    """Resample users in bounded chunks; each replicate remains event weighted."""
    if not len(uid):
        raise ValueError("cannot bootstrap an empty user/query subset")
    users, group = np.unique(uid, return_inverse=True)
    n_users = len(users)
    event_count = np.bincount(group, minlength=n_users).astype(np.float64)
    delta_sum = np.bincount(group, weights=np.asarray(b, np.float64) - np.asarray(a, np.float64),
                             minlength=n_users)
    rng = np.random.default_rng(seed)
    sampled_deltas = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - start)
        draws = rng.integers(0, n_users, size=(size, n_users), dtype=np.int64)
        multiplicity = np.zeros((size, n_users), dtype=np.float64)
        np.add.at(multiplicity, (np.arange(size)[:, None], draws), 1.0)
        denominator = multiplicity @ event_count
        numerator = multiplicity @ delta_sum
        sampled_deltas[start:start + size] = numerator / denominator
    return {
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "paired_difference_b_minus_a": float(np.mean(np.asarray(b, np.float64) - np.asarray(a, np.float64))),
        "user_cluster_bootstrap_95pct_ci": [float(v) for v in np.quantile(sampled_deltas, [0.025, 0.975])],
        "bootstrap_mean_difference": float(sampled_deltas.mean()),
        "events": int(len(uid)), "users": int(n_users), "replicates": replicates,
        "estimand": "event-weighted paired difference; users resampled as clusters",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.replicates < 1 or args.chunk_size < 1:
        raise ValueError("replicates and chunk-size must be positive")
    run_path = args.output.with_suffix(".run.json")
    if args.output.exists() or run_path.exists():
        raise FileExistsError("refusing to overwrite bootstrap output or run metadata")

    run, selection, policies, comparisons = load_frozen_contract(args.queries)
    data = load_export(args.queries, policies)
    result = {"all": {}, "organic": {}, "recommended": {}}
    subset_id = {name: i for i, name in enumerate(SUBSETS)}
    active_comparisons = [c for c in comparisons if c["resolved"]]
    for comparison_index, row in enumerate(active_comparisons):
        policy, reference = row["policy"], row["reference"]
        for subset in SUBSETS:
            mask = (np.ones(len(data["uid"]), dtype=bool) if subset == "all" else
                    data["organic"].astype(bool) if subset == "organic" else
                    ~data["organic"].astype(bool))
            uid = data["uid"][mask]
            result[subset][row["id"]] = {
                "policy": policy, "reference": reference,
                "metrics": {
                    metric: paired_user_cluster_ci(
                        uid,
                        data[f"{reference}_{metric}"][mask],
                        data[f"{policy}_{metric}"][mask],
                        replicates=args.replicates,
                        seed=args.seed + 100 * comparison_index + subset_id[subset],
                        chunk_size=args.chunk_size,
                    ) for metric in METRICS
                },
            }
    output = {
        "source_queries": str(args.queries.resolve()),
        "source_queries_sha256": sha256(args.queries),
        "source_export_run_json_sha256": sha256(args.queries.with_suffix(".run.json")),
        "selection_manifest": run["selection_manifest"],
        "selection_manifest_sha256": run["selection_manifest_sha256"],
        "replicates": args.replicates, "seed": args.seed,
        "chunk_size": args.chunk_size, "cluster": "uid",
        "estimand": "event-weighted paired differences; user clusters resampled",
        "policies": list(policies),
        "included_comparisons": list(active_comparisons),
        "omitted_comparisons": [row for row in comparisons if not row["resolved"]],
        "results": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    run_meta = {
        "source_queries_sha256": output["source_queries_sha256"],
        "source_export_run_json_sha256": output["source_export_run_json_sha256"],
        "selection_manifest_sha256": output["selection_manifest_sha256"],
        "bootstrap_script_sha256": sha256(Path(__file__).resolve()),
        "output_sha256": sha256(args.output),
        "replicates": args.replicates, "seed": args.seed, "chunk_size": args.chunk_size,
    }
    run_path.write_text(json.dumps(run_meta, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "contrasts": len(active_comparisons),
                      "replicates": args.replicates, "cluster": "uid"}, indent=2))


if __name__ == "__main__":
    main()
