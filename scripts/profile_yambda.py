#!/usr/bin/env python3
"""Bounded-memory audit of the pinned Yambda-50M listen stream."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


LAST_TIMESTAMP = 26_000_000
GAP_SECONDS = 1_800
WINDOW_SECONDS = 86_400
TEST_START = LAST_TIMESTAMP - WINDOW_SECONDS
VAL_START = TEST_START - GAP_SECONDS - WINDOW_SECONDS
VAL_END = VAL_START + WINDOW_SECONDS
TRAIN_END = VAL_START - GAP_SECONDS


def split_mask(ts: np.ndarray, name: str) -> np.ndarray:
    if name == "train":
        return ts < TRAIN_END
    if name == "valid":
        return (ts >= VAL_START) & (ts < VAL_END)
    if name == "test":
        return (ts >= TEST_START) & (ts <= LAST_TIMESTAMP)
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=131_072)
    args = parser.parse_args()

    parquet = pq.ParquetFile(args.parquet)
    required = ["uid", "item_id", "timestamp", "is_organic", "played_ratio_pct", "track_length_seconds"]
    counters = {s: {"rows": 0, "min_ts": None, "max_ts": None, "null_ratio": 0, "ratio_sum": 0.0,
                    "ratio_min": None, "ratio_max": None, "ratio_gt_100": 0, "ratio_lt_0": 0,
                    "track_length_null": 0, "track_length_min": None, "track_length_max": None,
                    "organic": 0, "non_organic": 0} for s in ("train", "valid", "test", "gap_or_outside")}
    item_counts: collections.Counter[int] = collections.Counter()
    user_split_counts = {s: collections.Counter() for s in ("train", "valid", "test")}
    tie_extra_rows = 0
    tie_groups = 0
    ordering_violations = 0
    previous_uid: int | None = None
    previous_ts: int | None = None

    for batch in parquet.iter_batches(batch_size=args.batch_rows, columns=required):
        uids = batch.column(0).to_numpy(zero_copy_only=False)
        items = batch.column(1).to_numpy(zero_copy_only=False)
        ts = batch.column(2).to_numpy(zero_copy_only=False)
        organic = batch.column(3).to_numpy(zero_copy_only=False)
        ratio_col = batch.column(4)
        lengths_col = batch.column(5)
        ratio = ratio_col.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        lengths = lengths_col.to_numpy(zero_copy_only=False)
        ratio_valid = ratio_col.is_valid().to_numpy(zero_copy_only=False)
        lengths_valid = lengths_col.is_valid().to_numpy(zero_copy_only=False)

        if previous_uid is not None and len(uids):
            if int(uids[0]) < previous_uid or (int(uids[0]) == previous_uid and int(ts[0]) < previous_ts):
                ordering_violations += 1
        if len(uids) > 1:
            ordering_violations += int(np.count_nonzero((uids[1:] < uids[:-1]) | ((uids[1:] == uids[:-1]) & (ts[1:] < ts[:-1]))))
            eq = (uids[1:] == uids[:-1]) & (ts[1:] == ts[:-1])
            tie_extra_rows += int(eq.sum())
            tie_groups += int(np.count_nonzero(eq & ~np.r_[False, eq[:-1]]))
        if previous_uid is not None and len(uids) and int(uids[0]) == previous_uid and int(ts[0]) == previous_ts:
            tie_extra_rows += 1
            if len(uids) == 1 or int(uids[1]) != previous_uid or int(ts[1]) != previous_ts:
                tie_groups += 1
        if len(uids):
            previous_uid, previous_ts = int(uids[-1]), int(ts[-1])

        assigned = np.zeros(len(ts), dtype=bool)
        for name in ("train", "valid", "test"):
            mask = split_mask(ts, name)
            assigned |= mask
            state = counters[name]
            n = int(mask.sum())
            if not n:
                continue
            state["rows"] += n
            lo, hi = int(ts[mask].min()), int(ts[mask].max())
            state["min_ts"] = lo if state["min_ts"] is None else min(state["min_ts"], lo)
            state["max_ts"] = hi if state["max_ts"] is None else max(state["max_ts"], hi)
            rv = ratio[mask]
            valid = ratio_valid[mask]
            state["null_ratio"] += int((~valid).sum())
            if valid.any():
                values = rv[valid]
                state["ratio_sum"] += float(values.sum())
                lo, hi = float(values.min()), float(values.max())
                state["ratio_min"] = lo if state["ratio_min"] is None else min(state["ratio_min"], lo)
                state["ratio_max"] = hi if state["ratio_max"] is None else max(state["ratio_max"], hi)
                state["ratio_gt_100"] += int((values > 100).sum())
                state["ratio_lt_0"] += int((values < 0).sum())
            lens = lengths[mask & lengths_valid]
            state["track_length_null"] += int(n - len(lens))
            if len(lens):
                lo, hi = int(lens.min()), int(lens.max())
                state["track_length_min"] = lo if state["track_length_min"] is None else min(state["track_length_min"], lo)
                state["track_length_max"] = hi if state["track_length_max"] is None else max(state["track_length_max"], hi)
            state["organic"] += int(organic[mask].sum())
            state["non_organic"] += int(n - organic[mask].sum())
            user_ids, user_n = np.unique(uids[mask], return_counts=True)
            user_split_counts[name].update(dict(zip(user_ids.tolist(), user_n.tolist())))
            if name == "train":
                item_ids, item_n = np.unique(items[mask], return_counts=True)
                item_counts.update(dict(zip(item_ids.tolist(), item_n.tolist())))
        gap_n = int((~assigned).sum())
        counters["gap_or_outside"]["rows"] += gap_n

    ranked = sorted(item_counts.items(), key=lambda pair: (-pair[1], pair[0]))
    catalog_sizes = [1_000, 10_000, 20_000, 50_000, 100_000, len(ranked)]
    catalog = {k: {item for item, _ in ranked[:k]} for k in catalog_sizes}

    # Second bounded pass: compute temporal target support against train-only catalogs.
    support = {s: {k: 0 for k in catalog_sizes} for s in ("valid", "test")}
    supported_events = {s: 0 for s in ("valid", "test")}
    for batch in parquet.iter_batches(batch_size=args.batch_rows, columns=["item_id", "timestamp"]):
        items = batch.column(0).to_numpy(zero_copy_only=False)
        ts = batch.column(1).to_numpy(zero_copy_only=False)
        for name in ("valid", "test"):
            mask = split_mask(ts, name)
            if not mask.any():
                continue
            supported_events[name] += int(mask.sum())
            for k, ids in catalog.items():
                support[name][k] += sum(int(item in ids) for item in items[mask])

    result = {
        "source": {
            "dataset": "yandex/yambda",
            "revision": "dd6f3a19eef5866e346c3270e098baa641a44948",
            "path": "flat/50m/listens.parquet",
            "file_bytes": args.parquet.stat().st_size,
            "rows_metadata": parquet.metadata.num_rows,
            "row_groups": parquet.metadata.num_row_groups,
            "schema": str(parquet.schema_arrow),
        },
        "split_protocol": {
            "LAST_TIMESTAMP": LAST_TIMESTAMP,
            "GAP_SECONDS": GAP_SECONDS,
            "VAL_SECONDS": WINDOW_SECONDS,
            "TEST_SECONDS": WINDOW_SECONDS,
            "TRAIN_END_EXCLUSIVE": TRAIN_END,
            "VAL_START_INCLUSIVE": VAL_START,
            "VAL_END_EXCLUSIVE": VAL_END,
            "TEST_START_INCLUSIVE": TEST_START,
            "TEST_END_INCLUSIVE": LAST_TIMESTAMP,
        },
        "counts": counters,
        "ordering_violations": ordering_violations,
        "same_user_same_timestamp_extra_rows": tie_extra_rows,
        "same_user_timestamp_tie_groups": tie_groups,
        "train_users_by_split": {s: len(c) for s, c in user_split_counts.items()},
        "history_events_per_user": {s: {
            "min": min(c.values()) if c else 0,
            "median": float(np.median(list(c.values()))) if c else 0,
            "p90": float(np.percentile(list(c.values()), 90)) if c else 0,
            "max": max(c.values()) if c else 0,
        } for s, c in user_split_counts.items()},
        "train_catalog_items": len(ranked),
        "train_catalog_top_counts": {str(k): ranked[k - 1][1] if len(ranked) >= k else None for k in catalog_sizes if k <= len(ranked)},
        "heldout_target_support": {s: {str(k): {"supported": n, "total": supported_events[s], "fraction": n / supported_events[s]}
                                       for k, n in by_k.items()} for s, by_k in support.items()},
        "target_filtering": "none; heldout targets outside train-only catalog are misses in the all-target denominator",
        "history_ties": "same uid+timestamp events are never included in one another's histories",
        "batch_rows": args.batch_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
