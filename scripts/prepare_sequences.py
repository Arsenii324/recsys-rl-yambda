#!/usr/bin/env python3
"""Build compact chronological event arrays and train-only candidate map."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from recsys_tt2.catalog import build_candidate_encoding, encode_item_ids

from profile_yambda import GAP_SECONDS, LAST_TIMESTAMP, TEST_START, TRAIN_END, VAL_END, VAL_START, WINDOW_SECONDS


FIELDS = ["uid", "timestamp", "item_id", "is_organic", "played_ratio_pct", "track_length_seconds"]
SPLITS = {"train": 0, "gap": 1, "valid": 2, "gap2": 3, "test": 4, "outside": 5}


def split_for(ts: np.ndarray) -> np.ndarray:
    result = np.full(len(ts), SPLITS["outside"], dtype=np.uint8)
    result[ts < TRAIN_END] = SPLITS["train"]
    result[(ts >= TRAIN_END) & (ts < VAL_START)] = SPLITS["gap"]
    result[(ts >= VAL_START) & (ts < VAL_END)] = SPLITS["valid"]
    result[(ts >= VAL_END) & (ts < TEST_START)] = SPLITS["gap2"]
    result[(ts >= TEST_START) & (ts <= LAST_TIMESTAMP)] = SPLITS["test"]
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("parquet", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--catalog-size", type=int, default=100_000)
    p.add_argument("--train-sample-rate", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--batch-rows", type=int, default=131_072)
    args = p.parse_args()
    if args.catalog_size < 1 or not 0 < args.train_sample_rate <= 1:
        raise ValueError("catalog size and train sample rate must be positive")

    parquet = pq.ParquetFile(args.parquet)

    # Pass 1: candidate vocabulary, computed from train timestamps only.
    counts: collections.Counter[int] = collections.Counter()
    for batch in parquet.iter_batches(batch_size=args.batch_rows, columns=["item_id", "timestamp"]):
        items = batch.column(0).to_numpy(zero_copy_only=False)
        ts = batch.column(1).to_numpy(zero_copy_only=False)
        train = ts < TRAIN_END
        if train.any():
            ids, frequencies = np.unique(items[train], return_counts=True)
            counts.update(dict(zip(ids.tolist(), frequencies.tolist())))
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    if args.catalog_size > len(ranked):
        raise ValueError(f"requested {args.catalog_size} candidates but only {len(ranked)} train-seen items")
    candidate_ids = np.asarray([item for item, _ in ranked[: args.catalog_size]], dtype=np.uint32)
    sorted_ids, sorted_tokens, _, _ = build_candidate_encoding(candidate_ids)

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "candidate_item_ids.npy", candidate_ids)
    arrays = {
        "uid": np.lib.format.open_memmap(out / "uid.npy", mode="w+", dtype=np.uint32, shape=(parquet.metadata.num_rows,)),
        "timestamp": np.lib.format.open_memmap(out / "timestamp.npy", mode="w+", dtype=np.uint32, shape=(parquet.metadata.num_rows,)),
        "item_token": np.lib.format.open_memmap(out / "item_token.npy", mode="w+", dtype=np.uint32, shape=(parquet.metadata.num_rows,)),
        "played_ratio_pct": np.lib.format.open_memmap(out / "played_ratio_pct.npy", mode="w+", dtype=np.uint16, shape=(parquet.metadata.num_rows,)),
        "track_length_seconds": np.lib.format.open_memmap(out / "track_length_seconds.npy", mode="w+", dtype=np.uint16, shape=(parquet.metadata.num_rows,)),
        "is_organic": np.lib.format.open_memmap(out / "is_organic.npy", mode="w+", dtype=np.uint8, shape=(parquet.metadata.num_rows,)),
        "split": np.lib.format.open_memmap(out / "split.npy", mode="w+", dtype=np.uint8, shape=(parquet.metadata.num_rows,)),
        "context_end": np.lib.format.open_memmap(out / "context_end.npy", mode="w+", dtype=np.uint32, shape=(parquet.metadata.num_rows,)),
        "history_start": np.lib.format.open_memmap(out / "history_start.npy", mode="w+", dtype=np.uint32, shape=(parquet.metadata.num_rows,)),
        "tie_group_size": np.lib.format.open_memmap(out / "tie_group_size.npy", mode="w+", dtype=np.uint16, shape=(parquet.metadata.num_rows,)),
    }
    max_train_samples = 6_000_000
    train_indices = np.lib.format.open_memmap(out / ".train_indices_buffer.npy", mode="w+", dtype=np.uint32, shape=(max_train_samples,))
    val_capacity = 1_000_000
    test_capacity = 1_000_000
    val_indices = np.lib.format.open_memmap(out / ".valid_indices_buffer.npy", mode="w+", dtype=np.uint32, shape=(val_capacity,))
    test_indices = np.lib.format.open_memmap(out / ".test_indices_buffer.npy", mode="w+", dtype=np.uint32, shape=(test_capacity,))

    cursor = 0
    user_index = 0
    train_n = val_n = test_n = 0
    exact_duplicate_rows = 0
    tie_groups = 0
    max_tie_group = 0
    current_uid: int | None = None
    user_parts: list[list[np.ndarray]] = [[] for _ in FIELDS]

    def flush_user() -> None:
        nonlocal cursor, user_index, train_n, val_n, test_n
        nonlocal exact_duplicate_rows, tie_groups, max_tie_group, current_uid, user_parts
        if current_uid is None:
            return
        uid, ts, items, organic, pct, length = [np.concatenate(parts) for parts in user_parts]
        permutation = np.lexsort((items, ts))
        ts, items = ts[permutation], items[permutation]
        organic, pct, length = organic[permutation], pct[permutation], length[permutation]
        n = len(ts)
        local_splits = split_for(ts)
        tokens = encode_item_ids(items, sorted_ids, sorted_tokens)
        supported = tokens > 0

        starts = np.r_[0, np.flatnonzero(ts[1:] != ts[:-1]) + 1]
        ends = np.r_[starts[1:], n]
        group_sizes = (ends - starts).astype(np.uint16)
        context_end = np.repeat(starts.astype(np.uint32) + cursor, group_sizes.astype(np.int64))
        tie_groups += int(np.count_nonzero(group_sizes > 1))
        max_tie_group = max(max_tie_group, int(group_sizes.max(initial=0)))
        exact_duplicate_rows += int(np.count_nonzero((ts[1:] == ts[:-1]) & (items[1:] == items[:-1])))

        arrays["uid"][cursor : cursor + n] = uid
        arrays["timestamp"][cursor : cursor + n] = ts
        arrays["item_token"][cursor : cursor + n] = tokens
        arrays["played_ratio_pct"][cursor : cursor + n] = pct
        arrays["track_length_seconds"][cursor : cursor + n] = length
        arrays["is_organic"][cursor : cursor + n] = organic
        arrays["split"][cursor : cursor + n] = local_splits
        arrays["context_end"][cursor : cursor + n] = context_end
        arrays["history_start"][cursor : cursor + n] = cursor
        arrays["tie_group_size"][cursor : cursor + n] = group_sizes.repeat(group_sizes.astype(np.int64))

        user_train = np.flatnonzero((local_splits == SPLITS["train"]) & supported)
        if len(user_train):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, user_index]))
            user_train = user_train[rng.random(len(user_train)) < args.train_sample_rate]
        if train_n + len(user_train) > max_train_samples:
            raise RuntimeError("train index capacity exceeded")
        train_indices[train_n : train_n + len(user_train)] = cursor + user_train
        train_n += len(user_train)

        val_local = np.flatnonzero(local_splits == SPLITS["valid"])
        test_local = np.flatnonzero(local_splits == SPLITS["test"])
        if val_n + len(val_local) > val_capacity or test_n + len(test_local) > test_capacity:
            raise RuntimeError("validation/test index capacity exceeded")
        val_indices[val_n : val_n + len(val_local)] = cursor + val_local
        test_indices[test_n : test_n + len(test_local)] = cursor + test_local
        val_n += len(val_local)
        test_n += len(test_local)

        cursor += n
        user_index += 1
        user_parts = [[] for _ in FIELDS]

    for batch in parquet.iter_batches(batch_size=args.batch_rows, columns=FIELDS):
        cols = [batch.column(i).to_numpy(zero_copy_only=False) for i in range(len(FIELDS))]
        uids = cols[0]
        boundaries = np.r_[0, np.flatnonzero(uids[1:] != uids[:-1]) + 1, len(uids)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            row_uid = int(uids[start])
            if current_uid is None:
                current_uid = row_uid
            elif row_uid != current_uid:
                flush_user()
                current_uid = row_uid
            for parts, column in zip(user_parts, cols):
                parts.append(column[start:end])
    flush_user()

    for array in arrays.values():
        array.flush()
    train_indices.flush()
    val_indices.flush()
    test_indices.flush()
    train_slice = np.asarray(train_indices[:train_n]).copy()
    val_slice = np.asarray(val_indices[:val_n]).copy()
    test_slice = np.asarray(test_indices[:test_n]).copy()
    del train_indices, val_indices, test_indices
    np.save(out / "train_indices.npy", train_slice)
    np.save(out / "valid_indices.npy", val_slice)
    np.save(out / "test_indices.npy", test_slice)
    for name in (".train_indices_buffer.npy", ".valid_indices_buffer.npy", ".test_indices_buffer.npy"):
        (out / name).unlink(missing_ok=True)
    manifest = {
        "source": "yandex/yambda flat/50m/listens.parquet",
        "revision": "dd6f3a19eef5866e346c3270e098baa641a44948",
        "source_sha256": sha256(args.parquet),
        "catalog_size": args.catalog_size,
        "train_sample_rate": args.train_sample_rate,
        "seed": args.seed,
        "events": int(cursor),
        "users": int(user_index),
        "train_anchor_count": int(train_n),
        "train_anchor_sampling": "independent Bernoulli sample over supported train events; event-uniform in expectation",
        "validation_event_count": int(val_n),
        "test_event_count": int(test_n),
        "exact_duplicate_rows_within_user_timestamp_item": int(exact_duplicate_rows),
        "user_timestamp_tie_groups": int(tie_groups),
        "max_tie_group_size": int(max_tie_group),
        "split_codes": SPLITS,
        "protocol": {
            "train_end_exclusive": TRAIN_END,
            "valid_start_inclusive": VAL_START,
            "valid_end_exclusive": VAL_END,
            "test_start_inclusive": TEST_START,
            "test_end_inclusive": LAST_TIMESTAMP,
            "gap_seconds": GAP_SECONDS,
            "window_seconds": WINDOW_SECONDS,
            "history_rule": "strictly timestamp < target timestamp; tied targets share context",
            "tie_history_order": "item_id ascending, disclosed canonical order for later histories only",
            "target_catalog": "top training-frequency items; unsupported targets remain index rows with item_token=0 and score as misses",
            "history_oov": "item_token=0 retained as an event in history",
        },
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                  for path in sorted(out.glob("*.npy"))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
