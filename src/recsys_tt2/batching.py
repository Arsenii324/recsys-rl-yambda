"""On-demand context construction from compact chronological event arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from .catalog import build_candidate_encoding


ARRAY_NAMES = (
    "uid",
    "timestamp",
    "item_token",
    "played_ratio_pct",
    "track_length_seconds",
    "is_organic",
    "split",
    "context_end",
    "history_start",
    "tie_group_size",
)


def open_event_arrays(path: str | Path) -> dict[str, np.ndarray]:
    root = Path(path)
    arrays = {name: np.load(root / f"{name}.npy", mmap_mode="r") for name in ARRAY_NAMES}
    # Reconstruct the exact legacy encode-v1 class map and its train-only rank.
    candidate_item_ids = np.load(root / "candidate_item_ids.npy", mmap_mode="r")
    _, _, class_to_raw, class_to_rank = build_candidate_encoding(candidate_item_ids)
    arrays["candidate_raw_item_id"] = class_to_raw
    arrays["candidate_popularity_rank"] = class_to_rank
    return arrays


def make_batch(
    indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    *,
    n_items: int,
    max_context: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Gather a batch; every context stops before its target timestamp block."""

    idx = np.asarray(indices, dtype=np.int64)
    batch = len(idx)
    pad_id, bos_id = n_items + 2, n_items + 1
    tokens = np.full((batch, max_context), pad_id, dtype=np.int64)
    rm_history = np.full((batch, max_context), -1, dtype=np.int64)
    context_ends = arrays["context_end"][idx].astype(np.int64, copy=False)
    history_starts = arrays["history_start"][idx].astype(np.int64, copy=False)
    all_items = arrays["item_token"]
    for row, (start, end) in enumerate(zip(history_starts, context_ends)):
        start = max(int(start), int(end) - max_context)
        end = int(end)
        length = end - start
        if length == 0:
            tokens[row, -1] = bos_id
            continue
        history = np.asarray(all_items[start:end], dtype=np.int64)
        tokens[row, -length:] = history
        rm_history[row, -length:] = history

    target_token = np.asarray(arrays["item_token"][idx], dtype=np.int64)
    supported = target_token > 0
    target_class = np.maximum(target_token - 1, 0)
    target_popularity_rank = np.asarray(arrays["candidate_popularity_rank"][target_class], dtype=np.int64)
    raw_ratio = np.asarray(arrays["played_ratio_pct"][idx], dtype=np.float32)
    reward = np.minimum(raw_ratio / 100.0, 1.0)
    duration = np.asarray(arrays["track_length_seconds"][idx], dtype=np.float32)
    organic = np.asarray(arrays["is_organic"][idx], dtype=np.int64)
    uid = np.asarray(arrays["uid"][idx], dtype=np.int64)
    timestamp = np.asarray(arrays["timestamp"][idx], dtype=np.int64)
    result = {
        "tokens": torch.as_tensor(tokens, dtype=torch.long, device=device),
        "padding_mask": torch.as_tensor(tokens == pad_id, dtype=torch.bool, device=device),
        "rm_history": torch.as_tensor(rm_history, dtype=torch.long, device=device),
        "target": torch.as_tensor(target_class, dtype=torch.long, device=device),
        "target_popularity_rank": torch.as_tensor(target_popularity_rank, dtype=torch.long, device=device),
        "supported": torch.as_tensor(supported, dtype=torch.bool, device=device),
        "reward": torch.as_tensor(reward, dtype=torch.float32, device=device),
        "raw_ratio": torch.as_tensor(raw_ratio, dtype=torch.float32, device=device),
        "duration": torch.as_tensor(duration, dtype=torch.float32, device=device),
        "organic": torch.as_tensor(organic, dtype=torch.bool, device=device),
        "uid": torch.as_tensor(uid, dtype=torch.long, device=device),
        "timestamp": torch.as_tensor(timestamp, dtype=torch.long, device=device),
    }
    return result


def batches(indices: np.ndarray, batch_size: int, *, shuffle: bool = False, seed: int = 0):
    order = np.asarray(indices, dtype=np.uint32)
    if shuffle:
        order = np.random.default_rng(seed).permutation(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]
