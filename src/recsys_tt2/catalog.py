"""Train-only catalog mapping for the frozen Yambda encode-v1 artifact."""
from __future__ import annotations
import numpy as np


def build_candidate_encoding(candidate_item_ids: np.ndarray):
    """Build the exact raw-ID search arrays and class metadata used by prep.

    ``candidate_item_ids`` is ordered by descending training count (item ID tie
    break). Legacy encode-v1 maps a raw-ID-sorted search position through the
    historical ``sorted_tokens[order] = 1..N`` assignment. Preserve it exactly
    so existing event tokens/checkpoints remain compatible.
    """
    ids = np.asarray(candidate_item_ids)
    order = np.argsort(ids)
    sorted_ids = ids[order]
    sorted_tokens = np.empty(len(ids), dtype=np.uint32)
    sorted_tokens[order] = np.arange(1, len(ids) + 1, dtype=np.uint32)
    class_to_raw_id = np.empty(len(ids), dtype=ids.dtype)
    class_to_popularity_rank = np.empty(len(ids), dtype=np.uint32)
    class_to_raw_id[sorted_tokens - 1] = sorted_ids
    class_to_popularity_rank[sorted_tokens - 1] = order.astype(np.uint32)
    if not np.array_equal(np.sort(class_to_popularity_rank), np.arange(len(ids), dtype=np.uint32)):
        raise ValueError("candidate popularity-rank map is not a permutation")
    if not np.array_equal(ids[class_to_popularity_rank], class_to_raw_id):
        raise ValueError("candidate raw-ID/rank round trip failed")
    return sorted_ids, sorted_tokens, class_to_raw_id, class_to_popularity_rank


def encode_item_ids(items: np.ndarray, sorted_ids: np.ndarray, sorted_tokens: np.ndarray) -> np.ndarray:
    """Encode raw item IDs to legacy candidate classes; OOV values become UNK=0."""
    items = np.asarray(items)
    positions = np.searchsorted(sorted_ids, items)
    in_bounds = positions < len(sorted_ids)
    supported = np.zeros(len(items), dtype=bool)
    supported[in_bounds] = sorted_ids[positions[in_bounds]] == items[in_bounds]
    tokens = np.zeros(len(items), dtype=np.uint32)
    tokens[supported] = sorted_tokens[positions[supported]]
    return tokens
