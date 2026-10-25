"""A small history-dependent gate between a frozen GPT and repeat channel."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def frozen_last_state(base: nn.Module, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Return the seed model's exact final normalized state, without training it."""
    if base.training:
        raise RuntimeError("frozen GPT must remain in eval mode")
    if any(parameter.requires_grad for parameter in base.parameters()):
        raise RuntimeError("frozen GPT parameters must have requires_grad=False")
    positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    x = base.dropout(base.item_embedding(tokens) + base.position_embedding(positions))
    for block in base.blocks:
        x = block(x, padding_mask)
    return base.norm(x[:, -1])


def history_repeat_stats(tokens: torch.Tensor, n_items: int) -> torch.Tensor:
    """Summarize known-ID repetition over the final 64 history positions.

    Features are known-position fraction, duplicate-event fraction, and maximum
    within-window item frequency. UNK/BOS/PAD are excluded from all statistics.
    """
    recent = tokens[:, -64:]
    known = recent.ge(1) & recent.le(n_items)
    known_f = known.to(torch.float32)
    total = known_f.sum(dim=1)
    same = recent[:, :, None].eq(recent[:, None, :]) & known[:, :, None] & known[:, None, :]
    earlier = torch.tril(torch.ones((recent.shape[1], recent.shape[1]),
                                    dtype=torch.bool, device=tokens.device), diagonal=-1)
    has_earlier_copy = (same & earlier[None]).any(dim=2)
    unique = known & ~has_earlier_copy
    unique_count = unique.sum(dim=1).to(torch.float32)
    counts_per_position = same.sum(dim=2).to(torch.float32)
    max_count = counts_per_position.masked_fill(~known, 0).max(dim=1).values
    denom = total.clamp_min(1.0)
    return torch.stack((total / 64.0, (total - unique_count) / denom, max_count / denom), dim=1)


class HistoryGate(nn.Module):
    """Small sigmoid gate; zero final layer initializes exactly to 50/50."""

    def __init__(self, width: int, hidden: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(width + 3, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, history_state: torch.Tensor, repeat_stats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(torch.cat((history_state, repeat_stats), dim=-1))).squeeze(-1)


def mix_distributions(gpt_probs: torch.Tensor, history_tokens: torch.Tensor, n_items: int,
                      history_weight: float | torch.Tensor) -> torch.Tensor:
    """Mix GPT probabilities with frequencies over the final 64 known IDs."""
    recent = history_tokens[:, -64:]
    known = recent.ge(1) & recent.le(n_items)
    classes = (recent - 1).clamp(min=0, max=n_items - 1)
    counts = torch.zeros_like(gpt_probs)
    counts.scatter_add_(1, classes, known.to(gpt_probs.dtype))
    totals = counts.sum(dim=1, keepdim=True)
    recent_probs = counts / totals.clamp_min(1.0)
    weight = torch.as_tensor(history_weight, device=gpt_probs.device, dtype=gpt_probs.dtype)
    if weight.ndim == 0:
        weight = weight.expand(gpt_probs.shape[0])
    if weight.shape != (gpt_probs.shape[0],):
        raise ValueError("history_weight must be scalar or one value per query")
    if bool(((weight < 0) | (weight > 1)).any()):
        raise ValueError("history_weight must be in [0, 1]")
    mixed = (1.0 - weight[:, None]) * gpt_probs + weight[:, None] * recent_probs
    return torch.where(totals > 0, mixed, gpt_probs)


def mixture_target_logprob(gpt_logits: torch.Tensor, target: torch.Tensor,
                           history_tokens: torch.Tensor, history_weight: torch.Tensor,
                           n_items: int) -> torch.Tensor:
    """Stable CE log-probability for a GPT/history mixture without dense counts."""
    logp_gpt = F.log_softmax(gpt_logits.float(), dim=-1).gather(1, target[:, None]).squeeze(1)
    recent = history_tokens[:, -64:]
    known = recent.ge(1) & recent.le(n_items)
    total = known.sum(dim=1).to(torch.float32)
    target_count = (known & recent.eq(target[:, None] + 1)).sum(dim=1).to(torch.float32)
    neg_inf = torch.full_like(total, -torch.inf)
    logp_history = torch.where(target_count > 0,
                                target_count.clamp_min(1).log() - total.clamp_min(1).log(), neg_inf)
    weight = history_weight.float().clamp(1e-7, 1.0 - 1e-7)
    logp = torch.logaddexp(torch.log1p(-weight) + logp_gpt,
                           torch.log(weight) + logp_history)
    # With no known history, the protocol falls back exactly to GPT.
    return torch.where(total > 0, logp, logp_gpt)
