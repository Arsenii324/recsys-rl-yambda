"""Small GPT-2-style causal item-ID policy and conditional-completion model."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_width)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (part.transpose(1, 2) for part in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_width)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        scores = scores.masked_fill(padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        attended = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.proj(attended)


class CausalBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float, ff_mult: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = CausalSelfAttention(width, heads, dropout)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, ff_mult * width),
            nn.GELU(),
            nn.Linear(ff_mult * width, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), padding_mask)
        return x + self.mlp(self.norm2(x))


class ItemGPT(nn.Module):
    """Causal transformer; output class j is candidate token j+1."""

    def __init__(
        self,
        n_items: int,
        width: int = 128,
        layers: int = 2,
        heads: int = 4,
        max_context: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_items < 1 or layers < 1 or max_context < 1:
            raise ValueError("n_items, layers and max_context must be positive")
        self.n_items = n_items
        self.width = width
        self.max_context = max_context
        self.unk_id = 0
        self.bos_id = n_items + 1
        self.pad_id = n_items + 2
        self.item_embedding = nn.Embedding(n_items + 3, width, padding_idx=self.pad_id)
        self.position_embedding = nn.Embedding(max_context, width)
        self.blocks = nn.ModuleList([CausalBlock(width, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.output_bias = nn.Parameter(torch.zeros(n_items))
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)
        with torch.no_grad():
            self.item_embedding.weight[self.pad_id].zero_()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, length]")
        batch, length = tokens.shape
        if length > self.max_context:
            raise ValueError(f"context length {length} exceeds max_context={self.max_context}")
        if padding_mask is None:
            padding_mask = tokens.eq(self.pad_id)
        positions = torch.arange(length, device=tokens.device).unsqueeze(0)
        x = self.dropout(self.item_embedding(tokens) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x, padding_mask)
        # Batches are left-padded, so the last token is always the final observed token or BOS.
        state = self.norm(x[:, -1])
        candidate_weights = self.item_embedding.weight[1 : self.n_items + 1]
        return F.linear(state, candidate_weights, self.output_bias)


class ConditionalCompletionRM(nn.Module):
    """Predict clipped completion conditional on an observed listen.

    `variant` is one of history, item, or combined. All scores are associative
    predictions on logged listens and are not counterfactual response probabilities.
    """

    def __init__(self, n_items: int, width: int = 32, variant: str = "combined") -> None:
        super().__init__()
        if variant not in {"history", "item", "combined"}:
            raise ValueError(f"unknown reward-model variant: {variant}")
        self.n_items = n_items
        self.variant = variant
        self.item_embedding = nn.Embedding(n_items + 1, width, padding_idx=0)
        input_width = (width if variant in {"history", "combined"} else 0)
        input_width += (width + 2 if variant in {"item", "combined"} else 0)
        self.mlp = nn.Sequential(nn.Linear(input_width, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(
        self,
        history_tokens: torch.Tensor,
        candidate_class: torch.Tensor,
        duration_seconds: torch.Tensor,
        popularity_rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features: list[torch.Tensor] = []
        if self.variant in {"history", "combined"}:
            valid = history_tokens.ne(-1)
            pooled = self.item_embedding(history_tokens.clamp(min=0)).masked_fill(~valid.unsqueeze(-1), 0.0)
            denom = valid.sum(dim=1, keepdim=True).clamp(min=1)
            features.append(pooled.sum(dim=1) / denom)
        if self.variant in {"item", "combined"}:
            features.append(self.item_embedding(candidate_class + 1))
            duration = torch.log1p(duration_seconds.float()).unsqueeze(-1) / math.log1p(2_500)
            rank_ids = candidate_class if popularity_rank is None else popularity_rank
            rank = rank_ids.float().unsqueeze(-1) / max(1, self.n_items - 1)
            features.extend((duration, rank))
        return torch.sigmoid(self.mlp(torch.cat(features, dim=-1))).squeeze(-1)
