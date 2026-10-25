"""Small Huginn-inspired recurrent item-ID model.

This is a task-scale adaptation of the published prelude/core/coda mechanism,
not a reproduction of Huginn's 3.5B pretraining recipe.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


MODES = ("R1source", "R4source", "R4self", "R4untiedsource")


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(dtype=x.dtype) * self.weight


def _apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 50_000.0) -> torch.Tensor:
    """Apply rotary embeddings to [batch, heads, length, head_width] queries/keys."""
    width = x.shape[-1]
    if width % 2:
        raise ValueError("RoPE head width must be even")
    inv_freq = 1.0 / (base ** (torch.arange(0, width, 2, device=x.device).float() / width))
    phase = positions.to(device=x.device, dtype=torch.float32)[:, None, :, None] * inv_freq[None, None, None, :]
    cos, sin = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class CausalAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width % heads or (width // heads) % 2:
            raise ValueError("width/heads must be divisible, with an even head width for RoPE")
        self.heads = heads
        self.head_width = width // heads
        self.q = nn.Linear(width, width, bias=True)
        self.k = nn.Linear(width, width, bias=True)
        self.v = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        shape = (batch, length, self.heads, self.head_width)
        q, k, v = (proj(x).view(shape).transpose(1, 2) for proj in (self.q, self.k, self.v))
        positions = torch.arange(length, device=x.device).expand(batch, length)
        q, k = _apply_rope(q, positions), _apply_rope(k, positions)
        future = torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1)
        allowed = (~future)[None, None, :, :] & (~padding_mask[:, None, None, :])
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed,
            dropout_p=self.dropout.p if self.training else 0.0,
        ).transpose(1, 2).contiguous().view(batch, length, width)
        return self.out(attended)


class SandwichBlock(nn.Module):
    """Huginn-style RMSNorm sandwich around attention and gated-SiLU MLP."""
    def __init__(self, width: int, heads: int, dropout: float, ff_mult: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attention = CausalAttention(width, heads, dropout)
        self.norm2 = RMSNorm(width)
        self.norm3 = RMSNorm(width)
        inner = ff_mult * width
        self.gate = nn.Linear(width, inner, bias=False)
        self.up = nn.Linear(width, inner, bias=False)
        self.down = nn.Linear(inner, width, bias=False)
        self.mlp_dropout = nn.Dropout(dropout)
        self.norm4 = RMSNorm(width)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm2(x + self.attention(self.norm1(x), padding_mask))
        mlp_input = self.norm3(x)
        mlp = F.silu(self.gate(mlp_input)) * self.up(mlp_input)
        return self.norm4(x + self.mlp_dropout(self.down(mlp)))


class HuginnItemGPT(nn.Module):
    """Trainable source-recurrent item predictor; forward returns catalog logits.

    Modes: one-pass source, fixed shared-source R4, matched first-pass-source/
    self-only R4, and untied four-block source R4.
    """

    def __init__(
        self,
        n_items: int,
        width: int = 128,
        heads: int = 4,
        max_context: int = 64,
        dropout: float = 0.0,
        mode: str = "R4source",
        seed: int = 73,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if n_items < 1 or max_context < 1:
            raise ValueError("n_items and max_context must be positive")
        if width % heads or (width // heads) % 2:
            raise ValueError("width/heads must be divisible, with even head width for RoPE")
        self.n_items, self.width, self.max_context = int(n_items), int(width), int(max_context)
        self.unk_id, self.bos_id, self.pad_id = 0, n_items + 1, n_items + 2
        self.mode = mode
        self.default_depth = 1 if mode == "R1source" else 4
        self.last_recurrence_depth = 4
        self.source_conditioned = mode != "R4self"
        self.untied = mode == "R4untiedsource"

        # Identically seeded modes share input/output, prelude, adapter, and
        # first-core initialization; untied core copies start equal as well.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.item_embedding = nn.Embedding(n_items + 3, width, padding_idx=self.pad_id)
            self.prelude = SandwichBlock(width, heads, dropout, ff_mult)
            self.adapter = nn.Linear(2 * width, width, bias=False)
            self.core = nn.ModuleList([SandwichBlock(width, heads, dropout, ff_mult)])
            self.core_norm = RMSNorm(width)
            self.coda = SandwichBlock(width, heads, dropout, ff_mult)
            self.coda_norm = RMSNorm(width)
            self.output_bias = nn.Parameter(torch.zeros(n_items))
            self.apply(self._init_weights)
            # The model-specific scale-aware initialization from the paper.
            with torch.no_grad():
                self.item_embedding.weight[self.pad_id].zero_()
            for module in self.modules():
                if isinstance(module, nn.Linear) and module is not self.adapter:
                    if module in self._output_projections():
                        continue
                    nn.init.trunc_normal_(module.weight, std=math.sqrt(2.0 / (5.0 * width)), a=-3 * math.sqrt(2.0 / (5.0 * width)), b=3 * math.sqrt(2.0 / (5.0 * width)))
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            out_std = math.sqrt(1.0 / (5.0 * width * 6.0))
            for projection in self._output_projections():
                nn.init.trunc_normal_(projection.weight, std=out_std, a=-3 * out_std, b=3 * out_std)
            emb_std = math.sqrt(2.0 / (5.0 * width))
            nn.init.trunc_normal_(self.item_embedding.weight, std=emb_std, a=-3 * emb_std, b=3 * emb_std)
            with torch.no_grad():
                self.item_embedding.weight[self.pad_id].zero_()
            adapter_std = math.sqrt(2.0 / (5.0 * width))
            nn.init.trunc_normal_(self.adapter.weight, std=adapter_std, a=-3 * adapter_std, b=3 * adapter_std)
            if self.untied:
                self.core = nn.ModuleList([self.core[0], *(copy.deepcopy(self.core[0]) for _ in range(3))])

    def _output_projections(self) -> list[nn.Linear]:
        return [m.out for m in self.modules() if isinstance(m, CausalAttention)] + [
            m.down for m in self.modules() if isinstance(m, SandwichBlock)
        ]

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.zeros_(module.weight)

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        recurrence_depth: int | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, length]")
        batch, length = tokens.shape
        if length > self.max_context:
            raise ValueError(f"context length {length} exceeds max_context={self.max_context}")
        if padding_mask is None:
            padding_mask = tokens.eq(self.pad_id)
        depth = self.default_depth if recurrence_depth is None else int(recurrence_depth)
        if depth < 1 or depth > 8:
            raise ValueError("recurrence_depth must be in [1, 8]")
        if self.untied and depth > len(self.core):
            raise ValueError(f"untied core has {len(self.core)} distinct blocks; depth cannot exceed it")
        self.last_recurrence_depth = depth

        embedded = self.item_embedding(tokens) * math.sqrt(self.width)
        source = self.prelude(embedded, padding_mask)
        # Random initial latent state with variance sigma_s^2=2/5.
        state_std = math.sqrt(2.0 / 5.0)
        state = torch.empty_like(source)
        nn.init.trunc_normal_(state, std=state_std, a=-3 * state_std, b=3 * state_std)
        for step in range(depth):
            block = self.core[step] if self.untied else self.core[0]
            recurrent_source = source if self.source_conditioned or step == 0 else state
            state = self.adapter(torch.cat((state, recurrent_source), dim=-1))
            state = self.core_norm(block(state, padding_mask))
        state = self.coda_norm(self.coda(state, padding_mask))[:, -1]
        candidates = self.item_embedding.weight[1 : self.n_items + 1]
        return F.linear(state, candidates, self.output_bias)
