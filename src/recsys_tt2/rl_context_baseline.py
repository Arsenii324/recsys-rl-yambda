"""Detached, action-independent history baseline for the registered RL control."""

from __future__ import annotations

from pathlib import Path

import torch

from .model import ConditionalCompletionRM


def load_frozen_history_rm(path: Path, *, device: torch.device, expected_n_items: int,
                           expected_seed: int | None = None) -> ConditionalCompletionRM:
    """Load the registered history-only RM without advancing the action RNG stream."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("variant") != "history":
        raise ValueError("context baseline checkpoint must be variant=history")
    config = payload.get("config", {})
    if int(config.get("n_items", -1)) != expected_n_items:
        raise ValueError("history baseline catalog size does not match policy")
    if expected_seed is not None and int(config.get("seed", -1)) != expected_seed:
        raise ValueError("history baseline checkpoint seed does not match registered seed")
    # Module initialization consumes the global CPU RNG. Keep it outside the
    # policy's seeded categorical-sampling stream; weights are immediately restored.
    with torch.random.fork_rng(devices=[]):
        model = ConditionalCompletionRM(expected_n_items, width=int(config["width"]), variant="history")
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def frozen_history_baseline(reward_model: torch.nn.Module, history_tokens: torch.Tensor) -> torch.Tensor:
    """Score each history with the frozen history-only RM; accepts no action inputs."""
    if getattr(reward_model, "variant", None) != "history":
        raise ValueError("context baseline requires a history-only reward model")
    if history_tokens.ndim != 2:
        raise ValueError("history_tokens must have shape [batch, context]")
    reward_model.eval()
    for parameter in reward_model.parameters():
        parameter.requires_grad_(False)
    candidates = torch.zeros(history_tokens.shape[0], dtype=torch.long, device=history_tokens.device)
    durations = torch.zeros(history_tokens.shape[0], dtype=torch.float32, device=history_tokens.device)
    return reward_model(history_tokens, candidates, durations).detach()


def detached_advantage(sampled_reward: torch.Tensor, context_baseline: torch.Tensor) -> torch.Tensor:
    """Subtract a detached per-context baseline without coupling its model to policy gradients."""
    if sampled_reward.shape != context_baseline.shape:
        raise ValueError("reward and context baseline must have identical per-example shapes")
    return (sampled_reward - context_baseline.detach()).detach()
