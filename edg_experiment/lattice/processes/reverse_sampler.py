"""Checkpoint-compatible direct-q CTMC sampler for lattice evaluation."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

from edg_experiment.dfm.schedules import KappaSchedule
from edg_experiment.lattice.problem import build_problem
from edg_experiment.lattice.processes.analytic_base import teacher_posterior


def _sample_probs(probs: Tensor) -> Tensor:
    if probs.ndim != 3:
        raise ValueError(f"probs must be [B,D,K], got {tuple(probs.shape)}")
    batch_size, length, num_states = probs.shape
    flat = probs.reshape(batch_size * length, num_states)
    row_sum = flat.sum(dim=-1, keepdim=True)
    fallback = torch.zeros_like(flat)
    fallback[:, 0] = 1.0
    flat = torch.where(row_sum > 0.0, flat, fallback)
    return torch.multinomial(flat, num_samples=1).reshape(batch_size, length)


@torch.no_grad()
def teacher_reverse_direct_q_ctmc_rollout(
    theta_prev,
    phi_k,
    cfg: Dict[str, Any],
    schedule: KappaSchedule,
    device: torch.device,
    batch_size: int | None = None,
    guidance_strength: float | None = None,
    return_debug: bool = True,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> Tuple[Tensor, Dict[str, float]]:
    rollout_cfg = cfg["rollout"]
    num_states = int(cfg["vocab"]["num_states"])
    if num_states < 2:
        raise ValueError("vocab.num_states must be at least 2")

    length = build_problem(cfg).length
    num_steps = int(rollout_cfg["num_steps"])
    if num_steps <= 0:
        raise ValueError("rollout.num_steps must be positive")
    size = int(batch_size or rollout_cfg["batch_size"])
    strength = float(
        rollout_cfg.get("guidance_strength", 1.0)
        if guidance_strength is None
        else guidance_strength
    )
    step_size = 1.0 / float(num_steps)

    x_t = torch.randint(0, num_states, (size, length), device=device, dtype=torch.long)
    rates: list[float] = []
    jumps: list[float] = []
    changes: list[float] = []
    stay_probs: list[float] = []

    iterator = tqdm(
        range(num_steps),
        desc=progress_desc or "reverse direct-q ctmc",
        disable=not show_progress,
        leave=False,
    )
    for step_idx in iterator:
        previous = x_t
        t = torch.full(
            (size,), step_idx / float(num_steps), device=device, dtype=torch.float32
        )
        posterior = teacher_posterior(
            theta_prev, phi_k, previous, t, guidance_strength=strength
        ).to(torch.float32)
        coefficient = schedule.rate_coeff(t).unsqueeze(-1).unsqueeze(-1)
        current = F.one_hot(previous, num_classes=num_states).to(torch.float32)
        jump_rates = coefficient * posterior * (1.0 - current)
        intensity = jump_rates.sum(dim=-1)
        jump_probability = torch.clamp(
            1.0 - torch.exp(-step_size * intensity), 0.0, 1.0
        )
        jump_mask = torch.rand_like(jump_probability) < jump_probability
        jump_tokens = _sample_probs(
            jump_rates / intensity.unsqueeze(-1).clamp_min(1.0e-12)
        )
        x_t = torch.where(jump_mask, jump_tokens, previous)

        rates.append(float(intensity.mean().item()))
        jumps.append(float(jump_mask.float().mean().item()))
        changes.append(float(x_t.ne(previous).float().mean().item()))
        stay_probs.append(float((posterior * current).sum(dim=-1).mean().item()))

    if not return_debug:
        return x_t, {}
    return x_t, {
        "mean_jump_rate": float(sum(rates) / max(len(rates), 1)),
        "mean_jump_fraction": float(sum(jumps) / max(len(jumps), 1)),
        "mean_state_change_fraction": float(sum(changes) / max(len(changes), 1)),
        "mean_posterior_stay_prob": float(
            sum(stay_probs) / max(len(stay_probs), 1)
        ),
        "final_state_mean": float(x_t.to(torch.float32).mean().item()),
    }


__all__ = ["teacher_reverse_direct_q_ctmc_rollout"]
