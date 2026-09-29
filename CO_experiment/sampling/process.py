from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from edg_experiment.dfm.schedules import KappaSchedule


def analytic_uniform_replace_posterior(
    xt: Tensor,
    t: Tensor,
    *,
    schedule: KappaSchedule,
) -> Tensor:
    if xt.ndim != 2:
        raise ValueError(f"xt must be [B,N], got {tuple(xt.shape)}")
    if t.ndim != 1 or t.shape[0] != xt.shape[0]:
        raise ValueError("t must be [B] and match xt")
    kappa = schedule.kappa(t).to(torch.float32).view(-1, 1, 1)
    uniform = torch.full((*xt.shape, 2), 0.5, device=xt.device)
    observed = F.one_hot(xt.to(torch.long), num_classes=2).to(torch.float32)
    probs = uniform * (1.0 - kappa) + observed * kappa
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


def teacher_posterior(base_probs: Tensor, log_h: Tensor) -> Tensor:
    if base_probs.shape != log_h.shape:
        raise ValueError("base_probs/log_h shape mismatch")
    log_weight = torch.log(base_probs.clamp_min(1.0e-12)) + log_h
    return torch.softmax(log_weight, dim=-1)


def sample_probs(probs: Tensor) -> Tensor:
    if probs.ndim != 3:
        raise ValueError(f"probs must be [B,N,K], got {tuple(probs.shape)}")
    bsz, n_nodes, num_states = probs.shape
    flat = probs.reshape(bsz * n_nodes, num_states)
    row_sum = flat.sum(dim=-1, keepdim=True)
    fallback = torch.zeros_like(flat)
    fallback[:, 0] = 1.0
    flat = torch.where(row_sum > 0.0, flat / row_sum.clamp_min(1.0e-12), fallback)
    return torch.multinomial(flat, num_samples=1).reshape(bsz, n_nodes)


__all__ = [
    "analytic_uniform_replace_posterior",
    "sample_probs",
    "teacher_posterior",
]
