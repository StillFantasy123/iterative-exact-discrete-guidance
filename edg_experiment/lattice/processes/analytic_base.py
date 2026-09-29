"""Analytic uniform-replacement posterior shared by lattice domains."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from edg_experiment.dfm.schedules import KappaSchedule
from edg_experiment.dfm.uniform_replace import analytic_uniform_replace_posterior as _analytic_uniform_replace_posterior


def analytic_uniform_replace_posterior(
    xt: Tensor,
    t: Tensor,
    schedule: KappaSchedule,
    num_states: int = 2,
) -> Tensor:
    return _analytic_uniform_replace_posterior(
        xt=xt,
        t=t,
        schedule=schedule,
        num_states=int(num_states),
    )


class AnalyticUniformReplacePosteriorModel(nn.Module):
    def __init__(self, schedule: KappaSchedule, num_states: int = 2) -> None:
        super().__init__()
        self.schedule = schedule
        self.num_states = int(num_states)

    def forward_probs(self, xt: Tensor, t: Tensor) -> Tensor:
        return analytic_uniform_replace_posterior(
            xt=xt,
            t=t,
            schedule=self.schedule,
            num_states=self.num_states,
        )

    def forward(self, xt: Tensor, t: Tensor) -> Tensor:
        probs = self.forward_probs(xt=xt, t=t)
        return torch.log(probs.clamp_min(1.0e-12))


def base_posterior(theta_model, xt: Tensor, t: Tensor) -> Tensor:
    if not isinstance(theta_model, AnalyticUniformReplacePosteriorModel):
        raise TypeError("release evaluation requires the analytic uniform posterior")
    probs = theta_model.forward_probs(xt, t)
    if probs.ndim != 3:
        raise ValueError("base_posterior must return [B,D,S] probabilities")
    return probs


def teacher_posterior(
    theta_model,
    guidance_model,
    xt: Tensor,
    t: Tensor,
    guidance_strength: float = 1.0,
) -> Tensor:
    p = base_posterior(theta_model=theta_model, xt=xt, t=t)
    if guidance_model is None or not hasattr(guidance_model, "forward_with_log_h"):
        raise TypeError("release evaluation requires a guidance model with log-h output")
    _, log_h = guidance_model.forward_with_log_h(xt, t)

    log_weight = torch.log(p.clamp_min(1.0e-12)) + float(guidance_strength) * log_h
    return torch.softmax(log_weight, dim=-1)


__all__ = [
    "analytic_uniform_replace_posterior",
    "AnalyticUniformReplacePosteriorModel",
    "base_posterior",
    "teacher_posterior",
]
