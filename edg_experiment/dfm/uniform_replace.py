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
    num_states: int,
) -> Tensor:
    if xt.ndim != 2:
        raise ValueError(f"xt must be [B,D], got {tuple(xt.shape)}")
    if t.ndim == 0:
        t = t[None]
    if t.ndim != 1 or t.shape[0] != xt.shape[0]:
        raise ValueError("t must be [B] and match xt")
    if int(num_states) < 2:
        raise ValueError("num_states must be at least 2")
    if bool(((xt < 0) | (xt >= int(num_states))).any()):
        raise ValueError("xt contains labels outside the terminal vocabulary")
    kappa = schedule.kappa(t).to(torch.float32).view(-1, 1, 1)
    uniform = torch.full(
        (*xt.shape, int(num_states)),
        1.0 / float(num_states),
        device=xt.device,
        dtype=torch.float32,
    )
    observed = F.one_hot(xt.to(torch.long), num_classes=int(num_states)).to(torch.float32)
    probs = uniform * (1.0 - kappa) + observed * kappa
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


__all__ = ["analytic_uniform_replace_posterior"]
