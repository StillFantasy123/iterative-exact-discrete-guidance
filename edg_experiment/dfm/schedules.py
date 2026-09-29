from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class KappaSchedule:
    kind: str = "cosine"
    eps: float = 1.0e-4

    def _clamp_t(self, t: Tensor) -> Tensor:
        return torch.clamp(t, min=0.0, max=1.0 - self.eps)

    def kappa(self, t: Tensor) -> Tensor:
        tc = self._clamp_t(t)
        if self.kind == "linear":
            return tc
        if self.kind == "cosine":
            # kappa(t)=sin^2(pi t / 2)
            return torch.sin(0.5 * math.pi * tc) ** 2
        raise ValueError(f"Unknown schedule kind: {self.kind}")

    def kappa_dot(self, t: Tensor) -> Tensor:
        tc = self._clamp_t(t)
        if self.kind == "linear":
            return torch.ones_like(tc)
        if self.kind == "cosine":
            # d/dt sin^2(pi t/2) = (pi/2) sin(pi t)
            return 0.5 * math.pi * torch.sin(math.pi * tc)
        raise ValueError(f"Unknown schedule kind: {self.kind}")

    def rate_coeff(self, t: Tensor) -> Tensor:
        k = self.kappa(t)
        kd = self.kappa_dot(t)
        denom = (1.0 - k).clamp_min(self.eps)
        return kd / denom


__all__ = ["KappaSchedule"]
