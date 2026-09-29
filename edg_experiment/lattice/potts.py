from __future__ import annotations

import torch
from torch import Tensor


def validate_potts_samples(samples: Tensor, *, L: int, q: int) -> Tensor:
    if int(L) <= 0:
        raise ValueError("L must be positive")
    if int(q) < 2:
        raise ValueError("q must be at least 2")
    if samples.ndim == 1:
        samples = samples.unsqueeze(0)
    if samples.ndim != 2 or int(samples.shape[1]) != int(L) * int(L):
        raise ValueError(f"samples must be [B,{int(L) * int(L)}], got {tuple(samples.shape)}")
    samples = samples.to(torch.long)
    if bool(((samples < 0) | (samples >= int(q))).any()):
        raise ValueError(f"Potts samples must contain labels in [0,{int(q) - 1}]")
    return samples


def potts_energy(samples: Tensor, *, L: int, q: int = 3, J: float = 1.0) -> Tensor:
    """Periodic q-state Potts Hamiltonian with each undirected edge counted once.

    H(x) = -J * sum_{<i,j>} 1[x_i == x_j].
    """
    x = validate_potts_samples(samples, L=int(L), q=int(q)).view(-1, int(L), int(L))
    same_down = x.eq(torch.roll(x, shifts=-1, dims=1))
    same_right = x.eq(torch.roll(x, shifts=-1, dims=2))
    matching_edges = same_down.sum(dim=(1, 2)) + same_right.sum(dim=(1, 2))
    return -float(J) * matching_edges.to(torch.float32)


__all__ = ["potts_energy", "validate_potts_samples"]
