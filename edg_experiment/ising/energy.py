from __future__ import annotations

import torch
from torch import Tensor


def binary_to_spin(x_binary: Tensor) -> Tensor:
    return (2 * x_binary.to(torch.int64) - 1).to(torch.int8)


def ising_energy_binary(x_binary: Tensor, L: int, J: float = 1.0, h: float = 0.0) -> Tensor:
    """Ising Hamiltonian for {0,1} encoded states. Returns [B]."""
    if x_binary.ndim == 1:
        x_binary = x_binary.unsqueeze(0)
    s = binary_to_spin(x_binary).to(torch.float32).view(-1, L, L)
    sx = torch.roll(s, shifts=-1, dims=1)
    sy = torch.roll(s, shifts=-1, dims=2)
    interaction = -J * (s * (sx + sy)).sum(dim=(1, 2))
    field = -h * s.sum(dim=(1, 2))
    return interaction + field


__all__ = [
    "binary_to_spin",
    "ising_energy_binary",
]
