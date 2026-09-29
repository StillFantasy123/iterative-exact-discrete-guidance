from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import Tensor

from edg_experiment.ising.energy import binary_to_spin, ising_energy_binary


def all_binary_configs(length: int, *, device: torch.device | str = "cpu") -> Tensor:
    if length <= 0 or length > 20:
        raise ValueError("exact binary enumeration requires 1 <= length <= 20")
    device = torch.device(device)
    shifts = torch.arange(length - 1, -1, -1, device=device, dtype=torch.int64)
    ids = torch.arange(2**length, device=device, dtype=torch.int64)
    return ((ids[:, None] >> shifts) & 1).to(torch.uint8)


def empirical_pmf_binary(samples: Tensor, *, length: int) -> Tensor:
    if samples.ndim != 2 or int(samples.shape[1]) != int(length):
        raise ValueError(f"samples must be [B,{length}]")
    samples = samples.to(device="cpu", dtype=torch.int64)
    if bool(((samples < 0) | (samples > 1)).any()):
        raise ValueError("samples must be binary")
    shifts = torch.arange(length - 1, -1, -1, dtype=torch.int64)
    ids = (samples << shifts).sum(dim=1)
    counts = torch.bincount(ids, minlength=2**length).to(torch.float64)
    if int(counts.sum().item()) <= 0:
        raise ValueError("samples must be non-empty")
    return counts / counts.sum()


@dataclass(frozen=True)
class ExactIsingTarget:
    L: int
    beta: float
    J: float
    h: float
    states: Tensor
    energy: Tensor
    log_z: float
    pmf: Tensor
    spins: Tensor

    @property
    def length(self) -> int:
        return int(self.L * self.L)

    def expectation(self, values: Tensor) -> Tensor:
        values = values.to(device="cpu", dtype=torch.float64)
        if int(values.shape[0]) != int(self.pmf.shape[0]):
            raise ValueError("values must have one leading entry per exact state")
        return torch.tensordot(self.pmf, values, dims=([0], [0]))

    def sample(self, num_samples: int, *, seed: int) -> Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        ids = torch.multinomial(self.pmf, num_samples=int(num_samples), replacement=True, generator=generator)
        return self.states.index_select(0, ids).clone()


@lru_cache(maxsize=32)
def exact_ising_target(
    *,
    L: int = 4,
    beta: float,
    J: float = 1.0,
    h: float = 0.0,
) -> ExactIsingTarget:
    L = int(L)
    if L <= 0 or L > 4:
        raise ValueError("exact_ising_target supports 1 <= L <= 4")
    states = all_binary_configs(L * L)
    energy = ising_energy_binary(states.to(torch.long), L=L, J=float(J), h=float(h)).to(torch.float64)
    log_weight = -float(beta) * energy
    log_z_tensor = torch.logsumexp(log_weight, dim=0)
    pmf = torch.exp(log_weight - log_z_tensor)
    spins = binary_to_spin(states.to(torch.long)).to(torch.float64).view(-1, L, L)
    return ExactIsingTarget(
        L=L,
        beta=float(beta),
        J=float(J),
        h=float(h),
        states=states,
        energy=energy,
        log_z=float(log_z_tensor.item()),
        pmf=pmf,
        spins=spins,
    )


__all__ = [
    "ExactIsingTarget",
    "all_binary_configs",
    "empirical_pmf_binary",
    "exact_ising_target",
]
