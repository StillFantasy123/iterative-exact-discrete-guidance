"""Paper metrics for exactly enumerated Ising 4x4 targets."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor

from edg_experiment.ising.energy import (
    binary_to_spin,
    ising_energy_binary,
)
from ising4_experiment.evaluation.exact_target import (
    empirical_pmf_binary,
    exact_ising_target,
)


def exact_terminal_metrics(
    samples_binary: Tensor,
    *,
    beta: float,
    L: int = 4,
    J: float = 1.0,
    h: float = 0.0,
    smoothing_eps: float = 1.0e-12,
) -> Dict[str, Any]:
    """Return only the distribution and observable errors reported in the paper."""

    if smoothing_eps <= 0.0:
        raise ValueError("smoothing_eps must be positive")
    samples = samples_binary.detach().to(device="cpu", dtype=torch.long)
    target = exact_ising_target(L=L, beta=float(beta), J=float(J), h=float(h))
    empirical = empirical_pmf_binary(samples, length=L * L)
    smoothed = empirical + float(smoothing_eps)
    smoothed = smoothed / smoothed.sum()
    exact = target.pmf

    tv = 0.5 * torch.abs(exact - empirical).sum()
    kl_emp_true = torch.sum(
        torch.where(
            empirical > 0.0,
            empirical
            * (torch.log(empirical.clamp_min(1.0e-300)) - torch.log(exact)),
            0.0,
        )
    )
    chi2 = torch.sum((empirical - exact).square() / exact.clamp_min(1.0e-300))

    sample_energy = ising_energy_binary(
        samples, L=L, J=float(J), h=float(h)
    ).to(torch.float64)
    exact_energy = target.expectation(target.energy)
    sample_spins = binary_to_spin(samples).to(torch.float64).view(-1, L, L)
    sample_cnn = 0.5 * (
        (sample_spins * torch.roll(sample_spins, shifts=-1, dims=1)).mean(dim=(1, 2))
        + (sample_spins * torch.roll(sample_spins, shifts=-1, dims=2)).mean(dim=(1, 2))
    )
    exact_cnn_per_state = 0.5 * (
        (target.spins * torch.roll(target.spins, shifts=-1, dims=1)).mean(dim=(1, 2))
        + (target.spins * torch.roll(target.spins, shifts=-1, dims=2)).mean(dim=(1, 2))
    )
    exact_cnn = target.expectation(exact_cnn_per_state)
    return {
        "TV": float(tv.item()),
        "KL_emp_true": float(kl_emp_true.item()),
        "Chi2_emp_true": float(chi2.item()),
        "dE_abs": abs(float(sample_energy.mean().item()) - float(exact_energy.item())),
        "dCnn_abs": abs(float(sample_cnn.mean().item()) - float(exact_cnn.item())),
    }


__all__ = ["exact_terminal_metrics"]
