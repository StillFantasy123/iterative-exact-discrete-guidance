"""Paper metrics for the released Ising 16x16 evaluations."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch import Tensor

from edg_experiment.eval.histogram_js import (
    histogram_js_1d,
    ising_energy_per_site_grid,
    ising_x_up_grid,
)
from edg_experiment.ising.energy import binary_to_spin, ising_energy_binary


def _validate(samples: Tensor, *, L: int) -> Tensor:
    if samples.ndim != 2 or int(samples.shape[1]) != int(L * L):
        raise ValueError(f"samples must have shape [N,{L * L}]")
    values = samples.detach().to(device="cpu", dtype=torch.uint8)
    if int(values.shape[0]) == 0 or int(values.min()) < 0 or int(values.max()) > 1:
        raise ValueError("Ising samples must be a non-empty binary tensor")
    return values


def _magnetization_profiles(samples: Tensor, *, L: int) -> tuple[Tensor, Tensor]:
    spins = binary_to_spin(_validate(samples, L=L).to(torch.int64)).to(torch.float64)
    spins = spins.view(-1, L, L)
    return spins.mean(dim=1).mean(dim=0), spins.mean(dim=2).mean(dim=0)


def absolute_magnetization_error(
    samples_binary: Tensor, reference_binary: Tensor, *, L: int
) -> float:
    sample_row, sample_col = _magnetization_profiles(samples_binary, L=L)
    reference_row, reference_col = _magnetization_profiles(reference_binary, L=L)
    return float(
        0.5
        * (
            (sample_row - reference_row).abs().mean()
            + (sample_col - reference_col).abs().mean()
        ).item()
    )


def _distance_grid(L: int, distance_mode: str) -> List[int]:
    if distance_mode == "signed":
        return list(range(-L // 2, L // 2))
    if distance_mode == "abs":
        return list(range(0, L // 2 + 1))
    raise ValueError(f"Unknown distance_mode: {distance_mode}")


def _corr_at_distance(spins: Tensor, *, L: int, r: int, direction: str) -> float:
    values = spins.to(torch.float32).view(-1, L, L)
    if direction == "x":
        neighbor = 0.5 * (
            torch.roll(values, shifts=-r, dims=1)
            + torch.roll(values, shifts=r, dims=1)
        )
    elif direction == "y":
        neighbor = 0.5 * (
            torch.roll(values, shifts=-r, dims=2)
            + torch.roll(values, shifts=r, dims=2)
        )
    elif direction == "xy":
        neighbor = 0.25 * (
            torch.roll(values, shifts=-r, dims=1)
            + torch.roll(values, shifts=r, dims=1)
            + torch.roll(values, shifts=-r, dims=2)
            + torch.roll(values, shifts=r, dims=2)
        )
    else:
        raise ValueError(f"Unknown direction: {direction}")
    return float((values * neighbor).mean().item())


def two_point_corr_curve(
    samples_binary: Tensor,
    *,
    L: int,
    direction: str = "x",
    distance_mode: str = "signed",
) -> list[float]:
    spins = binary_to_spin(_validate(samples_binary, L=L).to(torch.int64))
    return [
        _corr_at_distance(spins, L=L, r=r, direction=direction)
        for r in _distance_grid(L, distance_mode)
    ]


def _row_col_covariance_matrices(samples_binary: Tensor, *, L: int) -> tuple[Tensor, Tensor]:
    spins = binary_to_spin(_validate(samples_binary, L=L).to(torch.int64)).to(torch.float64)
    spins = spins.view(-1, L, L)
    site_mean = spins.mean(dim=0)
    row = torch.einsum("brc,bsc->rs", spins, spins) / float(spins.shape[0])
    col = torch.einsum("brc,brd->cd", spins, spins) / float(spins.shape[0])
    return row - site_mean @ site_mean.t(), col - site_mean.t() @ site_mean


def two_point_corr_error_eq28(
    samples_binary: Tensor, reference_binary: Tensor, *, L: int
) -> float:
    """MDNS Eq. (28)-style row/column connected-correlation error."""

    sample_row, sample_col = _row_col_covariance_matrices(samples_binary, L=L)
    reference_row, reference_col = _row_col_covariance_matrices(reference_binary, L=L)
    error = (sample_row - reference_row).abs().sum()
    error += (sample_col - reference_col).abs().sum()
    return float((error / float(L * L)).item())


def _uniform_wasserstein_2(values: Tensor, reference: Tensor) -> float:
    x = np.sort(values.detach().to(torch.float64).cpu().numpy().reshape(-1))
    y = np.sort(reference.detach().to(torch.float64).cpu().numpy().reshape(-1))
    if x.size == 0 or y.size == 0:
        raise ValueError("Wasserstein inputs must be non-empty")
    i = j = 0
    previous = squared_cost = 0.0
    while i < x.size and j < y.size:
        next_x = float(i + 1) / float(x.size)
        next_y = float(j + 1) / float(y.size)
        boundary = min(next_x, next_y)
        mass = boundary - previous
        if mass > 0.0:
            squared_cost += mass * (float(x[i]) - float(y[j])) ** 2
        previous = boundary
        if next_x <= boundary + 1.0e-15:
            i += 1
        if next_y <= boundary + 1.0e-15:
            j += 1
    return float(np.sqrt(max(squared_cost, 0.0)))


def observable_metrics(
    samples_binary: Tensor,
    reference_binary: Tensor,
    *,
    L: int,
    J: float = 1.0,
    h: float = 0.0,
    direction: str = "x",
    distance_mode: str = "signed",
) -> Dict[str, float]:
    """Compute exactly the six Ising metrics in the release contract."""

    samples = _validate(samples_binary, L=L)
    reference = _validate(reference_binary, L=L)
    sample_curve = torch.tensor(
        two_point_corr_curve(samples, L=L, direction=direction, distance_mode=distance_mode),
        dtype=torch.float64,
    )
    reference_curve = torch.tensor(
        two_point_corr_curve(reference, L=L, direction=direction, distance_mode=distance_mode),
        dtype=torch.float64,
    )
    sample_energy = ising_energy_binary(samples, L=L, J=J, h=h).to(torch.float64)
    reference_energy = ising_energy_binary(reference, L=L, J=J, h=h).to(torch.float64)
    sample_x_up = samples.to(torch.float64).mean(dim=1)
    reference_x_up = reference.to(torch.float64).mean(dim=1)
    return {
        "dAbsM_abs": absolute_magnetization_error(samples, reference, L=L),
        "two_point_corr_mae": two_point_corr_error_eq28(samples, reference, L=L),
        "two_point_corr_curve_mae": float((sample_curve - reference_curve).abs().mean().item()),
        "energy_per_site_js": histogram_js_1d(
            sample_energy / float(L * L),
            reference_energy / float(L * L),
            grid=ising_energy_per_site_grid(L=L, J=J, h=h),
        ),
        "x_up_js": histogram_js_1d(
            sample_x_up,
            reference_x_up,
            grid=ising_x_up_grid(L=L),
        ),
        "EW2": _uniform_wasserstein_2(sample_energy, reference_energy),
    }


__all__ = [
    "absolute_magnetization_error",
    "observable_metrics",
    "two_point_corr_curve",
    "two_point_corr_error_eq28",
]
