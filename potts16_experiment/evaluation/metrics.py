"""Paper metrics for the released three-state Potts 16x16 evaluations."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch import Tensor

from edg_experiment.eval.histogram_js import histogram_js_1d, potts_energy_per_site_grid
from edg_experiment.lattice.potts import potts_energy, validate_potts_samples


def _distance_grid(L: int, distance_mode: str) -> List[int]:
    if distance_mode == "signed":
        return list(range(-L // 2, L // 2))
    if distance_mode == "abs":
        return list(range(0, L // 2 + 1))
    raise ValueError(f"Unknown distance_mode: {distance_mode}")


def _site_magnetization(samples: Tensor, *, L: int, q: int) -> Tensor:
    x = validate_potts_samples(samples, L=L, q=q).view(-1, L, L)
    if int(x.shape[0]) == 0:
        raise ValueError("Potts metrics require at least one sample")
    frequencies = torch.nn.functional.one_hot(x, num_classes=q).to(torch.float64).mean(dim=0)
    return ((float(q) * frequencies.max(dim=-1).values - 1.0) / float(q - 1)).cpu()


def absolute_potts_magnetization_error(
    samples: Tensor, reference_samples: Tensor, *, L: int, q: int
) -> float:
    """MDNS Eq. (29)--(30) row/column magnetization error."""

    sample = _site_magnetization(samples, L=L, q=q)
    reference = _site_magnetization(reference_samples, L=L, q=q)
    row_error = (sample.sum(dim=1) - reference.sum(dim=1)).abs().mean()
    col_error = (sample.sum(dim=0) - reference.sum(dim=0)).abs().mean()
    return float((0.5 * (row_error + col_error)).item())


def _corr_at_distance(samples: Tensor, *, L: int, q: int, r: int, direction: str) -> float:
    x = validate_potts_samples(samples, L=L, q=q).view(-1, L, L)
    if direction == "x":
        matches = 0.5 * (
            x.eq(torch.roll(x, shifts=-r, dims=1)).to(torch.float64)
            + x.eq(torch.roll(x, shifts=r, dims=1)).to(torch.float64)
        )
    elif direction == "y":
        matches = 0.5 * (
            x.eq(torch.roll(x, shifts=-r, dims=2)).to(torch.float64)
            + x.eq(torch.roll(x, shifts=r, dims=2)).to(torch.float64)
        )
    elif direction == "xy":
        matches = 0.25 * (
            x.eq(torch.roll(x, shifts=-r, dims=1)).to(torch.float64)
            + x.eq(torch.roll(x, shifts=r, dims=1)).to(torch.float64)
            + x.eq(torch.roll(x, shifts=-r, dims=2)).to(torch.float64)
            + x.eq(torch.roll(x, shifts=r, dims=2)).to(torch.float64)
        )
    else:
        raise ValueError(f"Unknown direction: {direction}")
    return float(matches.mean().item() - 1.0 / float(q))


def two_point_corr_curve(
    samples: Tensor,
    *,
    L: int,
    q: int,
    direction: str = "x",
    distance_mode: str = "signed",
) -> list[float]:
    return [
        _corr_at_distance(samples, L=L, q=q, r=r, direction=direction)
        for r in _distance_grid(L, distance_mode)
    ]


def _row_col_correlation_matrices(
    samples: Tensor, *, L: int, q: int
) -> tuple[Tensor, Tensor]:
    x = validate_potts_samples(samples, L=L, q=q).view(-1, L, L).to(torch.long)
    if int(x.shape[0]) == 0:
        raise ValueError("Potts correlation requires at least one sample")
    one_hot = torch.nn.functional.one_hot(x, num_classes=q).to(torch.float64)
    offset = float(L) / float(q)
    row = torch.einsum("nrcq,nscq->rs", one_hot, one_hot) / float(x.shape[0]) - offset
    col = torch.einsum("nrcq,nrdq->cd", one_hot, one_hot) / float(x.shape[0]) - offset
    return row.cpu(), col.cpu()


def two_point_corr_error_eq32(
    samples: Tensor, reference_samples: Tensor, *, L: int, q: int
) -> float:
    """Absolute Potts two-point correlation error from MDNS Eq. (32)."""

    sample_row, sample_col = _row_col_correlation_matrices(samples, L=L, q=q)
    reference_row, reference_col = _row_col_correlation_matrices(
        reference_samples, L=L, q=q
    )
    error = (sample_row - reference_row).abs().sum()
    error += (sample_col - reference_col).abs().sum()
    return float((error / float(L * L)).item())


def _occupancies(samples: Tensor, *, L: int, q: int) -> Tensor:
    x = validate_potts_samples(samples, L=L, q=q).view(-1, L * L).to(torch.long)
    if int(x.shape[0]) == 0:
        raise ValueError("Potts occupancy metrics require at least one sample")
    return torch.nn.functional.one_hot(x, num_classes=q).to(torch.float64).mean(dim=1).cpu()


def _dominant_mode_sorted_l1(
    samples: Tensor, reference_samples: Tensor, *, L: int, q: int
) -> float:
    def frequencies(values: Tensor) -> Tensor:
        dominant = _occupancies(values, L=L, q=q).argmax(dim=1)
        mass = torch.bincount(dominant, minlength=q).to(torch.float64)
        return (mass / float(dominant.numel())).sort(descending=True).values

    return float((frequencies(samples) - frequencies(reference_samples)).abs().sum().item())


def _potts_cv(samples: Tensor, *, L: int, q: int) -> Tensor:
    if q != 3:
        raise ValueError("The Potts CV metric is defined only for q=3")
    f0, f1, f2 = _occupancies(samples, L=L, q=q).unbind(dim=1)
    return torch.stack(
        (f0 - 0.5 * (f1 + f2), (np.sqrt(3.0) / 2.0) * (f1 - f2)),
        dim=1,
    )


def potts_cv_js(
    samples: Tensor,
    reference_samples: Tensor,
    *,
    L: int,
    q: int,
    bins: int = 50,
) -> float:
    """JS divergence on the fixed 50x50 MetaDNS Potts-CV grid."""

    sample_cv = _potts_cv(samples, L=L, q=q).numpy()
    reference_cv = _potts_cv(reference_samples, L=L, q=q).numpy()
    sample_hist, x_edges, y_edges = np.histogram2d(
        sample_cv[:, 0], sample_cv[:, 1], bins=(bins, bins), range=((-0.6, 1.1), (-1.0, 1.0))
    )
    reference_hist, _, _ = np.histogram2d(
        reference_cv[:, 0], reference_cv[:, 1], bins=(x_edges, y_edges)
    )
    p = sample_hist.reshape(-1).astype(np.float64)
    qmass = reference_hist.reshape(-1).astype(np.float64)
    p /= p.sum()
    qmass /= qmass.sum()
    midpoint = 0.5 * (p + qmass)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * (np.log(left[mask]) - np.log(right[mask]))))

    return 0.5 * (kl(p, midpoint) + kl(qmass, midpoint))


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
    samples: Tensor,
    reference_samples: Tensor,
    *,
    L: int,
    q: int,
    J: float = 1.0,
    direction: str = "x",
    distance_mode: str = "signed",
) -> Dict[str, float]:
    """Compute exactly the seven Potts metrics in the release contract."""

    generated = validate_potts_samples(samples, L=L, q=q).cpu()
    reference = validate_potts_samples(reference_samples, L=L, q=q).cpu()
    sample_curve = torch.tensor(
        two_point_corr_curve(generated, L=L, q=q, direction=direction, distance_mode=distance_mode),
        dtype=torch.float64,
    )
    reference_curve = torch.tensor(
        two_point_corr_curve(reference, L=L, q=q, direction=direction, distance_mode=distance_mode),
        dtype=torch.float64,
    )
    sample_energy = potts_energy(generated, L=L, q=q, J=J).to(torch.float64)
    reference_energy = potts_energy(reference, L=L, q=q, J=J).to(torch.float64)
    return {
        "dAbsM_abs": absolute_potts_magnetization_error(generated, reference, L=L, q=q),
        "two_point_corr_mae": two_point_corr_error_eq32(generated, reference, L=L, q=q),
        "two_point_corr_curve_mae": float((sample_curve - reference_curve).abs().mean().item()),
        "dominant_mode_sorted_l1_to_ref": _dominant_mode_sorted_l1(
            generated, reference, L=L, q=q
        ),
        "metadns_cv_js": potts_cv_js(generated, reference, L=L, q=q),
        "energy_per_site_js": histogram_js_1d(
            sample_energy / float(L * L),
            reference_energy / float(L * L),
            grid=potts_energy_per_site_grid(L=L, J=J),
        ),
        "EW2": _uniform_wasserstein_2(sample_energy, reference_energy),
    }


__all__ = [
    "absolute_potts_magnetization_error",
    "observable_metrics",
    "potts_cv_js",
    "two_point_corr_curve",
    "two_point_corr_error_eq32",
]
