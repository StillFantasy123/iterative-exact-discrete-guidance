"""Fixed-grid histogram Jensen--Shannon metrics for paper evaluation.

The paper protocol must never infer histogram edges from generated samples.
This module provides small, auditable grid objects and validates that every
sample falls inside the registered support before computing a JS divergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


ArrayLike = np.ndarray | torch.Tensor | Iterable[float]


@dataclass(frozen=True)
class HistogramGrid1D:
    protocol_id: str
    edges: tuple[float, ...]

    def as_array(self) -> np.ndarray:
        edges = np.asarray(self.edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError(f"Invalid histogram grid {self.protocol_id!r}")
        return edges

def _linspace_grid(protocol_id: str, lower: float, upper: float, bins: int) -> HistogramGrid1D:
    if int(bins) <= 0 or not float(upper) > float(lower):
        raise ValueError("Histogram grids require positive bins and upper > lower")
    edges = np.linspace(float(lower), float(upper), int(bins) + 1, dtype=np.float64)
    return HistogramGrid1D(protocol_id=protocol_id, edges=tuple(float(v) for v in edges))


def ising_energy_per_site_grid(*, L: int, J: float, h: float) -> HistogramGrid1D:
    """Analytic square-lattice energy support with ``L**2`` fixed bins."""
    radius = 2.0 * abs(float(J)) + abs(float(h))
    return _linspace_grid(
        f"lattice-hist-v1:ising-energy:L{int(L)}:J{float(J):g}:h{float(h):g}",
        -radius,
        radius,
        int(L) * int(L),
    )


def ising_x_up_grid(*, L: int) -> HistogramGrid1D:
    return _linspace_grid(
        f"lattice-hist-v1:ising-x-up:L{int(L)}",
        0.0,
        1.0,
        int(L) * int(L),
    )


def potts_energy_per_site_grid(*, L: int, J: float) -> HistogramGrid1D:
    lower, upper = sorted((-2.0 * float(J), 0.0))
    return _linspace_grid(
        f"lattice-hist-v1:potts-energy:L{int(L)}:J{float(J):g}",
        lower,
        upper,
        int(L) * int(L),
    )


def _as_finite_array(values: ArrayLike, *, name: str) -> np.ndarray:
    if torch.is_tensor(values):
        array = values.detach().to(device="cpu", dtype=torch.float64).numpy()
    else:
        array = np.asarray(values, dtype=np.float64)
    array = array.reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_support(values: np.ndarray, edges: np.ndarray, *, name: str) -> None:
    tolerance = 1.0e-12 * max(1.0, abs(float(edges[0])), abs(float(edges[-1])))
    below = int(np.count_nonzero(values < float(edges[0]) - tolerance))
    above = int(np.count_nonzero(values > float(edges[-1]) + tolerance))
    if below or above:
        raise ValueError(
            f"{name} falls outside registered support [{edges[0]}, {edges[-1]}]: "
            f"underflow={below}, overflow={above}"
        )


def _js_from_masses(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if p.shape != q.shape or p.sum() <= 0.0 or q.sum() <= 0.0:
        raise ValueError("Histogram masses must have equal shape and positive total mass")
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)

    def _kl(a: np.ndarray) -> float:
        mask = a > 0.0
        return float(np.sum(a[mask] * (np.log(a[mask]) - np.log(midpoint[mask]))))

    return 0.5 * (_kl(p) + _kl(q))


def histogram_js_1d(values: ArrayLike, reference: ArrayLike, *, grid: HistogramGrid1D) -> float:
    x = _as_finite_array(values, name="values")
    y = _as_finite_array(reference, name="reference")
    edges = grid.as_array()
    _validate_support(x, edges, name="values")
    _validate_support(y, edges, name="reference")
    x_hist, _ = np.histogram(x, bins=edges)
    y_hist, _ = np.histogram(y, bins=edges)
    if int(x_hist.sum()) != int(x.size) or int(y_hist.sum()) != int(y.size):
        raise RuntimeError(f"Histogram grid {grid.protocol_id!r} silently dropped samples")
    return _js_from_masses(x_hist, y_hist)


__all__ = [
    "HistogramGrid1D",
    "histogram_js_1d",
    "ising_energy_per_site_grid",
    "ising_x_up_grid",
    "potts_energy_per_site_grid",
]
