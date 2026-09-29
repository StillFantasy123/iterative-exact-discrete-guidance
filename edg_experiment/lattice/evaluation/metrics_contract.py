"""Exact paper-metric contracts for released lattice evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


METRICS_SCHEMA_VERSION = "lattice_paper_metrics_v1"

_PAPER_KEYS = {
    "ising4": (
        "TV",
        "KL_emp_true",
        "Chi2_emp_true",
        "dE_abs",
        "dCnn_abs",
    ),
    "ising": (
        "dAbsM_abs",
        "two_point_corr_mae",
        "two_point_corr_curve_mae",
        "energy_per_site_js",
        "x_up_js",
        "EW2",
    ),
    "potts": (
        "dAbsM_abs",
        "two_point_corr_mae",
        "two_point_corr_curve_mae",
        "dominant_mode_sorted_l1_to_ref",
        "metadns_cv_js",
        "energy_per_site_js",
        "EW2",
    ),
}


def _problem_key(problem_kind: str, *, lattice_size: int | None = None) -> str:
    kind = str(problem_kind).lower()
    if kind == "ising" and int(lattice_size or 0) == 4:
        return "ising4"
    if kind not in {"ising", "potts"}:
        raise ValueError(f"Unsupported lattice problem: {problem_kind}")
    return kind


def paper_metric_names(
    problem_kind: str, *, lattice_size: int | None = None
) -> tuple[str, ...]:
    return _PAPER_KEYS[_problem_key(problem_kind, lattice_size=lattice_size)]


@dataclass(frozen=True)
class MetricGroups:
    paper: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": METRICS_SCHEMA_VERSION,
            "paper": self.paper,
        }


def partition_metrics(
    metrics: Mapping[str, Any], *, problem_kind: str, lattice_size: int | None = None
) -> MetricGroups:
    names = paper_metric_names(problem_kind, lattice_size=lattice_size)
    missing = sorted(set(names) - set(metrics))
    if missing:
        raise ValueError("Missing paper metric(s): " + ", ".join(missing))
    return MetricGroups(paper={name: metrics[name] for name in names})


def is_metric_groups(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    paper = payload.get("paper")
    return bool(
        payload.get("schema_version") == METRICS_SCHEMA_VERSION
        and isinstance(paper, Mapping)
        and any(set(paper) == set(names) for names in _PAPER_KEYS.values())
    )


__all__ = [
    "METRICS_SCHEMA_VERSION",
    "MetricGroups",
    "is_metric_groups",
    "paper_metric_names",
    "partition_metrics",
]
