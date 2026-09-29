"""Stable paper-metric output for lattice evaluation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from edg_experiment.lattice.evaluation.metrics_contract import is_metric_groups
from edg_experiment.utils.io import ensure_dir, save_json


def write_paper_metrics(
    *, run_dir: str | Path, guided: Mapping[str, Any], context: Mapping[str, Any]
) -> str:
    if not is_metric_groups(guided):
        raise ValueError("guided metrics must follow lattice_paper_metrics_v1")
    path = ensure_dir(Path(run_dir) / "metrics") / "paper_metrics.json"
    save_json(
        path,
        {
            "schema_version": "lattice_paper_metrics_v1",
            "context": copy.deepcopy(dict(context)),
            "metrics": copy.deepcopy(dict(guided["paper"])),
        },
    )
    return str(path)


__all__ = ["write_paper_metrics"]
