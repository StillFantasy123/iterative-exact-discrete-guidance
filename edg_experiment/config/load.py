from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    evaluation = cfg.get("evaluation")
    schema = evaluation.get("schema") if isinstance(evaluation, dict) else None
    if schema == "maxcut-evaluation-v1":
        from CO_experiment.paper_config import resolve_evaluation_config

        return resolve_evaluation_config(cfg, config_path=path)
    if schema == "lattice-evaluation-v1":
        from edg_experiment.lattice.paper_config import resolve_evaluation_config

        return resolve_evaluation_config(cfg, config_path=path)

    raise ValueError(
        f"{path} is not a supported release evaluation config; "
        "expected lattice-evaluation-v1 or maxcut-evaluation-v1"
    )


__all__ = ["load_config"]
