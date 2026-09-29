"""Strict configuration resolver for released Max-Cut checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


EVALUATION_SCHEMA_VERSION = "maxcut-evaluation-v1"
EVALUATION_METHODS = ("one_shot", "iedg")
EVALUATION_SCALES = ("ba20", "ba40", "ba100")


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class MaxCutEvaluationSpec:
    method: str
    scale: str

    @classmethod
    def parse(cls, raw: Any) -> "MaxCutEvaluationSpec":
        value = _mapping(raw, name="evaluation")
        allowed = {"schema", "method", "scale"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"Unknown evaluation field(s): {', '.join(unknown)}")
        schema = _string(value.get("schema"), name="evaluation.schema")
        if schema != EVALUATION_SCHEMA_VERSION:
            raise ValueError(
                f"evaluation.schema must be {EVALUATION_SCHEMA_VERSION!r}"
            )
        method = _string(value.get("method"), name="evaluation.method")
        if method not in EVALUATION_METHODS:
            raise ValueError(f"evaluation.method must be one of {EVALUATION_METHODS}")
        scale = _string(value.get("scale"), name="evaluation.scale")
        if scale not in EVALUATION_SCALES:
            raise ValueError(f"evaluation.scale must be one of {EVALUATION_SCALES}")
        return cls(method=method, scale=scale)


def resolve_evaluation_config(
    raw: Mapping[str, Any], *, config_path: str | Path
) -> dict[str, Any]:
    top = _mapping(raw, name="config")
    if set(top) != {"evaluation"}:
        raise ValueError("Max-Cut configs must contain exactly one evaluation field")
    spec = MaxCutEvaluationSpec.parse(top["evaluation"])
    return {
        "evaluation": {
            "schema": EVALUATION_SCHEMA_VERSION,
            "method": spec.method,
            "scale": spec.scale,
        },
        "device": "cuda",
        "experiment": {"seed": 42},
        "model": {
            "hidden_dim": 128,
            "num_layers": 3,
            "num_heads": 4,
            "time_embed_dim": 32,
            "problem_embed_dim": 8,
            "log_h_clip_min": -30.0,
            "log_h_clip_max": 30.0,
        },
        "rollout": {"num_steps": 128, "batch_size": 256},
        "eval": {"progress_bar": False},
        "_config_path": str(Path(config_path).expanduser().resolve()),
    }


__all__ = [
    "EVALUATION_METHODS",
    "EVALUATION_SCALES",
    "EVALUATION_SCHEMA_VERSION",
    "MaxCutEvaluationSpec",
    "resolve_evaluation_config",
]
