"""Resolve manifest-defined lattice evaluation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from edg_experiment.artifacts import load_manifest_result
from edg_experiment.config.load import load_config
from edg_experiment.lattice.problem import build_problem


@dataclass(frozen=True)
class ReleaseResult:
    result_id: str
    checkpoint_path: Path
    config_path: Path
    checkpoint_step: int
    weight_source: str
    problem: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    evaluation_protocol: Mapping[str, Any]
    reference_samples_path: Path


def load_release_result(
    manifest_path: str | Path,
    result_id: str,
    *,
    require_checkpoint: bool = True,
) -> ReleaseResult:
    """Resolve one raw EMA artifact together with its manifest-owned config."""

    artifact = load_manifest_result(
        manifest_path,
        result_id,
        require_checkpoint=require_checkpoint,
        expected_evaluator="lattice",
    )
    if artifact.checkpoint_format != "raw_pytorch_state_dict":
        raise ValueError("Release lineage requires raw_pytorch_state_dict artifacts")
    if artifact.weight_source != "ema":
        raise ValueError("Release lineage currently requires weights=ema")
    evaluation = artifact.evaluation
    reference_name = Path(str(evaluation.get("reference_samples", "")))
    if len(reference_name.parts) != 1 or not reference_name.name:
        raise ValueError(
            f"results.{result_id}.evaluation.reference_samples must be a flat filename"
        )
    reference_samples_path = (artifact.checkpoint_path.parent / reference_name).resolve()

    cfg = load_config(artifact.config_path)
    config_problem = build_problem(cfg)
    declared = artifact.problem
    expected_states = int(declared.get("num_states", 2))
    checks = {
        "kind": (config_problem.kind, str(declared.get("kind", ""))),
        "lattice_size": (config_problem.L, int(declared.get("lattice_size", 0))),
        "num_states": (config_problem.num_states, expected_states),
    }
    for field, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                f"results.{result_id}.problem.{field} does not match its config: "
                f"{actual!r} != {expected!r}"
            )
    if abs(config_problem.beta - float(declared.get("beta"))) > 1.0e-10:
        raise ValueError(f"results.{result_id}.problem.beta does not match its config")

    return ReleaseResult(
        result_id=str(result_id),
        checkpoint_path=artifact.checkpoint_path,
        config_path=artifact.config_path,
        checkpoint_step=artifact.checkpoint_step,
        weight_source=artifact.weight_source,
        problem=dict(declared),
        evaluation=dict(evaluation),
        evaluation_protocol=dict(artifact.evaluation_protocol),
        reference_samples_path=reference_samples_path,
    )


__all__ = ["ReleaseResult", "load_release_result"]
