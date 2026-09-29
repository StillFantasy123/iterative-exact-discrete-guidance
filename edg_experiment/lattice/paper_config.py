"""Strict configuration resolver for released lattice checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


EVALUATION_SCHEMA_VERSION = "lattice-evaluation-v1"
EVALUATION_PRESETS = ("ising4", "ising16", "potts16")
EVALUATION_METHODS = ("iedg", "one_shot")


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _strict_keys(value: Mapping[str, Any], *, allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} field(s): {', '.join(unknown)}")


def _string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


@dataclass(frozen=True)
class PreconditionerSpec:
    enabled: bool
    ratio_scale: float
    base_beta: float | None

    @classmethod
    def parse(cls, raw: Any, *, preset: str, target_beta: float) -> "PreconditionerSpec":
        value = _mapping(raw, name="evaluation.preconditioner")
        _strict_keys(
            value,
            allowed={"enabled", "ratio_scale", "base_beta"},
            name="evaluation.preconditioner",
        )
        enabled = value.get("enabled", preset != "ising4")
        if not isinstance(enabled, bool):
            raise ValueError("evaluation.preconditioner.enabled must be a boolean")
        ratio_scale = _number(
            value.get("ratio_scale", 1.0),
            name="evaluation.preconditioner.ratio_scale",
        )
        if ratio_scale <= 0.0:
            raise ValueError("evaluation.preconditioner.ratio_scale must be positive")
        base_beta_raw = value.get("base_beta", target_beta if enabled else None)
        base_beta = (
            None
            if base_beta_raw is None
            else _number(base_beta_raw, name="evaluation.preconditioner.base_beta")
        )
        if preset == "ising4" and enabled:
            raise ValueError("the released Ising4 checkpoints do not use preconditioning")
        if preset != "ising4" and not enabled:
            raise ValueError("the released 16x16 checkpoints require preconditioning")
        return cls(enabled=enabled, ratio_scale=ratio_scale, base_beta=base_beta)


@dataclass(frozen=True)
class LatticeEvaluationSpec:
    preset: str
    method: str
    target_beta: float
    preconditioner: PreconditionerSpec

    @classmethod
    def parse(cls, raw: Any) -> "LatticeEvaluationSpec":
        value = _mapping(raw, name="evaluation")
        _strict_keys(
            value,
            allowed={"schema", "preset", "method", "target_beta", "preconditioner"},
            name="evaluation",
        )
        schema = _string(value.get("schema"), name="evaluation.schema")
        if schema != EVALUATION_SCHEMA_VERSION:
            raise ValueError(
                f"evaluation.schema must be {EVALUATION_SCHEMA_VERSION!r}"
            )
        preset = _string(value.get("preset"), name="evaluation.preset")
        if preset not in EVALUATION_PRESETS:
            raise ValueError(f"evaluation.preset must be one of {EVALUATION_PRESETS}")
        method = _string(value.get("method"), name="evaluation.method")
        if method not in EVALUATION_METHODS:
            raise ValueError(f"evaluation.method must be one of {EVALUATION_METHODS}")
        target_beta = _number(value.get("target_beta"), name="evaluation.target_beta")
        if target_beta <= 0.0:
            raise ValueError("evaluation.target_beta must be positive")
        return cls(
            preset=preset,
            method=method,
            target_beta=target_beta,
            preconditioner=PreconditionerSpec.parse(
                value.get("preconditioner", {}),
                preset=preset,
                target_beta=target_beta,
            ),
        )


def _resolved_model(spec: LatticeEvaluationSpec) -> dict[str, Any]:
    is_ising4 = spec.preset == "ising4"
    is_potts = spec.preset == "potts16"
    kind = "potts" if is_potts else "ising"
    model: dict[str, Any] = {
        "guidance_parametrization": "log_h",
        "log_h_clip_min": -30.0 if is_ising4 else -50.0,
        "log_h_clip_max": 30.0 if is_ising4 else 50.0,
        "center_log_h_over_states": False,
        "preconditioning": {
            "enabled": spec.preconditioner.enabled,
            "kind": f"{kind}_bp_4nbr",
            "ratio_scale": spec.preconditioner.ratio_scale,
            "log_prob_eps": 1.0e-12,
            "boundary": "periodic",
            "residual_zero_init": True,
            "num_message_passing_steps": 2,
            "message_damping": 1.0,
        },
    }
    if spec.preconditioner.base_beta is not None:
        model["preconditioning"]["base_beta"] = spec.preconditioner.base_beta
    if is_ising4:
        model["presets"] = {}
        model["guidance_backbone"] = {
            "type": "tiny_mlp",
            "variant": "deep",
            "d_model": 64,
            "hidden_dim": 256,
        }
    else:
        model["presets"] = {
            "mdns_rope_deit2d": {
                "embed_dim": 96 if is_potts else 64,
                "num_blocks": 3 if is_potts else 4,
                "num_heads": 4,
                "mlp_ratio": 4.0,
                "dropout": 0.0,
                "dtype": "bfloat16",
                "use_ape": True,
                "rope_mixed": True,
                "rope_theta": 10.0,
            }
        }
        model["guidance_backbone"] = {"type": "mdns_rope_deit2d"}
    return model


def resolve_evaluation_config(
    raw: Mapping[str, Any], *, config_path: str | Path
) -> dict[str, Any]:
    """Resolve one compact evaluation leaf into checkpoint-compatible settings."""

    top = _mapping(raw, name="config")
    _strict_keys(top, allowed={"evaluation"}, name="config")
    spec = LatticeEvaluationSpec.parse(top["evaluation"])
    is_ising4 = spec.preset == "ising4"
    is_potts = spec.preset == "potts16"
    kind = "potts" if is_potts else "ising"
    num_states = 3 if is_potts else 2
    problem_cfg: dict[str, Any] = {
        "L": 4 if is_ising4 else 16,
        "beta": spec.target_beta,
        "J": 1.0,
    }
    if is_potts:
        problem_cfg["q"] = 3
    else:
        problem_cfg["h"] = 0.1 if is_ising4 else 0.0
    return {
        "evaluation": {
            "schema": EVALUATION_SCHEMA_VERSION,
            "preset": spec.preset,
            "method": spec.method,
        },
        "device": "cuda",
        "problem": {"kind": kind},
        kind: problem_cfg,
        "vocab": {"num_states": num_states},
        "schedule": {"type": "cosine", "eps": 1.0e-4},
        "forward": {"kind": "uniform_replace"},
        "model": _resolved_model(spec),
        "rollout": {
            "sampler_mode": "direct_q_ctmc",
            "num_steps": 256,
            "batch_size": 16384 if is_ising4 else 256,
            "guidance_strength": 1.0,
            "progress_bar": True,
        },
        "benchmark": (
            {"empirical_smoothing_eps": 1.0e-12}
            if is_ising4
            else {"correlation_direction": "x", "distance_mode": "signed"}
        ),
        "_config_path": str(Path(config_path).expanduser().resolve()),
    }


__all__ = [
    "EVALUATION_METHODS",
    "EVALUATION_PRESETS",
    "EVALUATION_SCHEMA_VERSION",
    "LatticeEvaluationSpec",
    "PreconditionerSpec",
    "resolve_evaluation_config",
]
