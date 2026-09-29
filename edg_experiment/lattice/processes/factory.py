from __future__ import annotations

from typing import Any, Dict

import torch

from edg_experiment.dfm.schedules import KappaSchedule
from edg_experiment.lattice.problem import build_problem
from edg_experiment.lattice.processes.analytic_base import AnalyticUniformReplacePosteriorModel

_ALLOWED_FORWARD_KINDS = {"uniform_replace"}


def get_forward_kind(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("forward", {}).get("kind", "uniform_replace")).lower()


def terminal_num_states(cfg: Dict[str, Any]) -> int:
    return int(cfg["vocab"]["num_states"])


def input_vocab_size(cfg: Dict[str, Any]) -> int:
    return terminal_num_states(cfg)


def validate_process_cfg(cfg: Dict[str, Any]) -> None:
    kind = get_forward_kind(cfg)
    if kind not in _ALLOWED_FORWARD_KINDS:
        raise ValueError(f"forward.kind must be one of {sorted(_ALLOWED_FORWARD_KINDS)}, got {kind}")

    build_problem(cfg)


def build_base_posterior_model(
    cfg: Dict[str, Any],
    *,
    schedule: KappaSchedule,
    device: torch.device,
):
    validate_process_cfg(cfg)
    num_states = terminal_num_states(cfg)
    return AnalyticUniformReplacePosteriorModel(schedule=schedule, num_states=num_states).to(device)


__all__ = [
    "build_base_posterior_model",
    "get_forward_kind",
    "input_vocab_size",
    "terminal_num_states",
    "validate_process_cfg",
]
