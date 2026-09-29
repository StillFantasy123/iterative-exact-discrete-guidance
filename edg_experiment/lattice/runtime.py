"""Shared model construction and guided sampling for lattice evaluation."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from tqdm.auto import tqdm

from edg_experiment.dfm.schedules import KappaSchedule
from edg_experiment.lattice.models.guidance_head import GuidanceHead
from edg_experiment.lattice.problem import build_problem, problem_config
from edg_experiment.lattice.processes.factory import (
    get_forward_kind,
    input_vocab_size,
    validate_process_cfg,
)
from edg_experiment.lattice.processes.reverse_sampler import (
    teacher_reverse_direct_q_ctmc_rollout,
)


def build_guidance(cfg: Dict[str, Any], device: torch.device) -> GuidanceHead:
    validate_process_cfg(cfg)
    problem = build_problem(cfg)
    schedule_cfg = cfg.get("schedule", {})
    schedule = KappaSchedule(
        kind=str(schedule_cfg.get("type", "cosine")),
        eps=float(schedule_cfg.get("eps", 1.0e-4)),
    )
    return GuidanceHead(
        cfg=cfg["model"],
        length=problem.length,
        num_states=problem.num_states,
        input_vocab_size=input_vocab_size(cfg),
        forward_kind=get_forward_kind(cfg),
        schedule=schedule,
        ising_cfg=dict(problem_config(cfg)),
        problem_kind=problem.kind,
    ).to(device)


def collect_samples(
    *,
    base_model: Any,
    guidance_model: GuidanceHead,
    cfg: Dict[str, Any],
    schedule: KappaSchedule,
    device: torch.device,
    total_samples: int,
    guidance_strength: float,
    show_progress: bool = True,
    desc: str = "sampling",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Draw terminal samples with one released guidance checkpoint."""

    validate_process_cfg(cfg)
    if total_samples <= 0:
        raise ValueError("total_samples must be positive")
    batch_size = int(cfg["rollout"]["batch_size"])
    sampler_mode = str(cfg["rollout"]["sampler_mode"]).lower()
    if sampler_mode != "direct_q_ctmc":
        raise ValueError("release evaluation requires rollout.sampler_mode=direct_q_ctmc")
    chunks: list[torch.Tensor] = []
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    metadata: Dict[str, Any] = {}
    remaining = int(total_samples)
    iterator = tqdm(
        range((remaining + batch_size - 1) // batch_size),
        desc=desc,
        disable=not show_progress,
        leave=False,
    )
    for _ in iterator:
        current = min(batch_size, remaining)
        samples, debug = teacher_reverse_direct_q_ctmc_rollout(
            theta_prev=base_model,
            phi_k=guidance_model,
            cfg=cfg,
            schedule=schedule,
            device=device,
            batch_size=current,
            guidance_strength=float(guidance_strength),
            return_debug=True,
            show_progress=False,
        )
        chunks.append(samples)
        for key, value in debug.items():
            if isinstance(value, (int, float)):
                sums[key] = sums.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
            else:
                metadata[key] = value
        remaining -= current
        if show_progress:
            iterator.set_postfix({"done": f"{total_samples - remaining}/{total_samples}"})

    summary = {key: sums[key] / counts[key] for key in sums}
    summary.update(metadata)
    summary.update(
        {
            "sampler_mode": sampler_mode,
            "preconditioning_enabled": float(
                bool(getattr(guidance_model, "preconditioning_enabled", False))
            ),
            "log_h_source": str(getattr(guidance_model, "log_h_source", "unknown")),
        }
    )
    return torch.cat(chunks, dim=0), summary


__all__ = ["build_guidance", "collect_samples"]
