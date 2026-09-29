"""Evaluate generated Potts 16x16 samples against a frozen reference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from edg_experiment.lattice import build_problem
from edg_experiment.lattice.evaluation.metrics_contract import partition_metrics
from edg_experiment.lattice.evaluation.output import write_paper_metrics
from potts16_experiment.evaluation.metrics import observable_metrics


def benchmark_sample_sets(
    *,
    guided_samples: torch.Tensor,
    reference_samples: torch.Tensor,
    cfg: Dict[str, Any],
    run_dir: Path | None = None,
    reference_split: str = "test",
) -> Dict[str, Any]:
    if reference_split != "test":
        raise ValueError("release evaluation accepts only the frozen test reference")
    problem = build_problem(cfg)
    if problem.kind != "potts":
        raise ValueError("Potts benchmark requires problem.kind=potts")
    generated = guided_samples.detach().to(device="cpu", dtype=torch.uint8)
    reference = reference_samples.detach().to(device="cpu", dtype=torch.uint8)
    problem.validate_samples(generated)
    problem.validate_samples(reference)
    benchmark = cfg["benchmark"]
    flat = observable_metrics(
        generated,
        reference,
        L=problem.L,
        q=problem.num_states,
        J=problem.J,
        direction=str(benchmark.get("correlation_direction", "x")),
        distance_mode=str(benchmark.get("distance_mode", "signed")),
    )
    guided = partition_metrics(
        flat, problem_kind="potts", lattice_size=problem.L
    ).as_dict()
    result: Dict[str, Any] = {
        "reference_split": "test",
        "reference_num_samples": int(reference.shape[0]),
        "guided": guided,
    }
    if run_dir is not None:
        result["paper_metrics_path"] = write_paper_metrics(
            run_dir=run_dir,
            guided=guided,
            context={
                "problem_kind": "potts",
                "lattice_size": problem.L,
                "num_states": problem.num_states,
                "reference_split": "test",
                "reference_num_samples": int(reference.shape[0]),
            },
        )
    return result


__all__ = ["benchmark_sample_sets"]
