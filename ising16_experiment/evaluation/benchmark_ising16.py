"""Evaluate generated Ising 16x16 samples against a frozen reference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from edg_experiment.lattice.evaluation.metrics_contract import partition_metrics
from edg_experiment.lattice.evaluation.output import write_paper_metrics
from ising16_experiment.evaluation.metrics import observable_metrics


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
    L = int(cfg["ising"]["L"])
    generated = guided_samples.detach().to(device="cpu", dtype=torch.uint8)
    reference = reference_samples.detach().to(device="cpu", dtype=torch.uint8)
    for name, values in (("generated", generated), ("reference", reference)):
        if values.ndim != 2 or int(values.shape[1]) != L * L:
            raise ValueError(f"{name} samples must have shape [N,{L * L}]")
        if int(values.shape[0]) == 0 or int(values.min()) < 0 or int(values.max()) > 1:
            raise ValueError(f"{name} samples must be non-empty and binary")
    benchmark = cfg["benchmark"]
    flat = observable_metrics(
        generated,
        reference,
        L=L,
        J=float(cfg["ising"].get("J", 1.0)),
        h=float(cfg["ising"].get("h", 0.0)),
        direction=str(benchmark.get("correlation_direction", "x")),
        distance_mode=str(benchmark.get("distance_mode", "signed")),
    )
    guided = partition_metrics(flat, problem_kind="ising", lattice_size=L).as_dict()
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
                "problem_kind": "ising",
                "lattice_size": L,
                "reference_split": "test",
                "reference_num_samples": int(reference.shape[0]),
            },
        )
    return result


__all__ = ["benchmark_sample_sets"]
