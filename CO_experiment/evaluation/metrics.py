"""Paper metrics for the released Max-Cut evaluations."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor
from tqdm.auto import tqdm

from CO_experiment.data.objectives import summarize_samples
from CO_experiment.data.types import GraphInstance, collate_graphs
from CO_experiment.protocol import PROBLEM
from CO_experiment.sampling.sampler import collect_samples_for_graph
from edg_experiment.dfm.schedules import KappaSchedule


PAPER_METRIC_KEYS = ("best_approx_ratio_mean", "sample_approx_ratio_mean")


def paper_metrics(
    metrics: dict[str, Any], *, num_samples_per_graph: int, rollout_seed: int
) -> dict[str, Any]:
    missing = [key for key in PAPER_METRIC_KEYS if key not in metrics]
    if missing:
        raise ValueError("missing paper metrics: " + ", ".join(missing))
    return {
        "schema_version": "maxcut_paper_metrics_v1",
        **{key: float(metrics[key]) for key in PAPER_METRIC_KEYS},
        "num_graphs": int(metrics["num_graphs"]),
        "num_samples_per_graph": int(num_samples_per_graph),
        "rollout_seed": int(rollout_seed),
    }


def random_samples_for_graph(
    graph: GraphInstance, num_samples: int, *, device: torch.device
) -> Tensor:
    return torch.randint(
        0,
        2,
        (int(num_samples), int(graph.num_nodes)),
        device=device,
        dtype=torch.long,
    )


def evaluate_samples_for_graph(
    *, graph: GraphInstance, samples: Tensor
) -> dict[str, Any]:
    batch = collate_graphs(
        [graph for _ in range(int(samples.shape[0]))], device=samples.device
    )
    padded = torch.zeros(
        (int(samples.shape[0]), batch.max_num_nodes),
        device=samples.device,
        dtype=torch.long,
    )
    padded[:, : graph.num_nodes] = samples.to(torch.long)
    summary = summarize_samples(padded, batch)
    summary["graph_name"] = graph.name
    return summary


def aggregate_graph_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("at least one graph record is required")
    if any(not bool(record.get("oracle_available", False)) for record in records):
        raise ValueError("all release graphs must have certified optima")
    return {
        "num_graphs": len(records),
        "problem": PROBLEM,
        "best_approx_ratio_mean": float(
            sum(float(record["best_approx_ratio"]) for record in records) / len(records)
        ),
        "sample_approx_ratio_mean": float(
            sum(float(record["sample_approx_ratio_mean"]) for record in records)
            / len(records)
        ),
    }


@torch.no_grad()
def evaluate_graphs(
    *,
    graphs: Sequence[GraphInstance],
    guidance_model: Any,
    schedule: KappaSchedule,
    device: torch.device,
    rollout_cfg: dict[str, Any],
    num_samples_per_graph: int,
    show_progress: bool = False,
    desc: str = "maxcut eval",
    return_samples: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    for graph in tqdm(graphs, desc=desc, disable=not show_progress, leave=False):
        samples, _ = collect_samples_for_graph(
            graph=graph,
            guidance_model=guidance_model,
            schedule=schedule,
            device=device,
            total_samples=int(num_samples_per_graph),
            rollout_cfg=rollout_cfg,
        )
        records.append(evaluate_samples_for_graph(graph=graph, samples=samples))
        if return_samples:
            sample_records.append(
                {
                    "graph_name": graph.name,
                    "num_nodes": int(graph.num_nodes),
                    "samples": samples.detach().cpu().to(torch.uint8),
                }
            )
    return aggregate_graph_metrics(records), sample_records


__all__ = [
    "PAPER_METRIC_KEYS",
    "aggregate_graph_metrics",
    "evaluate_graphs",
    "evaluate_samples_for_graph",
    "paper_metrics",
    "random_samples_for_graph",
]
