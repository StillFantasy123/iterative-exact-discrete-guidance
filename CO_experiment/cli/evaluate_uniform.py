"""Evaluate the single-seed Uniform baseline on all frozen Max-Cut test sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from CO_experiment.data.manifests import load_graph_manifest
from CO_experiment.evaluation.metrics import (
    aggregate_graph_metrics,
    evaluate_samples_for_graph,
    random_samples_for_graph,
)
from CO_experiment.oracle.certificates import require_embedded_oracles


SCALES = {
    "ba20": "BA[20,32]",
    "ba40": "BA[40,64]",
    "ba100": "BA[100,128]",
}
ROLLOUT_SEED = 42
SAMPLES_PER_GRAPH = 512


def _manifest_path(scale: str) -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "data/manifests"
        / f"maxcut_{scale}"
        / "test_graphs.json.gz"
    )


def evaluate_uniform() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return formal metric and sample payloads for all three BA scales."""

    metric_buckets: dict[str, Any] = {}
    sample_buckets: dict[str, Any] = {}
    device = torch.device("cpu")
    for scale, label in SCALES.items():
        manifest_path = _manifest_path(scale)
        graphs = load_graph_manifest(manifest_path, split="test", problem="maxcut")
        graphs, _ = require_embedded_oracles(graphs, split="test")
        torch.manual_seed(ROLLOUT_SEED)
        graph_metrics: list[dict[str, Any]] = []
        graph_samples: list[dict[str, Any]] = []
        for graph in graphs:
            samples = random_samples_for_graph(
                graph, SAMPLES_PER_GRAPH, device=device
            )
            graph_metrics.append(
                evaluate_samples_for_graph(graph=graph, samples=samples)
            )
            graph_samples.append(
                {
                    "graph_name": graph.name,
                    "num_nodes": int(graph.num_nodes),
                    "samples": samples.detach().cpu().to(torch.uint8),
                }
            )
        metric_buckets[label] = {
            "scale": scale,
            "manifest": str(manifest_path),
            "aggregate": aggregate_graph_metrics(graph_metrics),
        }
        sample_buckets[label] = {
            "scale": scale,
            "manifest": str(manifest_path),
            "graphs": graph_samples,
        }

    return (
        {
            "schema_version": "maxcut_uniform_evaluation_v2",
            "baseline": "Uniform",
            "num_samples_per_graph": SAMPLES_PER_GRAPH,
            "rollout_seed": ROLLOUT_SEED,
            "buckets": metric_buckets,
        },
        {
            "schema_version": "maxcut_uniform_samples_v2",
            "num_samples_per_graph": SAMPLES_PER_GRAPH,
            "rollout_seed": ROLLOUT_SEED,
            "buckets": sample_buckets,
        },
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/maxcut_uniform"),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, samples = evaluate_uniform()
    metrics_path = output_dir / "paper_metrics.json"
    samples_path = output_dir / "generated_samples.pt"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    torch.save(samples, samples_path)
    print(json.dumps({"metrics": str(metrics_path), "samples": str(samples_path)}, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "ROLLOUT_SEED",
    "SAMPLES_PER_GRAPH",
    "SCALES",
    "evaluate_uniform",
    "main",
    "parse_args",
]
