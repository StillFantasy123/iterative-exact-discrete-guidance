"""Evaluate one manifest-registered Max-Cut checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from CO_experiment.data.manifests import load_graph_manifest
from CO_experiment.evaluation.metrics import evaluate_graphs, paper_metrics
from CO_experiment.models.guidance import GraphGuidanceHead
from CO_experiment.oracle.certificates import require_embedded_oracles
from CO_experiment.protocol import GUIDANCE_STRENGTH, PROBLEM, SCHEDULE_EPS, SCHEDULE_KIND
from edg_experiment.artifacts import load_manifest_result
from edg_experiment.config.load import load_config
from edg_experiment.dfm.schedules import KappaSchedule
from edg_experiment.utils.io import ensure_dir, save_json
from edg_experiment.utils.random import set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--manifest", default="artifacts/manifest.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def _device(value: str) -> torch.device:
    if value.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _prepare_output(path: Path) -> Path:
    target = path.expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"output directory is not empty: {target}")
    return ensure_dir(target)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    release = load_manifest_result(
        args.manifest, args.result, expected_evaluator="maxcut"
    )
    if release.checkpoint_format != "raw_pytorch_state_dict" or release.weight_source != "ema":
        raise ValueError("Max-Cut release evaluation requires a raw EMA state dict")
    cfg = load_config(release.config_path)
    if args.device is not None:
        cfg["device"] = str(args.device)
    resolved = cfg.get("evaluation", {})
    expected_method = "one_shot" if release.method == "one_shot_dgm" else "iedg"
    if resolved.get("schema") != "maxcut-evaluation-v1":
        raise ValueError("Max-Cut checkpoint requires a Max-Cut evaluation config")
    if resolved.get("method") != expected_method:
        raise ValueError("manifest method disagrees with the evaluation config")
    if resolved.get("scale") != release.problem.get("scale"):
        raise ValueError("manifest scale disagrees with the evaluation config")
    evaluation = release.evaluation
    num_samples = int(evaluation["samples_per_graph"])
    rollout_seed = int(evaluation["rollout_seed"])
    sampling_steps = int(evaluation["sampling_steps"])
    if int(cfg["experiment"]["seed"]) != rollout_seed:
        raise ValueError("paper config and manifest disagree on rollout seed")
    if int(cfg["rollout"]["num_steps"]) != sampling_steps:
        raise ValueError("paper config and manifest disagree on sampling steps")
    if float(evaluation["guidance_strength"]) != GUIDANCE_STRENGTH:
        raise ValueError("manifest disagrees with the fixed Max-Cut guidance strength")
    if release.problem.get("kind") != PROBLEM:
        raise ValueError("manifest result is not Max-Cut")

    graph_path = release.repository_root / str(evaluation["graph_manifest"])
    graphs = load_graph_manifest(graph_path, split="test")
    if len(graphs) != int(evaluation["graph_count"]):
        raise ValueError("test graph count does not match the manifest")
    graphs, oracle_metadata = require_embedded_oracles(graphs, split="test")

    device = _device(str(cfg["device"]))
    state = torch.load(release.checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or not state or not all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()
    ):
        raise ValueError("release checkpoint must be a raw tensor state dict")
    model = GraphGuidanceHead.from_config(dict(cfg["model"])).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    set_seed(rollout_seed)
    metrics, sample_records = evaluate_graphs(
        graphs=graphs,
        guidance_model=model,
        schedule=KappaSchedule(kind=SCHEDULE_KIND, eps=SCHEDULE_EPS),
        device=device,
        rollout_cfg=dict(cfg["rollout"]),
        num_samples_per_graph=num_samples,
        show_progress=bool(cfg.get("eval", {}).get("progress_bar", False)),
        desc=f"evaluate {release.result_id}",
        return_samples=True,
    )
    output = _prepare_output(Path(args.output_dir))
    samples_path = output / "generated_samples.pt"
    torch.save(
        {
            "schema_version": "maxcut_generated_samples_v2",
            "release_result": release.result_id,
            "rollout_seed": rollout_seed,
            "num_samples_per_graph": num_samples,
            "sampling_steps": sampling_steps,
            "guidance_strength": GUIDANCE_STRENGTH,
            "graphs": sample_records,
        },
        samples_path,
    )
    save_json(
        output / "paper_metrics.json",
        paper_metrics(
            metrics,
            num_samples_per_graph=num_samples,
            rollout_seed=rollout_seed,
        ),
    )
    save_json(
        output / "evaluation_receipt.json",
        {
            "schema_version": "maxcut_evaluation_receipt_v2",
            "release_result": release.result_id,
            "checkpoint": str(release.checkpoint_path),
            "checkpoint_step": int(release.checkpoint_step),
            "weights": release.weight_source,
            "graph_manifest": str(graph_path),
            "oracle": oracle_metadata,
            "rollout_seed": rollout_seed,
            "num_samples_per_graph": num_samples,
            "sampling_steps": sampling_steps,
            "guidance_strength": GUIDANCE_STRENGTH,
            "generated_samples": str(samples_path),
        },
    )
    print(f"Saved paper metrics: {output / 'paper_metrics.json'}")


if __name__ == "__main__":
    main()


__all__ = ["main", "parse_args"]
