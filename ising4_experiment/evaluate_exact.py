"""Evaluate one manifest-registered Ising 4x4 checkpoint exactly."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict

import torch

from edg_experiment.artifacts import ManifestResult, load_manifest_result
from edg_experiment.config.load import load_config
from edg_experiment.dfm.schedules import KappaSchedule
from edg_experiment.lattice.evaluation.metrics_contract import partition_metrics
from edg_experiment.lattice.processes.factory import build_base_posterior_model
from edg_experiment.lattice.runtime import build_guidance, collect_samples
from edg_experiment.utils.io import ensure_dir, save_json
from edg_experiment.utils.random import set_seed
from ising4_experiment.evaluation.exact_target import exact_ising_target
from ising4_experiment.evaluation.metrics import exact_terminal_metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--manifest", default="artifacts/manifest.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _validate_release(release: ManifestResult, cfg: Dict[str, Any]) -> tuple[int, int, int]:
    if release.evaluator != "ising4_exact":
        raise ValueError(f"{release.result_id} is not an Ising4 result")
    if release.checkpoint_format != "raw_pytorch_state_dict" or release.weight_source != "ema":
        raise ValueError("Ising4 release requires a raw EMA state dict")
    ising = cfg["ising"]
    checks = {
        "kind": (cfg["problem"]["kind"], release.problem["kind"]),
        "lattice_size": (int(ising["L"]), int(release.problem["lattice_size"])),
        "beta": (float(ising["beta"]), float(release.problem["beta"])),
        "J": (float(ising.get("J", 1.0)), float(release.problem.get("J", 1.0))),
        "h": (float(ising.get("h", 0.0)), float(release.problem.get("h", 0.0))),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"manifest {name} disagrees with the paper config")
    evaluation = release.evaluation
    resolved = cfg.get("evaluation", {})
    expected_method = "one_shot" if release.method == "one_shot_dgm" else "iedg"
    if resolved.get("schema") != "lattice-evaluation-v1" or resolved.get("preset") != "ising4":
        raise ValueError("Ising4 checkpoint requires an ising4 evaluation config")
    if resolved.get("method") != expected_method:
        raise ValueError("manifest method disagrees with the evaluation config")
    if str(cfg["rollout"]["sampler_mode"]) != "direct_q_ctmc":
        raise ValueError("Ising4 release requires the direct_q_ctmc sampler")
    if bool(cfg["model"]["preconditioning"]["enabled"]):
        raise ValueError("Ising4 release checkpoints do not use preconditioning")
    if float(evaluation["guidance_strength"]) != 1.0:
        raise ValueError("Ising4 release evaluation requires guidance strength 1")
    return (
        int(evaluation["rollout_samples"]),
        int(evaluation["sampling_steps"]),
        int(evaluation["rollout_seed"]),
    )


@torch.no_grad()
def evaluate_checkpoint(
    *,
    cfg: Dict[str, Any],
    release: ManifestResult,
    device: torch.device,
    num_samples: int,
    sampling_steps: int,
    seed: int,
) -> Dict[str, Any]:
    state = torch.load(release.checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or not state or not all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()
    ):
        raise ValueError("release checkpoint must be a raw tensor state dict")
    guidance = build_guidance(cfg=cfg, device=device)
    guidance.load_state_dict(state, strict=True)
    guidance.eval()
    schedule = KappaSchedule(
        kind=str(cfg["schedule"]["type"]), eps=float(cfg["schedule"]["eps"])
    )
    base = build_base_posterior_model(cfg, schedule=schedule, device=device)
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg["rollout"]["num_steps"] = sampling_steps
    set_seed(seed)
    samples, _ = collect_samples(
        base_model=base,
        guidance_model=guidance,
        cfg=eval_cfg,
        schedule=schedule,
        device=device,
        total_samples=num_samples,
        guidance_strength=1.0,
        show_progress=bool(eval_cfg["rollout"].get("progress_bar", True)),
        desc=f"evaluate {release.result_id}",
    )
    ising = cfg["ising"]
    metric_kwargs = {
        "beta": float(ising["beta"]),
        "L": int(ising["L"]),
        "J": float(ising.get("J", 1.0)),
        "h": float(ising.get("h", 0.0)),
        "smoothing_eps": float(cfg["benchmark"].get("empirical_smoothing_eps", 1.0e-12)),
    }
    metrics = partition_metrics(
        exact_terminal_metrics(samples, **metric_kwargs),
        problem_kind="ising",
        lattice_size=metric_kwargs["L"],
    ).as_dict()
    target = exact_ising_target(
        L=metric_kwargs["L"],
        beta=metric_kwargs["beta"],
        J=metric_kwargs["J"],
        h=metric_kwargs["h"],
    )
    floor = partition_metrics(
        exact_terminal_metrics(target.sample(num_samples=num_samples, seed=seed), **metric_kwargs),
        problem_kind="ising",
        lattice_size=metric_kwargs["L"],
    ).as_dict()
    return {
        "schema_version": "ising4_exact_evaluation_v2",
        "release_result": release.result_id,
        "checkpoint": str(release.checkpoint_path),
        "checkpoint_step": int(release.checkpoint_step),
        "weights": release.weight_source,
        "guidance_strength": 1.0,
        "num_samples": num_samples,
        "sampling_steps": sampling_steps,
        "rollout_seed": seed,
        "metrics": metrics,
        "exact_categorical_sampling_floor": floor,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    release = load_manifest_result(
        args.manifest, args.result, expected_evaluator="ising4_exact"
    )
    cfg = load_config(release.config_path)
    if args.device is not None:
        cfg["device"] = str(args.device)
    num_samples, sampling_steps, seed = _validate_release(release, cfg)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    result = evaluate_checkpoint(
        cfg=cfg,
        release=release,
        device=torch.device(cfg["device"]),
        num_samples=num_samples,
        sampling_steps=sampling_steps,
        seed=seed,
    )
    ensure_dir(output.parent)
    save_json(output, result)
    print(f"Saved exact evaluation: {output}")


if __name__ == "__main__":
    main()


__all__ = ["evaluate_checkpoint", "main", "parse_args"]
