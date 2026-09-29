"""Resample one manifest-registered Ising16 or Potts16 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from edg_experiment.config.load import load_config
from edg_experiment.dfm.schedules import KappaSchedule
from edg_experiment.lattice import build_problem
from edg_experiment.lattice.processes.factory import (
    build_base_posterior_model,
    validate_process_cfg,
)
from edg_experiment.lattice.release import ReleaseResult, load_release_result
from edg_experiment.lattice.runtime import build_guidance, collect_samples
from edg_experiment.utils.io import ensure_dir, save_json
from edg_experiment.utils.random import set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, help="Lattice result id in the manifest")
    parser.add_argument("--manifest", default="artifacts/manifest.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _benchmark(problem_kind: str):
    if problem_kind == "ising":
        from ising16_experiment.evaluation.benchmark_ising16 import benchmark_sample_sets

        return benchmark_sample_sets
    if problem_kind == "potts":
        from potts16_experiment.evaluation.benchmark_potts16 import benchmark_sample_sets

        return benchmark_sample_sets
    raise ValueError(f"Unsupported lattice problem: {problem_kind}")


def _load_samples(path: Path, *, release: ReleaseResult) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if torch.is_tensor(payload):
        samples = payload
    elif isinstance(payload, dict):
        tensors = [
            payload[key]
            for key in ("samples_binary", "samples_uint8", "samples")
            if torch.is_tensor(payload.get(key))
        ]
        if len(tensors) != 1:
            raise ValueError("frozen reference must contain exactly one sample tensor")
        samples = tensors[0]
    else:
        raise ValueError("frozen reference has an unsupported payload type")
    if samples.is_floating_point() or samples.is_complex():
        raise ValueError("frozen reference samples must use an integer dtype")
    samples = samples.detach().to(device="cpu", dtype=torch.uint8)
    expected_shape = (
        int(release.evaluation["reference_num_samples"]),
        int(release.problem["lattice_size"]) ** 2,
    )
    if tuple(samples.shape) != expected_shape:
        raise ValueError(
            f"frozen reference shape {tuple(samples.shape)} != {expected_shape}"
        )
    num_states = int(release.problem.get("num_states", 2))
    if int(samples.min()) < 0 or int(samples.max()) >= num_states:
        raise ValueError("frozen reference contains an out-of-range state")
    return samples


def _prepare_output_dir(path: Path, *, force: bool) -> None:
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(path)
    if path.is_dir() and any(path.iterdir()) and not force:
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _sample_payload(samples: torch.Tensor, *, problem_kind: str) -> Any:
    values = samples.detach().cpu().to(torch.uint8)
    return {"samples_uint8": values} if problem_kind == "potts" else values


def _validate_protocol(release: ReleaseResult, cfg: dict[str, Any]) -> tuple[int, int, float]:
    protocol = release.evaluation_protocol
    if str(protocol["mode"]) != "paper" or str(protocol["reference_split"]) != "test":
        raise ValueError("release lattice evaluation requires the frozen paper/test protocol")
    if int(protocol["reference_seed"]) != int(
        release.evaluation["reference_sample_cfg"]["seed"]
    ):
        raise ValueError("manifest reference seeds disagree")
    reference_cfg = release.evaluation["reference_sample_cfg"]
    reference_count = int(reference_cfg["batch_size"]) * int(reference_cfg["num_collect"])
    if reference_count != int(release.evaluation["reference_num_samples"]):
        raise ValueError("manifest reference sample count is inconsistent")
    sampling_steps = int(protocol["sampling_steps"])
    if int(cfg["rollout"]["num_steps"]) != sampling_steps:
        raise ValueError("paper config and manifest disagree on sampling steps")
    guidance_strength = float(protocol["guidance_strength"])
    if guidance_strength != 1.0:
        raise ValueError("release lattice evaluation requires guidance strength 1")
    return int(protocol["rollout_samples"]), int(protocol["rollout_seed"]), guidance_strength


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    release = load_release_result(args.manifest, args.result)
    cfg = load_config(release.config_path)
    if args.device is not None:
        cfg["device"] = str(args.device)
    validate_process_cfg(cfg)
    problem = build_problem(cfg)
    num_samples, rollout_seed, guidance_strength = _validate_protocol(release, cfg)

    output_dir = Path(args.output_dir).expanduser().resolve()
    _prepare_output_dir(output_dir, force=bool(args.force))
    samples_dir = ensure_dir(output_dir / "samples")
    metrics_dir = ensure_dir(output_dir / "metrics")

    device = torch.device(cfg["device"])
    state = torch.load(release.checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or not state or not all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()
    ):
        raise ValueError("release checkpoint must be a raw tensor state dict")

    schedule = KappaSchedule(
        kind=str(cfg["schedule"]["type"]), eps=float(cfg["schedule"]["eps"])
    )
    base_model = build_base_posterior_model(cfg=cfg, schedule=schedule, device=device)
    guidance_model = build_guidance(cfg=cfg, device=device)
    guidance_model.load_state_dict(state, strict=True)
    guidance_model.eval()

    set_seed(rollout_seed)
    generated, _ = collect_samples(
        base_model=base_model,
        guidance_model=guidance_model,
        cfg=cfg,
        schedule=schedule,
        device=device,
        total_samples=num_samples,
        guidance_strength=guidance_strength,
        show_progress=bool(cfg.get("rollout", {}).get("progress_bar", True)),
        desc=f"evaluate {release.result_id}",
    )
    generated_path = samples_dir / "generated_samples.pt"
    torch.save(_sample_payload(generated, problem_kind=problem.kind), generated_path)

    reference = _load_samples(release.reference_samples_path, release=release)
    benchmark = _benchmark(problem.kind)
    result = benchmark(
        guided_samples=generated,
        cfg=cfg,
        run_dir=output_dir,
        reference_split="test",
        reference_samples=reference,
    )
    receipt = {
        "schema_version": "lattice_evaluation_receipt_v2",
        "release_result": release.result_id,
        "checkpoint": str(release.checkpoint_path),
        "checkpoint_step": int(release.checkpoint_step),
        "weights": release.weight_source,
        "guidance_strength": guidance_strength,
        "num_samples": num_samples,
        "rollout_seed": rollout_seed,
        "reference_split": "test",
        "reference_samples": str(release.reference_samples_path),
        "generated_samples": str(generated_path),
        "paper_metrics": str(metrics_dir / "paper_metrics.json"),
        "metrics": result["guided"]["paper"],
    }
    receipt_path = metrics_dir / "evaluation_receipt.json"
    save_json(receipt_path, receipt)
    print(f"Saved generated samples: {generated_path}")
    print(f"Saved paper metrics: {metrics_dir / 'paper_metrics.json'}")


if __name__ == "__main__":
    main()


__all__ = ["main", "parse_args"]
