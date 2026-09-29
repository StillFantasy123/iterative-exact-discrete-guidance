"""Validate the manifest-defined evaluation release.

``artifacts/manifest.yaml`` is the sole release inventory. Validation is
content based: all 20 checkpoint payloads are validated and strictly loaded
into their registered models, and the six
frozen L16 IEDG sample/metric payloads are checked against the declared problem
and evaluation protocol. The command prints the paper metrics recomputed from
the frozen samples; no training runs or network access are required.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

from edg_experiment.config.load import load_config
from edg_experiment.artifacts import (
    load_manifest_result,
    release_filenames,
    repository_root_for_manifest,
    resolve_manifest_path,
)
from edg_experiment.lattice import build_problem
from edg_experiment.lattice.evaluation.metrics_contract import (
    paper_metric_names,
    partition_metrics,
)
from edg_experiment.lattice.runtime import build_guidance
from ising16_experiment.evaluation.metrics import (
    observable_metrics as ising_observable_metrics,
)
from potts16_experiment.evaluation.metrics import (
    observable_metrics as potts_observable_metrics,
)


SCHEMA_VERSION = "paper_release_v4"
FROZEN_L16_RESULTS = {
    "ising16_beta028",
    "ising16_beta04407",
    "ising16_beta06",
    "potts16_beta05",
    "potts16_beta1005",
    "potts16_beta12",
}
EXPECTED_ADDITIONAL_CHECKPOINTS = {
    "one_shot_ising16_beta028",
    "one_shot_ising16_beta04407",
    "one_shot_ising16_beta06",
    "one_shot_potts16_beta05",
    "one_shot_potts16_beta1005",
    "one_shot_potts16_beta12",
    "one_shot_ising4_beta06",
    "iedg_ising4_beta06",
    "one_shot_maxcut_ba20",
    "iedg_maxcut_ba20",
    "one_shot_maxcut_ba40",
    "iedg_maxcut_ba40",
    "one_shot_maxcut_ba100",
    "iedg_maxcut_ba100",
}
EXPECTED_RESULTS = FROZEN_L16_RESULTS | EXPECTED_ADDITIONAL_CHECKPOINTS
EVALUATION_FILE_KEYS = (
    "generated_samples",
    "reference_samples",
    "metrics",
)


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("release manifest must be a mapping")
    return payload


def _assert_machine_independent(value: Any, *, location: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_machine_independent(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_machine_independent(item, location=f"{location}[{index}]")
    elif isinstance(value, str):
        if value.startswith(("/mnt/", "/Users/", "\\\\", "~/")):
            raise ValueError(f"machine-specific path at {location}: {value}")


def _safe_repo_path(relative: Any, *, field: str) -> Path:
    value = Path(str(relative))
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"{field} must be a repository-relative path: {relative}")
    resolved = (REPO_ROOT / value).resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise ValueError(f"{field} escapes the repository: {relative}")
    return resolved


def _artifact_root(manifest: Mapping[str, Any]) -> Path:
    return _safe_repo_path(manifest.get("artifact_root", "artifacts"), field="artifact_root")


def _assert_close(actual: Any, expected: Any, *, field: str) -> None:
    if isinstance(expected, float):
        if abs(float(actual) - expected) > 1.0e-10:
            raise ValueError(f"{field} mismatch: {actual!r} != {expected!r}")
    elif actual != expected:
        raise ValueError(f"{field} mismatch: {actual!r} != {expected!r}")


def _validate_problem(
    *, result_id: str, entry: Mapping[str, Any], cfg: Mapping[str, Any]
) -> tuple[str, int, int]:
    declared = entry.get("problem")
    if not isinstance(declared, dict):
        raise ValueError(f"results.{result_id}.problem must be a mapping")
    problem = build_problem(dict(cfg))
    expected_kind = str(declared.get("kind"))
    _assert_close(problem.kind, expected_kind, field=f"{result_id}.problem.kind")
    _assert_close(problem.L, int(declared.get("lattice_size")), field=f"{result_id}.problem.lattice_size")
    _assert_close(problem.num_states, int(declared.get("num_states", 2)), field=f"{result_id}.problem.num_states")
    problem_cfg = cfg[expected_kind]
    for key in ("beta", "J"):
        _assert_close(problem_cfg[key], declared[key], field=f"{result_id}.problem.{key}")
    if expected_kind == "ising":
        _assert_close(problem_cfg.get("h", 0.0), declared.get("h", 0.0), field=f"{result_id}.problem.h")
    return problem.kind, problem.L, problem.num_states


def _validate_config(result_id: str, entry: Mapping[str, Any]) -> tuple[dict[str, Any], str, int, int]:
    config_path = _safe_repo_path(entry.get("config"), field=f"results.{result_id}.config")
    if not config_path.is_file():
        raise FileNotFoundError(f"missing paper config for {result_id}: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _assert_machine_independent(raw, location=str(config_path.relative_to(REPO_ROOT)))
    cfg = load_config(config_path)
    evaluation = cfg.get("evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("method") != "iedg":
        raise ValueError(f"{result_id} must resolve from a method=iedg evaluation leaf")
    if evaluation.get("schema") != "lattice-evaluation-v1":
        raise ValueError(f"{result_id} must use the lattice evaluation schema")
    kind, lattice_size, num_states = _validate_problem(
        result_id=result_id, entry=entry, cfg=cfg
    )
    step = int(entry.get("checkpoint_step", 0))
    if step <= 0:
        raise ValueError(f"{result_id}.checkpoint_step must be positive")
    return cfg, kind, lattice_size, num_states


def _validate_initialization(manifest: Mapping[str, Any]) -> None:
    results = manifest["results"]
    for result_id, entry in results.items():
        initialization = entry.get("initialization")
        if initialization is None:
            continue
        if initialization == "analytic_uniform":
            continue
        if not isinstance(initialization, dict) or set(initialization) != {"result"}:
            raise ValueError(f"{result_id}.initialization must name one result or analytic_uniform")
        parent_id = str(initialization["result"])
        if parent_id not in results or parent_id == result_id:
            raise ValueError(f"{result_id} has invalid initialization result {parent_id!r}")
        parent_problem = results[parent_id]["problem"]
        if parent_problem["kind"] != entry["problem"]["kind"]:
            raise ValueError(f"{result_id} initialization crosses problem families")
        if float(parent_problem["beta"]) >= float(entry["problem"]["beta"]):
            raise ValueError(f"{result_id} initialization must come from a lower beta")


def _extract_samples(payload: Any, *, result_id: str, role: str) -> torch.Tensor:
    if torch.is_tensor(payload):
        samples = payload
    elif isinstance(payload, dict):
        keys = [key for key in ("samples_binary", "samples_uint8", "samples") if torch.is_tensor(payload.get(key))]
        if len(keys) != 1:
            raise ValueError(f"{result_id} {role} payload must contain exactly one sample tensor")
        samples = payload[keys[0]]
    else:
        raise ValueError(f"{result_id} {role} payload has unsupported type {type(payload).__name__}")
    if samples.ndim != 2:
        raise ValueError(f"{result_id} {role} samples must have shape [N,L^2]")
    return samples


def _validate_sample_file(
    path: Path,
    *,
    result_id: str,
    role: str,
    expected_count: int,
    length: int,
    num_states: int,
) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    samples = _extract_samples(payload, result_id=result_id, role=role)
    if tuple(samples.shape) != (expected_count, length):
        raise ValueError(
            f"{result_id} {role} shape mismatch: {tuple(samples.shape)} != {(expected_count, length)}"
        )
    if samples.is_floating_point() or samples.is_complex():
        raise ValueError(f"{result_id} {role} samples must have an integer dtype")
    if int(samples.min()) < 0 or int(samples.max()) >= num_states:
        raise ValueError(f"{result_id} {role} samples fall outside [0,{num_states - 1}]")
    return samples.detach().to(device="cpu", dtype=torch.uint8)


def _validate_metrics(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    result_id: str,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "paper_lattice_evaluation_v2":
        raise ValueError(f"{result_id} metrics use an unexpected schema")
    _assert_close(payload.get("checkpoint_step"), int(entry["checkpoint_step"]), field=f"{result_id}.metrics.checkpoint_step")
    for key, expected in entry["problem"].items():
        _assert_close(payload["problem"].get(key), expected, field=f"{result_id}.metrics.problem.{key}")
    protocol = manifest["evaluation_protocol"]
    recorded = payload.get("evaluation", {})
    comparisons = {
        "mode": protocol["mode"],
        "selection": protocol["selection"],
        "reference_split": protocol["reference_split"],
        "reference_seed": protocol["reference_seed"],
        "rollout_seed": protocol["rollout_seed"],
        "rollout_num_samples": protocol["rollout_samples"],
        "guidance_strength": protocol["guidance_strength"],
        "sampling_steps": protocol["sampling_steps"],
        "evaluation_replicates": protocol["evaluation_replicates"],
        "histogram_binning": protocol["histogram_binning"],
        "weights": manifest["checkpoint_format"]["weights"],
        "reference_num_samples": int(entry["evaluation"]["reference_num_samples"]),
    }
    for key, expected in comparisons.items():
        _assert_close(recorded.get(key), expected, field=f"{result_id}.metrics.evaluation.{key}")
    if not isinstance(payload.get("metrics"), dict) or not payload["metrics"]:
        raise ValueError(f"{result_id} metrics payload is empty")
    expected_metric_keys = set(
        paper_metric_names(
            str(entry["problem"]["kind"]),
            lattice_size=int(entry["problem"]["lattice_size"]),
        )
    )
    actual_metric_keys = set(payload["metrics"])
    if actual_metric_keys != expected_metric_keys:
        missing = sorted(expected_metric_keys - actual_metric_keys)
        extra = sorted(actual_metric_keys - expected_metric_keys)
        raise ValueError(
            f"{result_id} metrics do not match the paper contract; "
            f"missing={missing}, extra={extra}"
        )
    reference_cfg = entry["evaluation"].get("reference_sample_cfg")
    if not isinstance(reference_cfg, dict):
        raise ValueError(f"{result_id} is missing reference_sample_cfg")
    _assert_close(
        reference_cfg.get("seed"),
        protocol["reference_seed"],
        field=f"{result_id}.reference_sample_cfg.seed",
    )
    declared_count = int(reference_cfg["batch_size"]) * int(
        reference_cfg["num_collect"]
    )
    _assert_close(
        declared_count,
        int(entry["evaluation"]["reference_num_samples"]),
        field=f"{result_id}.reference_sample_cfg.sample_count",
    )
    return payload


def _recompute_paper_metrics(
    *,
    cfg: Mapping[str, Any],
    problem_kind: str,
    generated: torch.Tensor,
    reference: torch.Tensor,
    lattice_size: int,
) -> dict[str, float]:
    benchmark = cfg.get("benchmark", {})
    direction = str(benchmark.get("correlation_direction", "x"))
    distance_mode = str(benchmark.get("distance_mode", "signed"))
    if problem_kind == "ising":
        problem = cfg["ising"]
        flat = ising_observable_metrics(
            samples_binary=generated,
            reference_binary=reference,
            L=lattice_size,
            J=float(problem.get("J", 1.0)),
            h=float(problem.get("h", 0.0)),
            direction=direction,
            distance_mode=distance_mode,
        )
    elif problem_kind == "potts":
        problem = cfg["potts"]
        flat = potts_observable_metrics(
            generated,
            reference,
            L=lattice_size,
            q=int(problem.get("q", cfg["vocab"]["num_states"])),
            J=float(problem.get("J", 1.0)),
            direction=direction,
            distance_mode=distance_mode,
        )
    else:
        raise ValueError(f"Unsupported release problem kind: {problem_kind}")
    grouped = partition_metrics(
        flat, problem_kind=problem_kind, lattice_size=lattice_size
    )
    return {key: float(value) for key, value in grouped.paper.items()}


def _validate_recomputed_metrics(
    *,
    result_id: str,
    recorded: Mapping[str, Any],
    recomputed: Mapping[str, float],
) -> None:
    for key, expected in recomputed.items():
        if key not in recorded:
            raise ValueError(f"{result_id}.metrics is missing paper metric {key}")
        actual = float(recorded[key])
        if not math.isclose(actual, expected, rel_tol=1.0e-10, abs_tol=1.0e-10):
            raise ValueError(
                f"{result_id}.metrics.{key} does not match frozen samples: "
                f"{actual!r} != {expected!r}"
            )


def _validate_checkpoint(path: Path, *, cfg: dict[str, Any], result_id: str) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{result_id} checkpoint is not a non-empty state dict")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise ValueError(f"{result_id} checkpoint contains training-only state")
    model = build_guidance(cfg=cfg, device=torch.device("cpu"))
    model.load_state_dict(state, strict=True)


def _validate_distribution(manifest: Mapping[str, Any]) -> None:
    distribution = manifest.get("distribution")
    if not isinstance(distribution, Mapping):
        raise ValueError("manifest.distribution must be a mapping")
    if distribution.get("layout") != "flat":
        raise ValueError("manifest.distribution.layout must be flat")
    if distribution.get("bundled") is not True:
        raise ValueError("manifest.distribution.bundled must be true")
    if set(distribution) != {"layout", "bundled"}:
        raise ValueError("bundled releases must not declare external artifact sources")


def _additional_checkpoint_entries(
    manifest: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    results = manifest.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("manifest.results must be a mapping")
    entries = {
        result_id: entry
        for result_id, entry in results.items()
        if result_id in EXPECTED_ADDITIONAL_CHECKPOINTS
    }
    if set(entries) != EXPECTED_ADDITIONAL_CHECKPOINTS:
        raise ValueError(
            "manifest.results must contain the complete 14-result non-frozen set"
        )
    return entries


def _validate_additional_checkpoint_metadata(
    checkpoint_id: str, entry: Mapping[str, Any]
) -> tuple[Path, dict[str, Any] | None]:
    required = {
        "method",
        "evaluator",
        "config",
        "checkpoint_step",
        "checkpoint",
        "format",
        "weights",
        "problem",
        "evaluation",
    }
    allowed = required | {"initialization", "training_gpu"}
    if not required.issubset(entry) or not set(entry).issubset(allowed):
        raise ValueError(
            f"results.{checkpoint_id} has incomplete or unsupported result metadata"
        )
    if entry["weights"] != "ema":
        raise ValueError(f"{checkpoint_id}.weights must be ema")
    checkpoint_format = str(entry["format"])
    if checkpoint_format != "raw_pytorch_state_dict":
        raise ValueError(f"{checkpoint_id} has unsupported format {checkpoint_format!r}")
    step = int(entry["checkpoint_step"])
    if step <= 0:
        raise ValueError(f"{checkpoint_id}.checkpoint_step must be positive")
    config_path = _safe_repo_path(
        entry["config"], field=f"results.{checkpoint_id}.config"
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config for {checkpoint_id}: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _assert_machine_independent(raw, location=str(config_path.relative_to(REPO_ROOT)))
    cfg = load_config(config_path)
    evaluation = cfg.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError(f"{checkpoint_id} config must resolve an evaluation preset")
    expected_method = "one_shot" if entry["method"] == "one_shot_dgm" else "iedg"
    if evaluation.get("method") != expected_method:
        raise ValueError(f"{checkpoint_id} method disagrees with its evaluation config")
    if evaluation.get("schema") not in {
        "lattice-evaluation-v1",
        "maxcut-evaluation-v1",
    }:
        raise ValueError(f"{checkpoint_id} uses an unsupported evaluation schema")
    if evaluation.get("schema") == "lattice-evaluation-v1":
        _validate_problem(result_id=checkpoint_id, entry=entry, cfg=cfg)
    else:
        problem = entry.get("problem", {})
        if problem.get("kind") != "maxcut":
            raise ValueError(f"{checkpoint_id} is not registered as Max-Cut")
        if evaluation.get("scale") != problem.get("scale"):
            raise ValueError(
                f"{checkpoint_id} scale disagrees with its evaluation config"
            )
    return config_path, cfg


def _validate_additional_checkpoint_file(
    *,
    checkpoint_id: str,
    entry: Mapping[str, Any],
    path: Path,
    cfg: dict[str, Any] | None,
) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in state.items()
    ):
        raise ValueError(f"{checkpoint_id} is not a raw inference state dict")
    assert cfg is not None
    if cfg["evaluation"]["schema"] == "maxcut-evaluation-v1":
        from CO_experiment.models.guidance import GraphGuidanceHead

        model = GraphGuidanceHead.from_config(dict(cfg["model"]))
    else:
        model = build_guidance(cfg=cfg, device=torch.device("cpu"))
    model.load_state_dict(state, strict=True)


def validate_release(
    manifest_path: str | Path = "artifacts/manifest.yaml",
    *,
    require_artifacts: bool = True,
) -> dict[str, Any]:
    path = resolve_manifest_path(manifest_path)
    global REPO_ROOT
    REPO_ROOT = repository_root_for_manifest(path)
    manifest = _load_manifest(path)
    _assert_machine_independent(manifest, location=str(path.relative_to(REPO_ROOT)))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"manifest.schema_version must be {SCHEMA_VERSION}")
    if manifest.get("method") != "IEDG":
        raise ValueError("manifest.method must be IEDG")
    results = manifest.get("results")
    if not isinstance(results, dict) or set(results) != EXPECTED_RESULTS:
        raise ValueError(f"manifest.results must contain exactly {sorted(EXPECTED_RESULTS)}")
    for result_id in results:
        load_manifest_result(path, result_id, require_checkpoint=False)
    _validate_distribution(manifest)
    additional = _additional_checkpoint_entries(manifest)
    inventory = release_filenames(manifest)
    _validate_initialization(manifest)

    artifact_root = _artifact_root(manifest)
    validated_configs: list[str] = []
    validated_files: list[str] = []
    recomputed_results: list[str] = []
    recomputed_paper_metrics: dict[str, dict[str, float]] = {}
    for result_id in FROZEN_L16_RESULTS:
        entry = results[result_id]
        cfg, _kind, lattice_size, num_states = _validate_config(result_id, entry)
        validated_configs.append(str(entry["config"]))
        if not require_artifacts:
            continue
        checkpoint = artifact_root / str(entry["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing checkpoint for {result_id}: {checkpoint}")
        _validate_checkpoint(checkpoint, cfg=cfg, result_id=result_id)
        validated_files.append(checkpoint.name)

        evaluation = entry["evaluation"]
        rollout_count = int(manifest["evaluation_protocol"]["rollout_samples"])
        reference_count = int(evaluation["reference_num_samples"])
        sample_tensors: dict[str, torch.Tensor] = {}
        for role, expected_count in (
            ("generated_samples", rollout_count),
            ("reference_samples", reference_count),
        ):
            sample_path = artifact_root / str(evaluation[role])
            if not sample_path.is_file():
                raise FileNotFoundError(f"missing {role} for {result_id}: {sample_path}")
            sample_tensors[role] = _validate_sample_file(
                sample_path,
                result_id=result_id,
                role=role,
                expected_count=expected_count,
                length=lattice_size * lattice_size,
                num_states=num_states,
            )
            validated_files.append(sample_path.name)
        metrics_path = artifact_root / str(evaluation["metrics"])
        if not metrics_path.is_file():
            raise FileNotFoundError(f"missing metrics for {result_id}: {metrics_path}")
        metrics_payload = _validate_metrics(
            metrics_path,
            manifest=manifest,
            result_id=result_id,
            entry=entry,
        )
        recomputed = _recompute_paper_metrics(
            cfg=cfg,
            problem_kind=_kind,
            generated=sample_tensors["generated_samples"],
            reference=sample_tensors["reference_samples"],
            lattice_size=lattice_size,
        )
        _validate_recomputed_metrics(
            result_id=result_id,
            recorded=metrics_payload["metrics"],
            recomputed=recomputed,
        )
        recomputed_results.append(result_id)
        recomputed_paper_metrics[result_id] = recomputed
        validated_files.append(metrics_path.name)

    additional_validated: list[str] = []
    for checkpoint_id, entry in additional.items():
        _, cfg = _validate_additional_checkpoint_metadata(checkpoint_id, entry)
        additional_validated.append(checkpoint_id)
        validated_configs.append(str(entry["config"]))
        if not require_artifacts:
            continue
        checkpoint_path = artifact_root / str(entry["checkpoint"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"missing additional checkpoint for {checkpoint_id}: {checkpoint_path}"
            )
        _validate_additional_checkpoint_file(
            checkpoint_id=checkpoint_id,
            entry=entry,
            path=checkpoint_path,
            cfg=cfg,
        )
        validated_files.append(checkpoint_path.name)

    return {
        "manifest": str(path.relative_to(REPO_ROOT)),
        "schema_version": manifest["schema_version"],
        "results_validated": len(validated_configs),
        "artifacts_required": require_artifacts,
        "artifacts_validated": len(validated_files),
        "checkpoints_validated": len(results),
        "additional_checkpoints_validated": len(additional_validated),
        "inventory_size": len(inventory),
        "metrics_recomputed": len(recomputed_results),
        "paper_metrics": recomputed_paper_metrics,
        "validated_results": sorted(results),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="artifacts/manifest.yaml")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="validate the manifest and evaluation configs without opening bundled artifacts",
    )
    args = parser.parse_args(argv)
    report = validate_release(args.manifest, require_artifacts=not args.metadata_only)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "EXPECTED_ADDITIONAL_CHECKPOINTS",
    "EXPECTED_RESULTS",
    "SCHEMA_VERSION",
    "_assert_machine_independent",
    "release_filenames",
    "validate_release",
]
