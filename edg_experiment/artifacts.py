"""Resolve and list artifacts bundled with the evaluation release."""

from __future__ import annotations

import argparse
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


EVALUATION_FILE_KEYS = (
    "generated_samples",
    "reference_samples",
    "metrics",
)
SCOPES = ("checkpoints", "evaluation", "all")
RESULT_CONTRACT_SCHEMA = "paper_release_v4"


@dataclass(frozen=True)
class ManifestResult:
    """One manifest-owned paper result with machine-independent paths."""

    result_id: str
    repository_root: Path
    method: str
    evaluator: str
    config_path: Path
    checkpoint_path: Path
    checkpoint_step: int
    checkpoint_format: str
    weight_source: str
    problem: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    evaluation_protocol: Mapping[str, Any]
    entry: Mapping[str, Any]


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = resolve_manifest_path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Release manifest not found: {manifest_path}")
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Release manifest must be a mapping")
    return payload


def repository_root_for_manifest(manifest_path: Path) -> Path:
    """Return the source-tree or installed-data root owning a manifest."""

    for candidate in (manifest_path.parent, *manifest_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate.resolve()
    installed_root = manifest_path.parent.parent
    if (
        manifest_path.name == "manifest.yaml"
        and manifest_path.parent.name == "artifacts"
        and (installed_root / "configs").is_dir()
    ):
        return installed_root.resolve()
    raise ValueError(
        f"Could not locate the repository root above manifest {manifest_path}"
    )


def _repo_path(root: Path, value: Any, *, field: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field} must be a repository-relative path")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{field} escapes the repository")
    return resolved


def load_manifest_result(
    manifest_path: str | Path,
    result_id: str,
    *,
    require_checkpoint: bool = True,
    expected_evaluator: str | None = None,
) -> ManifestResult:
    """Resolve any paper result through the common release contract."""

    manifest = resolve_manifest_path(manifest_path)
    payload = load_manifest(manifest)
    if str(payload.get("schema_version", "")) != RESULT_CONTRACT_SCHEMA:
        raise ValueError(
            f"Manifest-evaluable results require schema_version={RESULT_CONTRACT_SCHEMA}"
        )
    results = payload.get("results")
    if not isinstance(results, Mapping) or result_id not in results:
        available = ", ".join(sorted(str(key) for key in (results or {})))
        raise KeyError(f"Unknown release result {result_id!r}; available: {available}")
    entry = results[result_id]
    if not isinstance(entry, Mapping):
        raise ValueError(f"results.{result_id} must be a mapping")

    evaluator = str(entry.get("evaluator", "")).strip()
    if not evaluator:
        raise ValueError(f"results.{result_id}.evaluator must be set")
    if expected_evaluator is not None and evaluator != expected_evaluator:
        raise ValueError(
            f"result {result_id!r} uses evaluator={evaluator!r}, expected "
            f"{expected_evaluator!r}"
        )
    method = str(entry.get("method", "")).strip()
    if not method:
        raise ValueError(f"results.{result_id}.method must be set")

    root = repository_root_for_manifest(manifest)
    artifact_root = _repo_path(
        root, payload.get("artifact_root", "artifacts"), field="artifact_root"
    )
    checkpoint_name = Path(str(entry.get("checkpoint", "")))
    if len(checkpoint_name.parts) != 1 or not checkpoint_name.name:
        raise ValueError(f"results.{result_id}.checkpoint must be a flat filename")
    checkpoint_path = (artifact_root / checkpoint_name).resolve()
    if require_checkpoint and not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Release checkpoint for {result_id} is not available: {checkpoint_path}"
        )

    config_path = _repo_path(
        root, entry.get("config", ""), field=f"results.{result_id}.config"
    )
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Release config for {result_id} is missing: {config_path}"
        )

    checkpoint_defaults = payload.get("checkpoint_format", {})
    if not isinstance(checkpoint_defaults, Mapping):
        raise ValueError("manifest.checkpoint_format must be a mapping")
    checkpoint_format = str(
        entry.get("format", checkpoint_defaults.get("payload", ""))
    ).strip()
    weight_source = str(
        entry.get("weights", checkpoint_defaults.get("weights", ""))
    ).lower()
    if not checkpoint_format or not weight_source:
        raise ValueError(
            f"results.{result_id} must resolve checkpoint format and weights"
        )

    evaluation = entry.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError(f"results.{result_id}.evaluation must be a mapping")
    evaluation_protocol = payload.get("evaluation_protocol", {})
    if not isinstance(evaluation_protocol, Mapping):
        raise ValueError("manifest.evaluation_protocol must be a mapping")
    checkpoint_step = int(entry.get("checkpoint_step", 0))
    if checkpoint_step <= 0:
        raise ValueError(f"results.{result_id}.checkpoint_step must be positive")

    problem = entry.get("problem", {})
    if not isinstance(problem, Mapping):
        raise ValueError(f"results.{result_id}.problem must be a mapping")
    return ManifestResult(
        result_id=str(result_id),
        repository_root=root,
        method=method,
        evaluator=evaluator,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        checkpoint_step=checkpoint_step,
        checkpoint_format=checkpoint_format,
        weight_source=weight_source,
        problem=dict(problem),
        evaluation=dict(evaluation),
        evaluation_protocol=dict(evaluation_protocol),
        entry=dict(entry),
    )


def release_filenames(
    manifest: Mapping[str, Any], *, scope: str = "all"
) -> list[str]:
    """Return the unique flat filenames owned by one manifest."""

    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {', '.join(SCOPES)}")
    names: list[str] = []
    results = manifest.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("manifest.results must be a mapping")
    for result_id, raw_entry in results.items():
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"results.{result_id} must be a mapping")
        if scope in {"checkpoints", "all"}:
            names.append(str(raw_entry.get("checkpoint", "")))
        if scope in {"evaluation", "all"}:
            evaluation = raw_entry.get("evaluation")
            if not isinstance(evaluation, Mapping):
                raise ValueError(f"results.{result_id}.evaluation must be a mapping")
            names.extend(
                str(evaluation[key])
                for key in EVALUATION_FILE_KEYS
                if str(evaluation.get(key, "")).strip()
            )

    if any(not name for name in names):
        raise ValueError("Release inventory contains an empty filename")
    for name in names:
        value = Path(name)
        if value.is_absolute() or len(value.parts) != 1 or value.name != name:
            raise ValueError(f"Release artifacts must use flat filenames: {name}")
    return list(dict.fromkeys(names))


def _default_manifest_path() -> Path:
    candidates = (
        Path.cwd() / "artifacts/manifest.yaml",
        Path(__file__).resolve().parents[1] / "artifacts/manifest.yaml",
        Path(sysconfig.get_path("data")) / "share/iedg/artifacts/manifest.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_manifest_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if str(candidate) == "artifacts/manifest.yaml":
        fallback = _default_manifest_path()
        if fallback.is_file():
            return fallback
    raise FileNotFoundError(f"Release manifest not found: {candidate.resolve()}")


def _add_manifest_and_scope(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--scope", choices=SCOPES, default="all")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List artifacts declared by the bundled release manifest."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    list_parser = commands.add_parser("list", help="List release filenames")
    _add_manifest_and_scope(list_parser)

    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_path = (
        _default_manifest_path()
        if args.manifest is None
        else args.manifest.expanduser().resolve()
    )
    manifest = load_manifest(manifest_path)
    print("\n".join(release_filenames(manifest, scope=args.scope)))


if __name__ == "__main__":
    main()


__all__ = [
    "EVALUATION_FILE_KEYS",
    "ManifestResult",
    "RESULT_CONTRACT_SCHEMA",
    "load_manifest",
    "load_manifest_result",
    "main",
    "parse_args",
    "release_filenames",
    "repository_root_for_manifest",
    "resolve_manifest_path",
]
