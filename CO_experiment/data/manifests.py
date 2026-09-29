"""Read the fixed Max-Cut test graph manifests bundled with the release."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from CO_experiment.data.types import GraphInstance, make_graph_instance


def _graph_from_record(record: dict) -> GraphInstance:
    pairs = record.get("edges", record.get("edge_index"))
    if pairs is None:
        raise ValueError("graph manifest record is missing edges")
    oracle = record.get("oracle_value", record.get("objective_value"))
    return make_graph_instance(
        pairs=[(int(u), int(v)) for u, v in pairs],
        num_nodes=int(record["num_nodes"]),
        name=str(record.get("graph_name", record.get("name", ""))),
        edge_weight=record.get("edge_weight", 1.0),
        oracle_value=None if oracle is None else float(oracle),
    )


def load_graph_manifest(
    path: str | Path, *, split: str | None = None, problem: str | None = None
) -> list[GraphInstance]:
    if problem not in (None, "maxcut"):
        raise ValueError(f"unsupported problem: {problem}")
    source = Path(path).expanduser()
    opener = gzip.open if source.suffix == ".gz" else Path.open
    with opener(source, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        if str(payload.get("problem", "maxcut")) != "maxcut":
            raise ValueError("graph manifest is not a Max-Cut dataset")
        records = payload.get("records", [])
    else:
        records = payload
    if not isinstance(records, list):
        raise ValueError("graph manifest records must be a list")
    selected = [
        record
        for record in records
        if split is None or str(record.get("split", split)).lower() == split.lower()
    ]
    return [_graph_from_record(record) for record in selected]


__all__ = ["load_graph_manifest"]
