from __future__ import annotations

import math
from typing import Any, Sequence

from CO_experiment.data.types import GraphInstance


def require_embedded_oracles(
    graphs: Sequence[GraphInstance], *, split: str
) -> tuple[list[GraphInstance], dict[str, Any]]:
    """Validate the frozen exact objectives embedded in a graph manifest.

    Training and final evaluation never solve graphs or consult a mutable cache.
    Gurobi is an offline manifest-construction dependency only.
    """

    missing = [
        graph.name
        for graph in graphs
        if graph.oracle_value is None or not math.isfinite(float(graph.oracle_value))
    ]
    if missing:
        raise ValueError(
            f"{split} manifest lacks embedded certified optima for "
            f"{len(missing)} graph(s), first: {', '.join(missing[:5])}"
        )
    return list(graphs), {
        "source": "embedded_certified_manifest",
        "split": str(split),
        "all_available": True,
        "embedded_oracle_count": len(graphs),
    }


__all__ = ["require_embedded_oracles"]
