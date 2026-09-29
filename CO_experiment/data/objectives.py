from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from CO_experiment.data.types import GraphBatch


def _validate_x(x: Tensor, graph: GraphBatch) -> Tensor:
    if x.ndim != 2 or x.shape != graph.node_mask.shape:
        raise ValueError("x must match the graph node mask [B,Nmax]")
    if bool(((x < 0) | (x >= 2)).any()):
        raise ValueError("MaxCut states must be binary")
    return x.to(torch.long)


def maxcut_cut_values(x: Tensor, graph: GraphBatch) -> Tensor:
    x = _validate_x(x, graph)
    out = torch.zeros((graph.batch_size,), device=x.device, dtype=torch.float32)
    if graph.num_edges <= 0:
        return out
    src, dst = graph.edge_index
    gids = graph.edge_graph_ids
    cut = x[gids, src].ne(x[gids, dst]).to(torch.float32) * graph.edge_weight.to(x.device)
    out.index_add_(0, gids, cut)
    return out


def co_cost(x: Tensor, graph: GraphBatch) -> Tensor:
    return -maxcut_cut_values(x, graph)


def summarize_samples(x: Tensor, graph: GraphBatch) -> dict[str, Any]:
    cut_values = maxcut_cut_values(x, graph)
    cost = -cut_values
    best_idx = int(torch.argmax(cut_values).item()) if cut_values.numel() else 0
    top_k = min(8, int(cut_values.numel()))
    out: dict[str, Any] = {
        "sample_count": int(x.shape[0]),
        "energy_cost_mean": float(cost.mean().item()),
        "best_energy_cost": float(cost[best_idx].item()),
        "score_mean": float(cut_values.mean().item()),
        "best_score": float(cut_values[best_idx].item()),
        "best_sample_index": best_idx,
        "cut_mean": float(cut_values.mean().item()),
        "best_cut": float(cut_values.max().item()),
        "cut_q99": float(torch.quantile(cut_values, 0.99).item()),
        "top8_cut_mean": float(torch.topk(cut_values, k=top_k).values.mean().item()),
    }
    if graph.oracle_values is None:
        out["oracle_available"] = False
        return out
    oracle = graph.oracle_values.to(x.device)
    ratio = cut_values / oracle.clamp_min(1.0e-12)
    out.update(
        {
            "oracle_available": True,
            "oracle_score": float(oracle.mean().item()),
            "sample_approx_ratio_mean": float(ratio.mean().item()),
            "best_approx_ratio": float(ratio.max().item()),
            "sample_approx_ratio_q99": float(torch.quantile(ratio.to(torch.float32), 0.99).item()),
            "top8_approx_ratio_mean": float(torch.topk(ratio, k=top_k).values.mean().item()),
            "sample_drop_mean": float((1.0 - ratio).mean().item()),
            "best_drop": float(1.0 - ratio.max().item()),
        }
    )
    return out


__all__ = ["co_cost", "maxcut_cut_values", "summarize_samples"]
