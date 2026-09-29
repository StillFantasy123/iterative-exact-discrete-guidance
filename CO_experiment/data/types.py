from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor

from CO_experiment.protocol import PROBLEM


# The released checkpoint architecture uses a three-row problem embedding;
# this evaluator activates only its Max-Cut row.
PROBLEM_TO_ID = {"mis": 0, "maxcut": 1, "bgp": 2}


@dataclass(frozen=True)
class GraphInstance:
    edge_index: Tensor
    num_nodes: int
    degrees: Tensor
    edge_weight: Tensor
    name: str
    oracle_value: float | None = None
    def __post_init__(self) -> None:
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if self.degrees.shape != (self.num_nodes,):
            raise ValueError("degrees must have shape [num_nodes]")
        if self.edge_weight.shape != (self.edge_index.shape[1],):
            raise ValueError("edge_weight must have shape [num_edges]")
        if self.edge_index.numel() and (
            int(self.edge_index.min().item()) < 0
            or int(self.edge_index.max().item()) >= self.num_nodes
        ):
            raise ValueError("edge endpoint outside node range")

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to(self, device: torch.device | str) -> "GraphInstance":
        return GraphInstance(
            edge_index=self.edge_index.to(device=device, dtype=torch.long),
            num_nodes=int(self.num_nodes),
            degrees=self.degrees.to(device=device, dtype=torch.float32),
            edge_weight=self.edge_weight.to(device=device, dtype=torch.float32),
            name=self.name,
            oracle_value=self.oracle_value,
        )


@dataclass(frozen=True)
class GraphBatch:
    edge_index: Tensor
    edge_graph_ids: Tensor
    edge_weight: Tensor
    node_mask: Tensor
    degrees: Tensor
    num_nodes: Tensor
    problem_ids: Tensor
    names: tuple[str, ...]
    oracle_values: Tensor | None = None

    def __post_init__(self) -> None:
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        if self.edge_graph_ids.shape != (self.edge_index.shape[1],):
            raise ValueError("edge_graph_ids must have shape [E]")
        if self.edge_weight.shape != (self.edge_index.shape[1],):
            raise ValueError("edge_weight must have shape [E]")
        if self.node_mask.ndim != 2 or self.degrees.shape != self.node_mask.shape:
            raise ValueError("node mask/degrees must have shape [B,Nmax]")
        if len(self.names) != self.node_mask.shape[0]:
            raise ValueError("names length must equal batch size")

    @property
    def batch_size(self) -> int:
        return int(self.node_mask.shape[0])

    @property
    def max_num_nodes(self) -> int:
        return int(self.node_mask.shape[1])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to(self, device: torch.device | str) -> "GraphBatch":
        return GraphBatch(
            edge_index=self.edge_index.to(device=device, dtype=torch.long),
            edge_graph_ids=self.edge_graph_ids.to(device=device, dtype=torch.long),
            edge_weight=self.edge_weight.to(device=device, dtype=torch.float32),
            node_mask=self.node_mask.to(device=device, dtype=torch.bool),
            degrees=self.degrees.to(device=device, dtype=torch.float32),
            num_nodes=self.num_nodes.to(device=device, dtype=torch.long),
            problem_ids=self.problem_ids.to(device=device, dtype=torch.long),
            names=self.names,
            oracle_values=(
                None
                if self.oracle_values is None
                else self.oracle_values.to(device=device, dtype=torch.float32)
            ),
        )


def _clean_edges(
    pairs: Iterable[tuple[int, int]], *, num_nodes: int
) -> list[tuple[int, int]]:
    clean: set[tuple[int, int]] = set()
    for u, v in pairs:
        u_i, v_i = int(u), int(v)
        if u_i == v_i:
            continue
        if u_i < 0 or v_i < 0 or u_i >= num_nodes or v_i >= num_nodes:
            raise ValueError("edge endpoint outside node range")
        clean.add((u_i, v_i) if u_i < v_i else (v_i, u_i))
    return sorted(clean)


def make_graph_instance(
    *,
    pairs: Iterable[tuple[int, int]],
    num_nodes: int,
    name: str,
    edge_weight: float | Sequence[float] = 1.0,
    oracle_value: float | None = None,
) -> GraphInstance:
    edges = _clean_edges(pairs, num_nodes=int(num_nodes))
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    if isinstance(edge_weight, (list, tuple)):
        weights = torch.tensor([float(v) for v in edge_weight], dtype=torch.float32)
        if weights.shape != (len(edges),):
            raise ValueError("edge_weight sequence must match cleaned edges")
    else:
        weights = torch.full((len(edges),), float(edge_weight), dtype=torch.float32)
    degrees = torch.zeros((int(num_nodes),), dtype=torch.float32)
    if edge_index.numel():
        ones = torch.ones((edge_index.shape[1],), dtype=torch.float32)
        degrees.index_add_(0, edge_index[0], ones)
        degrees.index_add_(0, edge_index[1], ones)
    return GraphInstance(
        edge_index, int(num_nodes), degrees, weights, str(name), oracle_value
    )


def collate_graphs(
    graphs: Sequence[GraphInstance], device: torch.device | str | None = None
) -> GraphBatch:
    if not graphs:
        raise ValueError("graphs must be non-empty")
    bsz = len(graphs)
    nmax = max(graph.num_nodes for graph in graphs)
    node_mask = torch.zeros((bsz, nmax), dtype=torch.bool)
    degrees = torch.zeros((bsz, nmax), dtype=torch.float32)
    num_nodes = torch.empty((bsz,), dtype=torch.long)
    names: list[str] = []
    edge_chunks: list[Tensor] = []
    edge_graph_chunks: list[Tensor] = []
    edge_weight_chunks: list[Tensor] = []
    oracle_vals: list[float] = []
    has_oracle = True
    for idx, graph in enumerate(graphs):
        n = graph.num_nodes
        node_mask[idx, :n] = True
        degrees[idx, :n] = graph.degrees
        num_nodes[idx] = n
        names.append(graph.name)
        if graph.num_edges:
            edge_chunks.append(graph.edge_index)
            edge_graph_chunks.append(
                torch.full((graph.num_edges,), idx, dtype=torch.long)
            )
            edge_weight_chunks.append(graph.edge_weight)
        has_oracle = has_oracle and graph.oracle_value is not None
        oracle_vals.append(
            float("nan") if graph.oracle_value is None else float(graph.oracle_value)
        )
    edge_index = (
        torch.cat(edge_chunks, dim=1).contiguous()
        if edge_chunks
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_graph_ids = (
        torch.cat(edge_graph_chunks).contiguous()
        if edge_graph_chunks
        else torch.empty((0,), dtype=torch.long)
    )
    edge_weight = (
        torch.cat(edge_weight_chunks).contiguous()
        if edge_weight_chunks
        else torch.empty((0,), dtype=torch.float32)
    )
    batch = GraphBatch(
        edge_index=edge_index,
        edge_graph_ids=edge_graph_ids,
        edge_weight=edge_weight,
        node_mask=node_mask,
        degrees=degrees,
        num_nodes=num_nodes,
        problem_ids=torch.full((bsz,), PROBLEM_TO_ID[PROBLEM], dtype=torch.long),
        names=tuple(names),
        oracle_values=(
            torch.tensor(oracle_vals, dtype=torch.float32) if has_oracle else None
        ),
    )
    return batch.to(device) if device is not None else batch


def repeat_graph(
    graph: GraphInstance,
    batch_size: int,
    device: torch.device | str | None = None,
) -> GraphBatch:
    return collate_graphs([graph for _ in range(int(batch_size))], device=device)


__all__ = [
    "GraphBatch",
    "GraphInstance",
    "PROBLEM_TO_ID",
    "collate_graphs",
    "make_graph_instance",
    "repeat_graph",
]
