from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from CO_experiment.data.types import GraphBatch, PROBLEM_TO_ID


def sinusoidal_time_embedding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    if dim <= 0:
        raise ValueError("time embedding dim must be positive")
    half = dim // 2
    if half == 0:
        return t.to(torch.float32).unsqueeze(-1)
    freq = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = t.to(torch.float32).unsqueeze(-1) * freq.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class _EdgeAwareDNFSBlock(nn.Module):
    """Local graph attention followed by a gated graph-level context update."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.attn_norm = nn.LayerNorm(self.hidden_dim)
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.global_norm = nn.LayerNorm(self.hidden_dim)
        self.global_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.global_gate = nn.Parameter(torch.tensor(-2.0))
        self.ff_norm = nn.LayerNorm(self.hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(self.hidden_dim, 4 * self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
        )
        self.dropout = nn.Dropout(0.0)

    @staticmethod
    def local_attention_mask(graph: GraphBatch) -> Tensor:
        bsz = graph.batch_size
        nmax = graph.max_num_nodes
        valid = graph.node_mask
        mask = torch.zeros((bsz, nmax, nmax), device=valid.device, dtype=torch.bool)
        if graph.num_edges > 0:
            src, dst = graph.edge_index
            gids = graph.edge_graph_ids
            mask[gids, src, dst] = True
            mask[gids, dst, src] = True
        eye = torch.eye(nmax, device=valid.device, dtype=torch.bool).unsqueeze(0)
        mask = mask | (eye & valid.unsqueeze(1) & valid.unsqueeze(2))
        return mask & valid.unsqueeze(1) & valid.unsqueeze(2)

    def _local_attention(self, x: Tensor, graph: GraphBatch) -> Tensor:
        bsz, nmax, _ = x.shape
        z = self.attn_norm(x)
        q = self.q_proj(z).view(bsz, nmax, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(z).view(bsz, nmax, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(z).view(bsz, nmax, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(float(self.head_dim))
        mask = self.local_attention_mask(graph).unsqueeze(1)
        scores = scores.masked_fill(~mask, -1.0e4)
        attn = torch.softmax(scores, dim=-1) * mask.to(scores.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        out = torch.einsum("bhij,bhjd->bhid", attn, v).transpose(1, 2).reshape(
            bsz, nmax, self.hidden_dim
        )
        return self.out_proj(out)

    def forward(self, x: Tensor, graph: GraphBatch) -> Tensor:
        mask_f = graph.node_mask.unsqueeze(-1).to(x.dtype)
        x = x + self.dropout(self._local_attention(x, graph))
        pooled = (self.global_norm(x) * mask_f).sum(dim=1) / graph.num_nodes.to(
            x.dtype
        ).unsqueeze(-1).clamp_min(1.0)
        global_update = self.global_mlp(pooled).unsqueeze(1)
        x = x + torch.sigmoid(self.global_gate) * self.dropout(global_update)
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x * mask_f


class DNFSLeGFEdgeAwareGuidanceHead(nn.Module):
    """The single checkpoint-compatible MaxCut guidance network."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        time_embed_dim: int = 32,
        problem_embed_dim: int = 8,
        log_h_clip_min: float = -30.0,
        log_h_clip_max: float = 30.0,
    ) -> None:
        super().__init__()
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if float(log_h_clip_min) > float(log_h_clip_max):
            raise ValueError("log_h_clip_min must be <= log_h_clip_max")
        self.num_states = 2
        self.time_embed_dim = int(time_embed_dim)
        self.log_h_clip_min = float(log_h_clip_min)
        self.log_h_clip_max = float(log_h_clip_max)
        self.absolute_scale_enabled = True

        # Keep three rows even though only MaxCut is supported. Formal checkpoints
        # were trained with PROBLEM_TO_ID={mis:0,maxcut:1,bgp:2}; changing this
        # tensor shape would break strict state-dict loading.
        self.problem_embed = nn.Embedding(int(max(PROBLEM_TO_ID.values()) + 1), int(problem_embed_dim))
        input_dim = 3 + self.time_embed_dim + int(problem_embed_dim)
        self.input_proj = nn.Linear(input_dim, int(hidden_dim))
        self.blocks = nn.ModuleList(
            [_EdgeAwareDNFSBlock(int(hidden_dim), int(num_heads)) for _ in range(int(num_layers))]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 1),
        )
        final = self.head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.scale_head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 1),
        )
        scale_final = self.scale_head[-1]
        nn.init.zeros_(scale_final.weight)
        nn.init.zeros_(scale_final.bias)
        self.log_h_source = "dnfs_legf_edgeaware"

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "DNFSLeGFEdgeAwareGuidanceHead":
        return cls(
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            num_layers=int(cfg.get("num_layers", 3)),
            num_heads=int(cfg.get("num_heads", 4)),
            time_embed_dim=int(cfg.get("time_embed_dim", 32)),
            problem_embed_dim=int(cfg.get("problem_embed_dim", 8)),
            log_h_clip_min=float(cfg.get("log_h_clip_min", -30.0)),
            log_h_clip_max=float(cfg.get("log_h_clip_max", 30.0)),
        )

    @staticmethod
    def _weighted_local_invariants(xt: Tensor, graph: GraphBatch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mask_f = graph.node_mask.to(torch.float32)
        spin = (2.0 * xt.to(torch.float32) - 1.0) * mask_f
        bsz, nmax = xt.shape
        flat_spin = spin.reshape(bsz * nmax)
        signed_sum = torch.zeros_like(flat_spin)
        weighted_degree = torch.zeros_like(flat_spin)
        if graph.num_edges > 0:
            src, dst = graph.edge_index
            gids = graph.edge_graph_ids
            src_flat = gids * nmax + src
            dst_flat = gids * nmax + dst
            weight = graph.edge_weight.to(spin.dtype)
            signed_sum.index_add_(0, src_flat, weight * flat_spin.index_select(0, dst_flat))
            signed_sum.index_add_(0, dst_flat, weight * flat_spin.index_select(0, src_flat))
            abs_weight = weight.abs()
            weighted_degree.index_add_(0, src_flat, abs_weight)
            weighted_degree.index_add_(0, dst_flat, abs_weight)
        signed_sum = signed_sum.view(bsz, nmax)
        weighted_degree = weighted_degree.view(bsz, nmax)
        neighbor_spin = signed_sum / weighted_degree.clamp_min(1.0e-12)
        agreement = spin * neighbor_spin
        local_cut_fraction = 0.5 * (1.0 - agreement)
        local_cut_fraction = torch.where(
            weighted_degree > 0.0,
            local_cut_fraction,
            torch.zeros_like(local_cut_fraction),
        )
        mean_degree = (weighted_degree * mask_f).sum(dim=1, keepdim=True) / graph.num_nodes.to(
            weighted_degree.dtype
        ).unsqueeze(-1).clamp_min(1.0)
        degree_scaled = weighted_degree / mean_degree.clamp_min(1.0e-12)
        return spin, degree_scaled * mask_f, agreement * mask_f, local_cut_fraction * mask_f

    def _node_features(self, xt: Tensor, t: Tensor, graph: GraphBatch) -> tuple[Tensor, Tensor]:
        if xt.shape != graph.node_mask.shape:
            raise ValueError("xt must match graph batch node shape")
        if t.ndim != 1 or t.shape[0] != xt.shape[0]:
            raise ValueError("t must be [B] and match xt")
        if not bool(graph.problem_ids.eq(PROBLEM_TO_ID["maxcut"]).all()):
            raise ValueError("the CO guidance network only supports MaxCut batches")
        spin, degree, agreement, cut_fraction = self._weighted_local_invariants(xt, graph)
        t_map = sinusoidal_time_embedding(t, self.time_embed_dim).unsqueeze(1).expand(
            -1, graph.max_num_nodes, -1
        )
        problem = self.problem_embed(graph.problem_ids.to(xt.device)).unsqueeze(1).expand(
            -1, graph.max_num_nodes, -1
        )
        features = torch.cat(
            [degree.unsqueeze(-1), agreement.unsqueeze(-1), cut_fraction.unsqueeze(-1), t_map, problem],
            dim=-1,
        )
        return features * graph.node_mask.unsqueeze(-1).to(features.dtype), spin

    def forward_with_log_h(self, xt: Tensor, t: Tensor, graph: GraphBatch) -> tuple[Tensor, Tensor]:
        graph = graph.to(xt.device)
        features, spin = self._node_features(xt, t, graph)
        h = F.silu(self.input_proj(features)) * graph.node_mask.unsqueeze(-1).to(features.dtype)
        for block in self.blocks:
            h = block(h, graph)
        delta = self.head(h).squeeze(-1) * spin
        scale = self.scale_head(h).squeeze(-1)
        log_h = torch.stack([scale - 0.5 * delta, scale + 0.5 * delta], dim=-1)
        log_h = torch.clamp(log_h, min=self.log_h_clip_min, max=self.log_h_clip_max)
        log_h = log_h * graph.node_mask.unsqueeze(-1).to(log_h.dtype)
        return torch.exp(log_h), log_h

    def forward(self, xt: Tensor, t: Tensor, graph: GraphBatch) -> Tensor:
        h, _ = self.forward_with_log_h(xt, t, graph)
        return h


# Public name used by the release evaluator.
GraphGuidanceHead = DNFSLeGFEdgeAwareGuidanceHead

__all__ = ["DNFSLeGFEdgeAwareGuidanceHead", "GraphGuidanceHead", "sinusoidal_time_embedding"]
