from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

from edg_experiment.dfm.schedules import KappaSchedule

from CO_experiment.data.types import GraphBatch, GraphInstance, repeat_graph
from CO_experiment.protocol import GUIDANCE_STRENGTH
from CO_experiment.sampling.process import analytic_uniform_replace_posterior, sample_probs, teacher_posterior


@torch.no_grad()
def guided_direct_q_ctmc_rollout(
    *,
    graph: GraphBatch,
    guidance_model,
    schedule: KappaSchedule,
    device: torch.device,
    num_steps: int,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive")
    graph = graph.to(device)
    bsz = graph.batch_size
    nmax = graph.max_num_nodes
    x_t = torch.randint(0, 2, (bsz, nmax), device=device, dtype=torch.long)
    x_t = x_t * graph.node_mask.to(torch.long)
    h_step = 1.0 / float(num_steps)
    avg_rates: list[float] = []
    avg_jumps: list[float] = []
    state_change_fracs: list[float] = []
    avg_entropy: list[float] = []
    step_iter = tqdm(
        range(int(num_steps)),
        desc=progress_desc or "maxcut direct-q ctmc",
        disable=not show_progress,
        leave=False,
    )
    mask_f = graph.node_mask.to(torch.float32)
    for step_idx in step_iter:
        x_prev = x_t
        t_cur = torch.full((bsz,), step_idx / float(num_steps), device=device, dtype=torch.float32)
        base = analytic_uniform_replace_posterior(x_prev, t_cur, schedule=schedule)
        _, residual_log_h = guidance_model.forward_with_log_h(x_prev, t_cur, graph)
        q_tilde = teacher_posterior(base, residual_log_h).to(torch.float32)
        one_hot_cur = F.one_hot(x_prev, num_classes=2).to(torch.float32)
        coeff = schedule.rate_coeff(t_cur).to(torch.float32).view(-1, 1, 1)
        jump_rates = coeff * q_tilde * (1.0 - one_hot_cur) * mask_f.unsqueeze(-1)
        intensity = jump_rates.sum(dim=-1)
        jump_prob = torch.clamp(1.0 - torch.exp(-h_step * intensity), 0.0, 1.0)
        jump_mask = torch.rand_like(jump_prob) < jump_prob
        jump_tokens = sample_probs(jump_rates / intensity.unsqueeze(-1).clamp_min(1.0e-12))
        x_t = torch.where(jump_mask, jump_tokens, x_prev)
        x_t = x_t * graph.node_mask.to(torch.long)
        entropy = -(q_tilde.clamp_min(1.0e-12) * q_tilde.clamp_min(1.0e-12).log()).sum(dim=-1)
        denom = mask_f.sum().clamp_min(1.0)
        avg_rates.append(float((intensity * mask_f).sum().item() / denom.item()))
        avg_jumps.append(float((jump_mask.to(torch.float32) * mask_f).sum().item() / denom.item()))
        state_change_fracs.append(float((x_t.ne(x_prev).to(torch.float32) * mask_f).sum().item() / denom.item()))
        avg_entropy.append(float((entropy * mask_f).sum().item() / denom.item()))
    return x_t, {
        "sampler_mode": "direct_q_ctmc",
        "mean_jump_rate": float(sum(avg_rates) / max(len(avg_rates), 1)),
        "mean_jump_fraction": float(sum(avg_jumps) / max(len(avg_jumps), 1)),
        "mean_state_change_fraction": float(sum(state_change_fracs) / max(len(state_change_fracs), 1)),
        "mean_guided_posterior_entropy": float(sum(avg_entropy) / max(len(avg_entropy), 1)),
        "guidance_strength": GUIDANCE_STRENGTH,
        "num_steps": int(num_steps),
    }


@torch.no_grad()
def collect_samples_for_graph(
    *,
    graph: GraphInstance,
    guidance_model,
    schedule: KappaSchedule,
    device: torch.device,
    total_samples: int,
    rollout_cfg: dict[str, Any],
    show_progress: bool = False,
    desc: str = "maxcut sampling",
) -> tuple[Tensor, dict[str, Any]]:
    batch_size = int(rollout_cfg.get("batch_size", 64))
    num_steps = int(rollout_cfg.get("num_steps", 64))
    chunks: list[Tensor] = []
    dbg_sums: dict[str, float] = {}
    dbg_counts: dict[str, int] = {}
    remaining = int(total_samples)
    n_chunks = (remaining + batch_size - 1) // batch_size
    iterator = tqdm(range(n_chunks), desc=desc, disable=not show_progress, leave=False)
    for _ in iterator:
        bsz = min(batch_size, remaining)
        if bsz <= 0:
            break
        samples, dbg = guided_direct_q_ctmc_rollout(
            graph=repeat_graph(graph, bsz, device=device),
            guidance_model=guidance_model,
            schedule=schedule,
            device=device,
            num_steps=num_steps,
        )
        chunks.append(samples[:, : graph.num_nodes].detach())
        for key, value in dbg.items():
            if isinstance(value, (int, float)):
                dbg_sums[key] = dbg_sums.get(key, 0.0) + float(value)
                dbg_counts[key] = dbg_counts.get(key, 0) + 1
        remaining -= bsz
        if show_progress:
            iterator.set_postfix({"done": f"{total_samples - remaining}/{total_samples}"})
    out = torch.cat(chunks, dim=0) if chunks else torch.empty((0, graph.num_nodes), device=device, dtype=torch.long)
    debug = {key: float(dbg_sums[key] / max(dbg_counts.get(key, 1), 1)) for key in dbg_sums}
    return out, debug


__all__ = ["collect_samples_for_graph", "guided_direct_q_ctmc_rollout"]
