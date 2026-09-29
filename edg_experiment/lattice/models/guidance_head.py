from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from edg_experiment.dfm.schedules import KappaSchedule
from edg_experiment.lattice.models.backbone_factory import build_backbone


_ALLOWED_PRECONDITIONING_KINDS = {
    "ising_bp_4nbr",
    "potts_bp_4nbr",
}


class GuidanceHead(nn.Module):
    def __init__(
        self,
        cfg: Dict[str, Any],
        length: int,
        num_states: int,
        input_vocab_size: int | None = None,
        forward_kind: str = "uniform_replace",
        schedule: KappaSchedule | None = None,
        ising_cfg: Dict[str, Any] | None = None,
        problem_kind: str = "ising",
    ) -> None:
        super().__init__()
        if input_vocab_size is None:
            input_vocab_size = int(num_states)
        self.backbone, d_model = build_backbone(
            model_cfg=cfg,
            vocab_size=int(input_vocab_size),
            length=length,
        )
        self.head = nn.Linear(d_model, num_states)
        self.input_vocab_size = int(input_vocab_size)
        self.num_states = int(num_states)
        self.forward_kind = str(forward_kind).lower()
        self.guidance_parametrization = str(cfg.get("guidance_parametrization", "log_h")).lower()
        if self.guidance_parametrization != "log_h":
            raise ValueError("released checkpoints require guidance_parametrization=log_h")
        self.log_h_clip_min = float(cfg.get("log_h_clip_min", -20.0))
        self.log_h_clip_max = float(cfg.get("log_h_clip_max", 20.0))
        self.center_log_h_over_states = bool(cfg.get("center_log_h_over_states", False))
        if self.center_log_h_over_states:
            raise ValueError("released checkpoints do not center log-h over states")
        if self.log_h_clip_min > self.log_h_clip_max:
            raise ValueError("model.log_h_clip_min must be <= model.log_h_clip_max")

        self.problem_kind = str(problem_kind).lower()
        pre_cfg = dict(cfg.get("preconditioning", {}))
        self.preconditioning_enabled = bool(pre_cfg.get("enabled", False))
        self.preconditioning_kind = str(
            pre_cfg.get("kind", f"{self.problem_kind}_bp_4nbr")
        ).lower()
        self.preconditioning_boundary = str(pre_cfg.get("boundary", "periodic")).lower()
        self.ratio_scale = float(pre_cfg.get("ratio_scale", 1.0))
        self.log_prob_eps = float(pre_cfg.get("log_prob_eps", 1.0e-12))
        self.residual_zero_init = bool(pre_cfg.get("residual_zero_init", False))
        self.num_message_passing_steps = int(pre_cfg.get("num_message_passing_steps", 2))
        self.message_damping = float(pre_cfg.get("message_damping", 1.0))
        preconditioning_base_beta_raw = pre_cfg.get("base_beta", pre_cfg.get("beta_override", None))
        self.schedule = schedule
        self.log_h_source = "raw"
        self._beta = 0.0
        self._J = 1.0
        self._h = 0.0
        self._coef_J = 0.0
        self._coef_h = 0.0
        self._grid = int(math.sqrt(length))
        self.preconditioning_base_beta: float | None = None

        if self.preconditioning_enabled:
            if self.preconditioning_kind not in _ALLOWED_PRECONDITIONING_KINDS:
                raise ValueError(
                    "model.preconditioning.kind must be one of "
                    f"{sorted(_ALLOWED_PRECONDITIONING_KINDS)}, got {self.preconditioning_kind}"
                )
            if self.preconditioning_boundary != "periodic":
                raise ValueError("model.preconditioning.boundary must be periodic")
            if not math.isfinite(self.ratio_scale):
                raise ValueError("model.preconditioning.ratio_scale must be finite")
            if self.log_prob_eps <= 0.0:
                raise ValueError("model.preconditioning.log_prob_eps must be > 0")
            if self.num_message_passing_steps < 1:
                raise ValueError("model.preconditioning.num_message_passing_steps must be >= 1")
            if not math.isfinite(self.message_damping) or not (0.0 < self.message_damping <= 1.0):
                raise ValueError("model.preconditioning.message_damping must be in (0, 1]")
            if self.guidance_parametrization != "log_h":
                raise ValueError(
                    "model.preconditioning.enabled=true requires model.guidance_parametrization=log_h"
                )
            if self.schedule is None:
                raise ValueError("KappaSchedule must be provided when preconditioning is enabled")
            if ising_cfg is None:
                raise ValueError("ising_cfg must be provided when preconditioning is enabled")
            if self._grid * self._grid != length:
                raise ValueError("length must be a perfect square for 4-neighbor BP preconditioning")
            if self.preconditioning_kind == "ising_bp_4nbr":
                if self.problem_kind != "ising":
                    raise ValueError(f"{self.preconditioning_kind} requires problem.kind=ising")
                if int(num_states) != 2:
                    raise ValueError(f"{self.preconditioning_kind} requires num_states=2")
            if self.preconditioning_kind == "potts_bp_4nbr":
                if self.problem_kind != "potts":
                    raise ValueError(f"{self.preconditioning_kind} requires problem.kind=potts")
                if int(num_states) < 2:
                    raise ValueError(f"{self.preconditioning_kind} requires num_states>=2")

            self._beta = (
                float(ising_cfg["beta"])
                if preconditioning_base_beta_raw in (None, "")
                else float(preconditioning_base_beta_raw)
            )
            self._J = float(ising_cfg["J"])
            self._h = float(ising_cfg.get("h", 0.0))
            self.preconditioning_base_beta = float(self._beta)
            self._coef_J = self.ratio_scale * self._beta * self._J
            self._coef_h = self.ratio_scale * self._beta * self._h
            self.log_h_source = self.preconditioning_kind
            self._init_lattice_buffers(length=length)

        if self.residual_zero_init:
            # The final linear layer produces the additive residual in log-h space.
            # Zero init gives:
            # - with preconditioning: Delta = 0, so the model starts from the baseline log-h
            # - without preconditioning and guidance_parametrization=log_h: log_h = 0, so h = 1
            nn.init.zeros_(self.head.weight)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)

    def _forward_raw(self, xt: Tensor, t: Tensor) -> Tensor:
        return self.head(self.backbone(xt, t))

    def _init_lattice_buffers(self, *, length: int) -> None:
        L = self._grid
        if L * L != length:
            raise ValueError("length must be a perfect square for 4-neighbor BP preconditioning")
        neighbor_index = torch.empty((length, 4), dtype=torch.long)
        for r in range(L):
            for c in range(L):
                idx = r * L + c
                neighbor_index[idx] = torch.tensor(
                    [
                        ((r - 1) % L) * L + c,
                        ((r + 1) % L) * L + c,
                        r * L + ((c - 1) % L),
                        r * L + ((c + 1) % L),
                    ],
                    dtype=torch.long,
                )
        self.register_buffer("_neighbor_index", neighbor_index, persistent=False)

        # Direction order is [up, down, left, right].  A message stored at
        # [site, direction] travels from ``site`` to that neighbor; looking up
        # the reverse direction therefore recovers incoming neighbor messages.
        self.register_buffer(
            "_reverse_direction",
            torch.tensor([1, 0, 3, 2], dtype=torch.long),
            persistent=False,
        )

    def _check_preconditioning_inputs(self, xt: Tensor, t: Tensor) -> None:
        if self.schedule is None:
            raise ValueError("KappaSchedule is not configured for preconditioning")
        if xt.ndim != 2:
            raise ValueError(f"xt must be [B,D], got {tuple(xt.shape)}")
        if t.ndim != 1:
            raise ValueError(f"t must be [B], got {tuple(t.shape)}")
        if xt.shape[0] != t.shape[0]:
            raise ValueError("Batch size mismatch between xt and t")
        if xt.shape[1] != self._grid * self._grid:
            raise ValueError(f"length mismatch: expected {self._grid * self._grid}, got {xt.shape[1]}")

    def _compute_p0_binary(self, xt: Tensor, t: Tensor) -> Tensor:
        self._check_preconditioning_inputs(xt=xt, t=t)
        kappa_t = self.schedule.kappa(t).to(torch.float32).view(-1, 1, 1)
        one_hot = F.one_hot(xt.to(torch.long), num_classes=2).to(torch.float32)
        p0 = one_hot * kappa_t + 0.5 * (1.0 - kappa_t)
        p0 = p0.clamp_min(self.log_prob_eps)
        p0 = p0 / p0.sum(dim=-1, keepdim=True).clamp_min(self.log_prob_eps)
        return p0

    def _compute_p0_q(self, xt: Tensor, t: Tensor) -> Tensor:
        self._check_preconditioning_inputs(xt=xt, t=t)
        kappa_t = self.schedule.kappa(t).to(torch.float32).view(-1, 1, 1)
        one_hot = F.one_hot(xt.to(torch.long), num_classes=self.num_states).to(torch.float32)
        p0 = one_hot * kappa_t + (1.0 - kappa_t) / float(self.num_states)
        p0 = p0.clamp_min(self.log_prob_eps)
        return p0 / p0.sum(dim=-1, keepdim=True).clamp_min(self.log_prob_eps)

    def _baseline_log_h(self, xt: Tensor, t: Tensor) -> Tensor:
        if not self.preconditioning_enabled:
            raise ValueError("baseline log-h is only available when preconditioning is enabled")
        if self.preconditioning_kind == "ising_bp_4nbr":
            return self._ising_bp_log_h(xt=xt, t=t)
        if self.preconditioning_kind == "potts_bp_4nbr":
            return self._potts_bp_log_h(xt=xt, t=t)
        raise ValueError(f"Unsupported preconditioning kind: {self.preconditioning_kind}")

    def _ising_edge_factor(self, *, like: Tensor) -> Tensor:
        spin = like.new_tensor([-1.0, 1.0])
        return torch.exp(self._coef_J * spin.view(2, 1) * spin.view(1, 2))

    def _ising_bp_log_h(self, xt: Tensor, t: Tensor) -> Tensor:
        """Finite-depth synchronous cavity BP for binary Ising interactions.

        Messages use normalized cavity beliefs, fixing the otherwise arbitrary
        message scale. Additional steps propagate evidence from more distant
        lattice shells.
        """
        p0 = self._compute_p0_binary(xt=xt, t=t)
        log_p0 = torch.log(p0.clamp_min(self.log_prob_eps))
        spin = p0.new_tensor([-1.0, 1.0]).view(1, 1, 2)
        site_logits = log_p0 + self._coef_h * spin
        edge_factor = self._ising_edge_factor(like=p0)
        batch_size, length, _ = p0.shape
        log_messages = p0.new_zeros((batch_size, length, 4, 2))

        for _ in range(self.num_message_passing_steps):
            incoming = log_messages[:, self._neighbor_index, self._reverse_direction]
            incoming_sum = incoming.sum(dim=2, keepdim=True)
            cavity_logits = site_logits.unsqueeze(2) + incoming_sum - incoming
            cavity_probs = torch.softmax(cavity_logits, dim=-1)
            candidate = torch.log(
                torch.einsum("bdnk,ak->bdna", cavity_probs, edge_factor).clamp_min(
                    self.log_prob_eps
                )
            )
            if self.message_damping < 1.0:
                log_messages = (
                    (1.0 - self.message_damping) * log_messages
                    + self.message_damping * candidate
                )
            else:
                log_messages = candidate

        final_incoming = log_messages[:, self._neighbor_index, self._reverse_direction]
        return self._coef_h * spin + final_incoming.sum(dim=2)

    def _potts_bp_log_h(self, xt: Tensor, t: Tensor) -> Tensor:
        """Return a finite-depth cavity-BP approximation to the Potts log-h.

        Messages use a fixed, physically meaningful gauge: the cavity belief is
        normalized before applying the Potts edge factor. A one-step update is
        the first finite-depth cavity approximation

            log m_{j->i}(a) = log[1 + (exp(rho beta J)-1) p0_j(a)],

        while later updates propagate evidence from successively more distant
        shells without accumulating arbitrary message scale factors.
        """

        p0 = self._compute_p0_q(xt=xt, t=t)
        log_p0 = torch.log(p0.clamp_min(self.log_prob_eps))
        batch_size, length, num_states = p0.shape
        log_messages = p0.new_zeros((batch_size, length, 4, num_states))
        interaction = float(math.expm1(self._coef_J))

        for _ in range(self.num_message_passing_steps):
            # incoming[:, i, d] is the message neighbor(i,d) -> i.
            incoming = log_messages[:, self._neighbor_index, self._reverse_direction]
            incoming_sum = incoming.sum(dim=2, keepdim=True)
            cavity_logits = log_p0.unsqueeze(2) + incoming_sum - incoming
            cavity_probs = torch.softmax(cavity_logits, dim=-1)
            candidate = torch.log1p(interaction * cavity_probs).clamp_min(
                math.log(self.log_prob_eps)
            )
            if self.message_damping < 1.0:
                log_messages = (
                    (1.0 - self.message_damping) * log_messages
                    + self.message_damping * candidate
                )
            else:
                log_messages = candidate

        final_incoming = log_messages[:, self._neighbor_index, self._reverse_direction]
        return final_incoming.sum(dim=2)

    def forward_with_log_h(self, xt: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        raw = self._forward_raw(xt, t)
        if not self.preconditioning_enabled:
            log_h = torch.clamp(raw, min=self.log_h_clip_min, max=self.log_h_clip_max)
            h = torch.exp(log_h)
            return h, log_h

        delta = torch.clamp(raw, min=self.log_h_clip_min, max=self.log_h_clip_max)
        base = self._baseline_log_h(xt=xt, t=t)
        log_h = base + delta
        log_h = torch.clamp(log_h, min=self.log_h_clip_min, max=self.log_h_clip_max)
        h = torch.exp(log_h)
        return h, log_h

    def forward(self, xt: Tensor, t: Tensor) -> Tensor:
        h, _ = self.forward_with_log_h(xt, t)
        return h
__all__ = ["GuidanceHead"]
