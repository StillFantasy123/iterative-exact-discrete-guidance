from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from torch import Tensor

from edg_experiment.ising.energy import ising_energy_binary
from edg_experiment.lattice.potts import potts_energy, validate_potts_samples


_SUPPORTED_PROBLEMS = {"ising", "potts"}


def get_problem_kind(cfg: Dict[str, Any]) -> str:
    raw = cfg.get("problem", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("problem must be a mapping")
    kind = str(raw.get("kind", "ising")).lower()
    if kind not in _SUPPORTED_PROBLEMS:
        raise ValueError(f"problem.kind must be one of {sorted(_SUPPORTED_PROBLEMS)}, got {kind}")
    return kind


def problem_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    kind = get_problem_kind(cfg)
    raw = cfg.get(kind)
    if not isinstance(raw, dict):
        raise ValueError(f"{kind} config must be a mapping")
    return raw


@dataclass(frozen=True)
class LatticeProblem:
    kind: str
    L: int
    num_states: int
    beta: float
    J: float
    h: float = 0.0

    @property
    def length(self) -> int:
        return int(self.L) * int(self.L)

    def validate_samples(self, samples: Tensor) -> Tensor:
        if self.kind == "potts":
            return validate_potts_samples(samples, L=self.L, q=self.num_states)
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        if samples.ndim != 2 or int(samples.shape[1]) != self.length:
            raise ValueError(f"samples must be [B,{self.length}], got {tuple(samples.shape)}")
        if bool(((samples < 0) | (samples >= self.num_states)).any()):
            raise ValueError(f"Ising samples must contain labels in [0,{self.num_states - 1}]")
        return samples

    def energy(self, samples: Tensor) -> Tensor:
        self.validate_samples(samples)
        if self.kind == "potts":
            return potts_energy(samples, L=self.L, q=self.num_states, J=self.J)
        return ising_energy_binary(samples, L=self.L, J=self.J, h=self.h)

def build_problem(cfg: Dict[str, Any]) -> LatticeProblem:
    kind = get_problem_kind(cfg)
    raw = problem_config(cfg)
    vocab = cfg.get("vocab", {})
    if not isinstance(vocab, dict):
        raise ValueError("vocab must be a mapping")
    num_states = int(vocab.get("num_states", 0))
    L = int(raw.get("L", 0))
    beta = float(raw.get("beta", 0.0))
    J = float(raw.get("J", 1.0))
    if L <= 0:
        raise ValueError(f"{kind}.L must be positive")
    if num_states < 2:
        raise ValueError("vocab.num_states must be at least 2")

    if kind == "potts":
        q = int(raw.get("q", num_states))
        if q != num_states:
            raise ValueError(f"potts.q ({q}) must equal vocab.num_states ({num_states})")
        h = 0.0
    else:
        if num_states != 2:
            raise ValueError("Ising requires vocab.num_states=2")
        h = float(raw.get("h", 0.0))

    return LatticeProblem(
        kind=kind,
        L=L,
        num_states=num_states,
        beta=beta,
        J=J,
        h=h,
    )


__all__ = [
    "LatticeProblem",
    "build_problem",
    "get_problem_kind",
    "problem_config",
]
