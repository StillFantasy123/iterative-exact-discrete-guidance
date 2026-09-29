from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class TinyMLPBackboneDeep(nn.Module):
    def __init__(self, vocab_size: int, length: int, d_model: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.length = int(length)
        self.d_model = int(d_model)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.time_proj = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.mlp = nn.Sequential(
            nn.Linear(length * d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, length * d_model),
        )

    def forward(self, xt: Tensor, t: Tensor) -> Tensor:
        x = self.embed(xt) + self.time_proj(t.unsqueeze(-1)).unsqueeze(1)
        return self.mlp(x.reshape(x.shape[0], -1)).reshape(x.shape[0], self.length, self.d_model)


class TinyMLPBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        length: int,
        d_model: int,
        hidden_dim: int = 128,
        variant: str = "shallow",
    ) -> None:
        super().__init__()
        if str(variant).lower() != "deep":
            raise ValueError("the released Ising4 checkpoints require variant=deep")
        self.impl = TinyMLPBackboneDeep(
            vocab_size=vocab_size,
            length=length,
            d_model=d_model,
            hidden_dim=hidden_dim,
        )

    def forward(self, xt: Tensor, t: Tensor) -> Tensor:
        return self.impl(xt, t)


__all__ = ["TinyMLPBackbone", "TinyMLPBackboneDeep"]
