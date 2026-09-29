"""RoPE DeiT backbone shared by lattice guidance models.

The mixed two-dimensional RoPE utilities are adapted from RoPE-ViT by NAVER
AI Lab: https://github.com/naver-ai/rope-vit.
"""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor


def init_random_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True) -> Tensor:
    if dim % 4 != 0:
        raise ValueError("head dimension must be divisible by 4 for mixed 2D RoPE")
    freqs_x = []
    freqs_y = []
    mag = 1.0 / (theta ** (torch.arange(0, dim, 4, dtype=torch.float32) / dim))
    for _ in range(num_heads):
        angle = torch.rand(1) * 2.0 * math.pi if rotate else torch.zeros(1)
        fx = torch.cat([mag * torch.cos(angle), mag * torch.cos(0.5 * math.pi + angle)], dim=-1)
        fy = torch.cat([mag * torch.sin(angle), mag * torch.sin(0.5 * math.pi + angle)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    return torch.stack([torch.stack(freqs_x, dim=0), torch.stack(freqs_y, dim=0)], dim=0)


def init_t_xy(end_x: int, end_y: int) -> tuple[Tensor, Tensor]:
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = torch.remainder(t, end_x)
    t_y = torch.div(t, end_x, rounding_mode="floor")
    return t_x, t_y


def compute_mixed_cis(freqs: Tensor, t_x: Tensor, t_y: Tensor, num_heads: int) -> Tensor:
    n_tokens = t_x.shape[0]
    depth = freqs.shape[1]
    with torch.autocast(device_type=t_x.device.type, enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(depth, n_tokens, num_heads, -1).permute(0, 2, 1, 3)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(depth, n_tokens, num_heads, -1).permute(0, 2, 1, 3)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
    return freqs_cis


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0) -> Tensor:
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4, dtype=torch.float32) / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4, dtype=torch.float32) / dim))
    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: Tensor, x: Tensor) -> Tensor:
    ndim = x.ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
    else:
        raise ValueError(f"freqs_cis shape {tuple(freqs_cis.shape)} is incompatible with {tuple(x.shape)}")
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RoPEAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
        bsz, n_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, n_tokens, 3, self.num_heads, dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q_patch, k_patch = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)
        # Avoid in-place writes through q/k views. They invalidate the complex views
        # retained by RoPE autograd and fail during backward on CPU (and some CUDA builds).
        q = torch.cat((q[:, :, :1], q_patch), dim=2)
        k = torch.cat((k[:, :, :1], k_patch), dim=2)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(bsz, n_tokens, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        init_values: float = 1.0e-4,
        norm_eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = RoPEAttention(dim=dim, num_heads=num_heads, qkv_bias=True, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
        x = x + self.gamma_1 * self.attn(self.norm1(x), freqs_cis=freqs_cis)
        x = x + self.gamma_2 * self.mlp(self.norm2(x))
        return x


class MDNSRoPEDeit2DBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        length: int,
        embed_dim: int = 64,
        num_blocks: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        dtype: str = "bfloat16",
        use_ape: bool = True,
        rope_mixed: bool = True,
        rope_theta: float = 10.0,
        norm_eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        grid = int(math.sqrt(length))
        if grid * grid != length:
            raise ValueError("length must be a perfect square")
        if (embed_dim // num_heads) % 4 != 0:
            raise ValueError("embed_dim // num_heads must be divisible by 4 for mixed RoPE")

        self.vocab_size = int(vocab_size)
        self.length = int(length)
        self.grid = int(grid)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_blocks = int(num_blocks)
        self.use_ape = bool(use_ape)
        self.rope_mixed = bool(rope_mixed)
        self.norm_eps = float(norm_eps)
        if self.norm_eps <= 0.0:
            raise ValueError("norm_eps must be > 0")
        self.patch_size = 1

        self.token_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.length, self.embed_dim)) if self.use_ape else None
        self.blocks = nn.ModuleList(
            [
                LayerScaleBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=dropout,
                    norm_eps=self.norm_eps,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim, eps=self.norm_eps)

        if self.rope_mixed:
            freqs = [
                init_random_2d_freqs(dim=self.embed_dim // self.num_heads, num_heads=self.num_heads, theta=rope_theta)
                for _ in range(self.num_blocks)
            ]
            self.freqs = nn.Parameter(torch.stack(freqs, dim=1).view(2, self.num_blocks, -1), requires_grad=True)
            t_x, t_y = init_t_xy(end_x=self.grid, end_y=self.grid)
            self.register_buffer("freqs_t_x", t_x, persistent=False)
            self.register_buffer("freqs_t_y", t_y, persistent=False)
        else:
            self.compute_cis = partial(compute_axial_cis, dim=self.embed_dim // self.num_heads, theta=rope_theta)
            freqs_cis = self.compute_cis(end_x=self.grid, end_y=self.grid)
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.autocast_dtype = dtype_map.get(str(dtype).lower(), torch.bfloat16)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _block_forward(self, block: LayerScaleBlock, x: Tensor, freqs_cis: Tensor) -> Tensor:
        return block(x, freqs_cis=freqs_cis)

    def _forward_impl(self, xt: Tensor, t: Tensor) -> Tensor:
        bsz = xt.shape[0]
        x = self.token_embed(xt)
        x = x + self.time_embed(t.unsqueeze(-1)).unsqueeze(1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        cls_tokens = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.rope_mixed:
            freqs_cis = compute_mixed_cis(
                freqs=self.freqs,
                t_x=self.freqs_t_x.to(x.device),
                t_y=self.freqs_t_y.to(x.device),
                num_heads=self.num_heads,
            )
            for idx, block in enumerate(self.blocks):
                x = self._block_forward(block, x, freqs_cis[idx])
        else:
            freqs_cis = self.freqs_cis.to(x.device)
            for block in self.blocks:
                x = self._block_forward(block, x, freqs_cis)

        x = self.norm(x)
        return x[:, 1:, :]

    def forward(self, xt: Tensor, t: Tensor) -> Tensor:
        if xt.ndim != 2:
            raise ValueError(f"xt must be [B,D], got {tuple(xt.shape)}")
        if t.ndim != 1:
            raise ValueError(f"t must be [B], got {tuple(t.shape)}")
        if xt.shape[1] != self.length:
            raise ValueError(f"Expected length={self.length}, got {xt.shape[1]}")
        if xt.shape[0] != t.shape[0]:
            raise ValueError("Batch size mismatch between xt and t")

        if xt.is_cuda:
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                out = self._forward_impl(xt=xt, t=t)
        else:
            out = self._forward_impl(xt=xt, t=t)
        return out.to(dtype=torch.float32)


__all__ = ["MDNSRoPEDeit2DBackbone"]
