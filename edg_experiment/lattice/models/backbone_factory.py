from __future__ import annotations

from typing import Any, Dict, Tuple

import torch.nn as nn

from edg_experiment.lattice.models.backbone_tiny_mlp import TinyMLPBackbone
from edg_experiment.lattice.models.backbone_mdns_rope_deit2d import MDNSRoPEDeit2DBackbone


def _guidance_backbone_cfg(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    role_cfg = dict(model_cfg.get("guidance_backbone", {}))
    btype = str(role_cfg.get("type", "mdns_rope_deit2d")).lower()
    presets = model_cfg.get("presets", {})
    preset_cfg = dict(presets.get(btype, {})) if isinstance(presets, dict) and isinstance(presets.get(btype), dict) else {}
    merged: Dict[str, Any] = {}
    merged.update(preset_cfg)
    merged.update(role_cfg)
    merged["type"] = btype
    return merged


def build_backbone(
    model_cfg: Dict[str, Any],
    vocab_size: int,
    length: int,
) -> Tuple[nn.Module, int]:
    backbone_cfg = _guidance_backbone_cfg(model_cfg=model_cfg)
    btype = str(backbone_cfg.get("type", "mdns_rope_deit2d")).lower()

    if btype == "mdns_rope_deit2d":
        embed_dim = int(backbone_cfg.get("embed_dim", backbone_cfg.get("d_model", 64)))
        backbone = MDNSRoPEDeit2DBackbone(
            vocab_size=vocab_size,
            length=length,
            embed_dim=embed_dim,
            num_blocks=int(backbone_cfg.get("num_blocks", 6)),
            num_heads=int(backbone_cfg.get("num_heads", 4)),
            mlp_ratio=float(backbone_cfg.get("mlp_ratio", 4.0)),
            dropout=float(backbone_cfg.get("dropout", 0.0)),
            dtype=str(backbone_cfg.get("dtype", "bfloat16")),
            use_ape=bool(backbone_cfg.get("use_ape", True)),
            rope_mixed=bool(backbone_cfg.get("rope_mixed", True)),
            rope_theta=float(backbone_cfg.get("rope_theta", 10.0)),
            norm_eps=float(backbone_cfg.get("norm_eps", 1.0e-5)),
        )
        return backbone, embed_dim

    if btype == "tiny_mlp":
        d_model = int(backbone_cfg.get("d_model", 64))
        backbone = TinyMLPBackbone(
            vocab_size=vocab_size,
            length=length,
            d_model=d_model,
            hidden_dim=int(backbone_cfg.get("hidden_dim", 256)),
            variant=str(backbone_cfg.get("variant", "deep")),
        )
        return backbone, d_model

    raise ValueError(f"Unknown guidance_backbone.type: {btype}")


__all__ = ["build_backbone"]
