"""A single transformer decoder layer.

Llama-style: ``input_norm + attention + pre_mlp_norm + mlp``.
DeepSeek-V2 MoE layer: ``input_norm + attention(MLA) + pre_mlp_norm + moe``.
"""

from __future__ import annotations

from typing import Literal

from projection.configs import ModelConfig
from projection.core.modules import (
    AttentionModule,
    MLPModule,
    ModuleParams,
    MoEModule,
    NormModule,
)

LayerKind = Literal["dense", "moe"]


class TransformerLayer:
    def __init__(self, model: ModelConfig, kind: LayerKind = "dense"):
        if kind == "moe" and not model.moe.enabled:
            raise ValueError("Cannot construct a MoE layer when model.moe.enabled is false")
        self._model = model
        self.kind = kind
        self.input_norm = NormModule(model)
        self.attention = AttentionModule(model)
        self.pre_mlp_norm = NormModule(model)
        self.mlp: MLPModule | MoEModule = MoEModule(model) if kind == "moe" else MLPModule(model)

    def _mlp_param_count(self, ep_size: int) -> int:
        if isinstance(self.mlp, MoEModule):
            return self.mlp.param_count(ep_size=ep_size)
        return self.mlp.param_count()

    def param_breakdown(self, ep_size: int = 1) -> list[ModuleParams]:
        ffn_name = "moe" if self.kind == "moe" else "mlp"
        return [
            ModuleParams("input_norm", self.input_norm.param_count()),
            ModuleParams("attention", self.attention.param_count()),
            ModuleParams("pre_mlp_norm", self.pre_mlp_norm.param_count()),
            ModuleParams(ffn_name, self._mlp_param_count(ep_size)),
        ]

    def param_count(self, ep_size: int = 1) -> int:
        return sum(p.count for p in self.param_breakdown(ep_size))
