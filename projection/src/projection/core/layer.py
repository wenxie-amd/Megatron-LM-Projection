"""A single transformer decoder layer.

Llama-style: ``input_norm + attention + pre_mlp_norm + mlp``.
DeepSeek-V2 MoE layer: ``input_norm + attention(MLA) + pre_mlp_norm + moe``.
DeepSeek-V4 layer: ``input_norm + attention(V4 + Compressor [+ Indexer]) +
pre_mlp_norm + moe`` plus 2 per-layer ``HyperMixer`` modules (counted
separately so the breakdown stays interpretable). The per-layer compress
ratio drives the Compressor / Indexer size.
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
    def __init__(
        self,
        model: ModelConfig,
        kind: LayerKind = "dense",
        *,
        compress_ratio: int | None = None,
    ):
        if kind == "moe" and not model.moe.enabled:
            raise ValueError("Cannot construct a MoE layer when model.moe.enabled is false")
        self._model = model
        self.kind = kind
        # Only used for V4 attention dispatch; ignored for MHA / MLA layers.
        self.compress_ratio = compress_ratio
        self.input_norm = NormModule(model)
        self.attention = AttentionModule(model)
        self.pre_mlp_norm = NormModule(model)
        self.mlp: MLPModule | MoEModule = MoEModule(model) if kind == "moe" else MLPModule(model)

    def _mlp_param_count(self, ep_size: int) -> int:
        if isinstance(self.mlp, MoEModule):
            return self.mlp.param_count(ep_size=ep_size)
        return self.mlp.param_count()

    def _attention_param_count(self) -> int:
        if self._model.attention.use_deepseek_v4:
            return self.attention.param_count(compress_ratio=self.compress_ratio or 0)
        return self.attention.param_count()

    def _hc_param_count(self) -> int:
        if not self._model.is_v4 or not self._model.hyper_connection.enabled:
            return 0
        from projection.core.deepseek_v4 import hc_per_layer_param_count

        return hc_per_layer_param_count(self._model)

    def param_breakdown(self, ep_size: int = 1) -> list[ModuleParams]:
        ffn_name = "moe" if self.kind == "moe" else "mlp"
        out = [
            ModuleParams("input_norm", self.input_norm.param_count()),
            ModuleParams("attention", self._attention_param_count()),
            ModuleParams("pre_mlp_norm", self.pre_mlp_norm.param_count()),
            ModuleParams(ffn_name, self._mlp_param_count(ep_size)),
        ]
        hc = self._hc_param_count()
        if hc > 0:
            out.append(ModuleParams("hc_mixers", hc))
        return out

    def param_count(self, ep_size: int = 1) -> int:
        return sum(p.count for p in self.param_breakdown(ep_size))
