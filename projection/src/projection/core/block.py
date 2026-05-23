"""A transformer block: a slice of layers owned by a single pipeline rank.

In dense models, all layers are identical so we keep one representative and
multiply. With MoE models that have a ``first_k_dense_replace`` prefix of
dense layers, we keep one representative of each kind and track how many of
each fall inside this rank's slice (``[first_layer_idx, first_layer_idx + num_layers_on_rank)``).
"""

from __future__ import annotations

from projection.configs import ModelConfig
from projection.core.layer import TransformerLayer


class TransformerBlock:
    def __init__(self, model: ModelConfig, num_layers_on_rank: int, first_layer_idx: int = 0):
        if num_layers_on_rank < 0:
            raise ValueError(f"num_layers_on_rank must be >= 0, got {num_layers_on_rank}")
        self._model = model
        self.num_layers_on_rank = num_layers_on_rank
        self.first_layer_idx = first_layer_idx
        self._dense_layer: TransformerLayer | None = None
        self._moe_layer: TransformerLayer | None = None
        if self.num_dense_on_rank > 0:
            self._dense_layer = TransformerLayer(model, kind="dense")
        if self.num_moe_on_rank > 0:
            self._moe_layer = TransformerLayer(model, kind="moe")

    @property
    def num_dense_on_rank(self) -> int:
        dense_end = (
            self._model.moe.first_k_dense_replace
            if self._model.moe.enabled
            else self._model.architecture.num_layers
        )
        start = self.first_layer_idx
        end = self.first_layer_idx + self.num_layers_on_rank
        return max(0, min(end, dense_end) - start)

    @property
    def num_moe_on_rank(self) -> int:
        if not self._model.moe.enabled:
            return 0
        return self.num_layers_on_rank - self.num_dense_on_rank

    @property
    def dense_layer(self) -> TransformerLayer | None:
        return self._dense_layer

    @property
    def moe_layer(self) -> TransformerLayer | None:
        return self._moe_layer

    def param_count(self, ep_size: int = 1) -> int:
        total = 0
        if self._dense_layer is not None:
            total += self.num_dense_on_rank * self._dense_layer.param_count(ep_size=ep_size)
        if self._moe_layer is not None:
            total += self.num_moe_on_rank * self._moe_layer.param_count(ep_size=ep_size)
        return total
