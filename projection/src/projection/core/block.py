"""A transformer block: a slice of layers owned by a single pipeline rank.

In dense / MLA models all layers of a given kind are identical so we keep one
representative and multiply. With MoE models that have a
``first_k_dense_replace`` prefix of dense layers, we keep one representative
of each kind and track how many of each fall inside this rank's slice
(``[first_layer_idx, first_layer_idx + num_layers_on_rank)``).

DeepSeek-V4 is the first model where layers are *not* identical even within a
kind: each layer has its own ``compress_ratio ∈ {0, 4, 128}`` and the
Compressor / Indexer parameter cost differs accordingly. For V4 we therefore
materialize every owned layer individually (still cheap — at most a few
dozen :class:`TransformerLayer` Python objects per rank).
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
        self._v4_layers: list[TransformerLayer] | None = None

        if model.is_v4:
            self._v4_layers = self._build_v4_layers()
            # For V4 we still keep one representative per kind so existing
            # consumers (activation memory, breakdown) keep working.
            if self.num_dense_on_rank > 0:
                self._dense_layer = TransformerLayer(model, kind="dense")
            if self.num_moe_on_rank > 0:
                self._moe_layer = TransformerLayer(
                    model, kind="moe", compress_ratio=self._representative_v4_cr()
                )
        else:
            if self.num_dense_on_rank > 0:
                self._dense_layer = TransformerLayer(model, kind="dense")
            if self.num_moe_on_rank > 0:
                self._moe_layer = TransformerLayer(model, kind="moe")

    def _build_v4_layers(self) -> list[TransformerLayer]:
        from projection.core.deepseek_v4 import normalize_compress_ratios

        decoder_ratios, _ = normalize_compress_ratios(self._model)
        layers: list[TransformerLayer] = []
        dense_end = (
            self._model.moe.first_k_dense_replace
            if self._model.moe.enabled
            else self._model.architecture.num_layers
        )
        for offset in range(self.num_layers_on_rank):
            idx = self.first_layer_idx + offset
            kind: str = "moe" if (self._model.moe.enabled and idx >= dense_end) else "dense"
            cr = decoder_ratios[idx] if idx < len(decoder_ratios) else 0
            layers.append(TransformerLayer(self._model, kind=kind, compress_ratio=cr))  # type: ignore[arg-type]
        return layers

    def _representative_v4_cr(self) -> int:
        """A typical compress_ratio for the rep MoE layer — used only by activation
        formulas that don't currently distinguish branches. The Compressor /
        Indexer activations are small relative to the main path, so a fixed
        representative value is acceptable. We default to 4 (CSA) since CSA is
        the most expensive branch."""
        return 4

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
    def num_hash_moe_on_rank(self) -> int:
        """Count of hash-routed MoE layers (those with index < num_hash_layers)
        falling inside this rank's slice. V4 only.
        """
        if not self._model.is_v4 or self._model.moe.num_hash_layers <= 0:
            return 0
        # Hash layers are the first num_hash_layers MoE layers (which sit
        # right after the dense prefix). For DeepSeek-V4 first_k_dense_replace=0.
        dense_end = self._model.moe.first_k_dense_replace if self._model.moe.enabled else 0
        hash_start = dense_end
        hash_end = dense_end + self._model.moe.num_hash_layers
        start = self.first_layer_idx
        end = self.first_layer_idx + self.num_layers_on_rank
        return max(0, min(end, hash_end) - max(start, hash_start))

    @property
    def dense_layer(self) -> TransformerLayer | None:
        return self._dense_layer

    @property
    def moe_layer(self) -> TransformerLayer | None:
        return self._moe_layer

    @property
    def v4_layers(self) -> list[TransformerLayer] | None:
        return self._v4_layers

    def param_count(self, ep_size: int = 1) -> int:
        if self._v4_layers is not None:
            return sum(layer.param_count(ep_size=ep_size) for layer in self._v4_layers)
        total = 0
        if self._dense_layer is not None:
            total += self.num_dense_on_rank * self._dense_layer.param_count(ep_size=ep_size)
        if self._moe_layer is not None:
            total += self.num_moe_on_rank * self._moe_layer.param_count(ep_size=ep_size)
        return total
