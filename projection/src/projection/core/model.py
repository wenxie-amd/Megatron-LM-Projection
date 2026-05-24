"""The whole transformer model on a single pipeline rank.

A ``TransformerModel`` holds whatever the given pipeline rank owns:

- ``embedding`` only on the first PP rank
- ``block`` (its slice of layers)
- ``final_norm`` + ``output_projection`` only on the last PP rank

Tensor-parallel and expert-parallel sharding are reflected in byte counts (not
in the parameter *count* of individual modules). We follow Megatron's
convention that ``param_count()`` returns the *logical* parameter count on this
rank after PP+EP sharding, before TP sharding.
"""

from __future__ import annotations

from projection.configs import ModelConfig
from projection.core.block import TransformerBlock
from projection.core.modules import (
    EmbeddingModule,
    ModuleParams,
    NormModule,
    padded_vocab_size,
)
from projection.parallel.ranks import ModelPartition


class TransformerModel:
    def __init__(
        self,
        model: ModelConfig,
        partition: ModelPartition,
        tensor_parallel_size: int = 1,
        expert_parallel_size: int = 1,
    ):
        self._model = model
        self._partition = partition
        self._tp = tensor_parallel_size
        self._ep = expert_parallel_size

        self.embedding = EmbeddingModule(model) if partition.has_embedding else None
        self.block = TransformerBlock(
            model,
            num_layers_on_rank=partition.num_layers_on_rank,
            first_layer_idx=partition.first_layer_idx,
        )
        self.final_norm = NormModule(model) if partition.has_final_norm else None
        self.output_projection_size = (
            padded_vocab_size(model.architecture, tensor_parallel_size) * model.architecture.hidden_size
            if partition.has_output_projection
            else 0
        )

    def param_count(self) -> int:
        dense, routed = self.param_count_split()
        return dense + routed

    def param_count_split(self) -> tuple[int, int]:
        """Return ``(dense_params, routed_expert_params)`` on this rank.

        ``dense_params`` includes the embedding, attention (Q/K/V/O), MLPs in
        dense layers, the MoE block's router + shared experts in MoE layers,
        all norms, the final norm and the output projection. For V4 models it
        additionally includes the per-layer HyperMixer parameters, the trunk
        HyperHead on the last PP rank, the per-MTP-depth HyperHead, the MTP
        ``eh_proj`` and inner V4 layer cost, and the non-trainable ``tid2eid``
        hash routing buffer (counted as if it were FP32 — close enough).

        ``routed_params`` is the routed-expert weights only (sharded by EP).
        """
        dense = 0
        if self.embedding is not None:
            dense += self.embedding.param_count(self._tp)
        if self.final_norm is not None:
            dense += self.final_norm.param_count()
        dense += self.output_projection_size

        v4_layers = self.block.v4_layers
        if v4_layers is not None:
            d, r = self._v4_param_count_split(v4_layers)
            dense += d
            return dense + self._v4_trunk_extras(), r

        dense_layer = self.block.dense_layer
        moe_layer = self.block.moe_layer
        if dense_layer is not None:
            dense += self.block.num_dense_on_rank * dense_layer.param_count(ep_size=self._ep)

        routed = 0
        if moe_layer is not None:
            moe_block = moe_layer.mlp
            from projection.core.modules import MoEModule

            assert isinstance(moe_block, MoEModule), "MoE layer must own a MoEModule"
            routed_per_layer = (
                moe_block.cfg.num_routed_experts // max(1, self._ep)
            ) * moe_block.routed_expert_param_count()
            non_routed_per_layer = moe_layer.param_count(ep_size=self._ep) - routed_per_layer
            dense += self.block.num_moe_on_rank * non_routed_per_layer
            routed += self.block.num_moe_on_rank * routed_per_layer

        return dense, routed

    def _v4_param_count_split(self, v4_layers: list) -> tuple[int, int]:
        """Per-rank dense / routed split for the V4 layer list. Each layer is
        iterated individually because attention cost depends on its compress
        ratio (Compressor / Indexer)."""
        from projection.core.modules import MoEModule

        dense = 0
        routed = 0
        for layer in v4_layers:
            layer_total = layer.param_count(ep_size=self._ep)
            if layer.kind == "moe":
                moe_block = layer.mlp
                assert isinstance(moe_block, MoEModule)
                routed_per_layer = (
                    moe_block.cfg.num_routed_experts // max(1, self._ep)
                ) * moe_block.routed_expert_param_count()
                dense += layer_total - routed_per_layer
                routed += routed_per_layer
            else:
                dense += layer_total
        # tid2eid lookup buffers (int32) — counted as FP32-equivalent element
        # count; tracked under "dense" so the params byte breakdown captures
        # them. They sit on whichever PP rank owns the hash-routed layer.
        from projection.core.deepseek_v4 import hash_routing_buffer_per_layer

        buf = hash_routing_buffer_per_layer(self._model)
        dense += buf.elements * self.block.num_hash_moe_on_rank
        return dense, routed

    def _v4_trunk_extras(self) -> int:
        """V4-only extras attached to the model trunk: trunk-end HyperHead on
        the last PP rank, MTP layers (full inner V4 layer + eh_proj + own
        HyperHead) on the last PP rank.
        """
        if not self._model.is_v4:
            return 0
        from projection.core.deepseek_v4 import (
            hc_per_layer_param_count,
            hyper_head_param_count,
            normalize_compress_ratios,
            v4_attention_param_count_per_layer,
        )
        from projection.core.layer import TransformerLayer
        from projection.core.modules import MoEModule

        extras = 0
        if not self._partition.has_final_norm:
            return 0

        h = self._model.architecture.hidden_size
        hc = self._model.hyper_connection
        if hc.enabled:
            extras += hyper_head_param_count(h, hc.hc_mult)

        # MTP layers (full inner V4 layer per depth + eh_proj per depth + per-
        # depth HyperHead). All live on the last PP rank.
        mtp = self._model.mtp
        if mtp.num_layers > 0:
            _, mtp_ratios = normalize_compress_ratios(self._model)
            # Each MTP depth is itself a V4 transformer layer (attention + MoE).
            # Build one TransformerLayer per depth and sum, then subtract the
            # routed expert portion (we attribute MTP's experts to "dense" for
            # simplicity since MTP expert weights are typically not EP-sharded
            # the same way as the main trunk experts in v1 — flagged in docs).
            for depth in range(mtp.num_layers):
                cr = mtp_ratios[depth] if depth < len(mtp_ratios) else 0
                kind: str = "moe" if self._model.moe.enabled else "dense"
                mtp_layer = TransformerLayer(
                    self._model, kind=kind, compress_ratio=cr  # type: ignore[arg-type]
                )
                extras += mtp_layer.param_count(ep_size=self._ep)

            # eh_proj per depth: 2H -> H, no bias.
            extras += mtp.num_layers * (2 * h) * h
            # Per-depth HyperHead (use_separate_hc_head=True is the V4 default).
            if hc.enabled and mtp.use_separate_hc_head:
                extras += mtp.num_layers * hyper_head_param_count(h, hc.hc_mult)
            else:
                # Shared trunk head — no extra params.
                pass

            # Avoid silencing the unused-attention helper imports
            _ = v4_attention_param_count_per_layer
            _ = hc_per_layer_param_count
            _ = MoEModule

        return extras

    def param_breakdown(self) -> list[ModuleParams]:
        out: list[ModuleParams] = []
        if self.embedding is not None:
            out.append(ModuleParams("embedding", self.embedding.param_count(self._tp)))

        v4_layers = self.block.v4_layers
        if v4_layers is not None:
            out.extend(self._v4_layer_breakdown(v4_layers))
            out.extend(self._v4_trunk_breakdown())
        else:
            dense_layer = self.block.dense_layer
            moe_layer = self.block.moe_layer
            if dense_layer is not None:
                for sub in dense_layer.param_breakdown(ep_size=self._ep):
                    out.append(ModuleParams(f"layer.{sub.name}", sub.count * self.block.num_dense_on_rank))
            if moe_layer is not None:
                for sub in moe_layer.param_breakdown(ep_size=self._ep):
                    out.append(ModuleParams(f"moe_layer.{sub.name}", sub.count * self.block.num_moe_on_rank))

        if self.final_norm is not None:
            out.append(ModuleParams("final_norm", self.final_norm.param_count()))
        if self.output_projection_size:
            out.append(ModuleParams("output_projection", self.output_projection_size))
        return out

    def _v4_layer_breakdown(self, v4_layers: list) -> list[ModuleParams]:
        """Aggregate per-layer V4 breakdowns into module-level totals."""
        totals: dict[str, int] = {}
        for layer in v4_layers:
            prefix = "moe_layer" if layer.kind == "moe" else "layer"
            for sub in layer.param_breakdown(ep_size=self._ep):
                key = f"{prefix}.{sub.name}"
                totals[key] = totals.get(key, 0) + sub.count
        # Hash routing tid2eid buffer per layer.
        if self.block.num_hash_moe_on_rank > 0:
            from projection.core.deepseek_v4 import hash_routing_buffer_per_layer

            buf = hash_routing_buffer_per_layer(self._model)
            totals["moe_layer.hash_routing_buffer"] = buf.elements * self.block.num_hash_moe_on_rank
        return [ModuleParams(name, count) for name, count in totals.items()]

    def _v4_trunk_breakdown(self) -> list[ModuleParams]:
        """HyperHead at trunk end + MTP block (on the last PP rank)."""
        if not self._model.is_v4 or not self._partition.has_final_norm:
            return []
        from projection.core.deepseek_v4 import (
            hyper_head_param_count,
            normalize_compress_ratios,
        )
        from projection.core.layer import TransformerLayer

        out: list[ModuleParams] = []
        h = self._model.architecture.hidden_size
        hc = self._model.hyper_connection
        if hc.enabled:
            out.append(ModuleParams("hyper_head", hyper_head_param_count(h, hc.hc_mult)))

        mtp = self._model.mtp
        if mtp.num_layers > 0:
            _, mtp_ratios = normalize_compress_ratios(self._model)
            mtp_total = 0
            for depth in range(mtp.num_layers):
                cr = mtp_ratios[depth] if depth < len(mtp_ratios) else 0
                kind: str = "moe" if self._model.moe.enabled else "dense"
                mtp_layer = TransformerLayer(
                    self._model, kind=kind, compress_ratio=cr  # type: ignore[arg-type]
                )
                mtp_total += mtp_layer.param_count(ep_size=self._ep)
            out.append(ModuleParams("mtp.layer", mtp_total))
            out.append(ModuleParams("mtp.eh_proj", mtp.num_layers * (2 * h) * h))
            if hc.enabled and mtp.use_separate_hc_head:
                out.append(
                    ModuleParams("mtp.hyper_head", mtp.num_layers * hyper_head_param_count(h, hc.hc_mult))
                )

        return out
