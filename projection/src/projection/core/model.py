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
        all norms, the final norm and the output projection.

        ``routed_params`` is the routed-expert weights only (sharded by EP).
        """
        dense = 0
        if self.embedding is not None:
            dense += self.embedding.param_count(self._tp)
        if self.final_norm is not None:
            dense += self.final_norm.param_count()
        dense += self.output_projection_size

        dense_layer = self.block.dense_layer
        moe_layer = self.block.moe_layer
        if dense_layer is not None:
            dense += self.block.num_dense_on_rank * dense_layer.param_count(ep_size=self._ep)

        routed = 0
        if moe_layer is not None:
            moe_block = moe_layer.mlp
            from projection.core.modules import MoEModule

            assert isinstance(moe_block, MoEModule), "MoE layer must own a MoEModule"
            # Routed experts: sharded by EP.
            routed_per_layer = (
                moe_block.cfg.num_routed_experts // max(1, self._ep)
            ) * moe_block.routed_expert_param_count()
            # Everything else in the MoE layer counts as dense (norms + attention +
            # router + shared experts).
            non_routed_per_layer = moe_layer.param_count(ep_size=self._ep) - routed_per_layer
            dense += self.block.num_moe_on_rank * non_routed_per_layer
            routed += self.block.num_moe_on_rank * routed_per_layer

        return dense, routed

    def param_breakdown(self) -> list[ModuleParams]:
        out: list[ModuleParams] = []
        if self.embedding is not None:
            out.append(ModuleParams("embedding", self.embedding.param_count(self._tp)))
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
