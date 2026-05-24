"""Per-rank trainer. Composes a transformer model + optimizer and produces a memory breakdown."""

from __future__ import annotations

from dataclasses import dataclass

from projection.configs import (
    ModelConfig,
    OptimizerKind,
    ParallelConfig,
    Precision,
    TrainingHyperparameters,
    Workload,
)
from projection.core.activations import total_activation_bytes_for_rank
from projection.core.model import TransformerModel
from projection.core.modules import ModuleParams
from projection.core.optimizer import (
    DistributedOptimizer,
    FSDPOptimizer,
    OptimizerMemory,
    model_param_bytes,
)
from projection.parallel.ranks import (
    ModelPartition,
    RankCoord,
    decompose_rank,
    gradient_accumulation_steps,
    partition_for_rank,
    validate_full_config,
)


@dataclass(frozen=True)
class MemoryBreakdown:
    """Per-rank training memory breakdown, in bytes.

    The main gradient buffer lives inside ``optimizer.grad_buffer_bytes`` (a
    single contiguous DDP-allocated buffer, fp32 by default in BF16 training).
    There is no separate model-side gradient — ``param.main_grad`` is a slice
    into that one buffer.
    """

    param_bytes: int
    activation_bytes: int
    optimizer: OptimizerMemory
    precision: Precision

    @property
    def total_bytes(self) -> int:
        return self.param_bytes + self.activation_bytes + self.optimizer.total_bytes


@dataclass(frozen=True)
class TrainerReport:
    """Full report for a single rank, intended for direct UI consumption."""

    global_rank: int
    rank_coord: RankCoord
    partition: ModelPartition
    param_count: int
    param_breakdown: list[ModuleParams]
    memory: MemoryBreakdown


class Trainer:
    """Bound to a single global rank. Composes a per-rank :class:`TransformerModel` and optimizer."""

    def __init__(
        self,
        model_config: ModelConfig,
        parallel: ParallelConfig,
        workload: Workload,
        global_rank: int = 0,
        hyperparameters: TrainingHyperparameters | None = None,
    ):
        validate_full_config(model_config, parallel, workload)
        self.model_config = model_config
        self.parallel = parallel
        self.workload = workload
        self.global_rank = global_rank
        self.hyperparameters = hyperparameters or TrainingHyperparameters()

        self.rank_coord = decompose_rank(global_rank, parallel)
        self.partition = partition_for_rank(model_config, parallel, global_rank)
        self.transformer_model = TransformerModel(
            model_config,
            self.partition,
            tensor_parallel_size=parallel.tensor_model_parallel_size,
            expert_parallel_size=parallel.expert_model_parallel_size,
        )
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self) -> DistributedOptimizer | FSDPOptimizer:
        if self.parallel.optimizer_kind is OptimizerKind.DISTRIBUTED_OPTIMIZER:
            return DistributedOptimizer(self.parallel)
        if self.parallel.optimizer_kind in (OptimizerKind.TORCH_FSDP2, OptimizerKind.MEGATRON_FSDP):
            return FSDPOptimizer(self.parallel)
        raise NotImplementedError(f"optimizer_kind={self.parallel.optimizer_kind} is not modeled")

    def report(self) -> TrainerReport:
        dense_params, routed_params = self._params_on_rank_split()
        param_breakdown = self.transformer_model.param_breakdown()
        total_param_bytes = model_param_bytes(dense_params + routed_params, self.parallel.precision)
        memory = MemoryBreakdown(
            param_bytes=total_param_bytes,
            activation_bytes=self._activation_bytes(),
            optimizer=self.optimizer.memory_for_split(dense_params, routed_params, self.parallel.precision),
            precision=self.parallel.precision,
        )
        return TrainerReport(
            global_rank=self.global_rank,
            rank_coord=self.rank_coord,
            partition=self.partition,
            param_count=self.transformer_model.param_count(),
            param_breakdown=param_breakdown,
            memory=memory,
        )

    def _params_on_rank_split(self) -> tuple[int, int]:
        """``(dense_params, routed_expert_params)`` after TP / ETP (and DP for FSDP).

        - PP sharding is reflected in :class:`ModelPartition`.
        - Dense params shard by TP (and additionally DP for FSDP).
        - Routed expert params shard by ETP (already sharded by EP inside
          :func:`TransformerModel.param_count_split`). When ``EDP < 1`` each
          rank physically holds ``1/EDP`` expert slices, so we multiply by that
          replication factor.
        """
        dense, routed = self.transformer_model.param_count_split()
        tp = self.parallel.tensor_model_parallel_size
        etp = self.parallel.effective_expert_tensor_parallel_size
        edp = self.parallel.expert_data_parallel_size  # float
        routed_replication = max(1.0, 1.0 / edp) if edp > 0 else 1.0

        dense_denom = tp
        routed_denom = etp
        if self.parallel.optimizer_kind in (OptimizerKind.TORCH_FSDP2, OptimizerKind.MEGATRON_FSDP):
            dense_denom *= self.parallel.data_parallel_size
            # Under FSDP we approximate routed experts as DP-sharded too, but
            # production FSDP+MoE setups are uncommon — flag in docs/UI.
            routed_denom *= self.parallel.data_parallel_size
        dense_on_rank = dense // max(1, dense_denom)
        routed_on_rank = int((routed * routed_replication) // max(1, routed_denom))
        return dense_on_rank, routed_on_rank

    def _params_on_rank(self) -> int:
        d, r = self._params_on_rank_split()
        return d + r

    def _activation_bytes(self) -> int:
        ga = gradient_accumulation_steps(self.parallel, self.workload) or None
        return total_activation_bytes_for_rank(
            self.model_config,
            self.workload,
            self.parallel,
            self.partition.num_layers_on_rank,
            pp_rank=self.rank_coord.pp,
            is_first_pp=self.partition.has_embedding,
            is_last_pp=self.partition.has_final_norm,
            num_microbatches=ga,
            num_dense_layers_on_rank=self.transformer_model.block.num_dense_on_rank,
            num_moe_layers_on_rank=self.transformer_model.block.num_moe_on_rank,
        )
