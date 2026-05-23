"""Rank decomposition and per-PP-stage layer assignment.

We follow Megatron's default rank ordering ``tp-cp-ep-dp-pp`` (innermost to
outermost), i.e. ``global_rank = ((((pp * dp + dp_r) * ep + ep_r) * cp + cp_r) * tp + tp_r)``.
"""

from __future__ import annotations

from dataclasses import dataclass

from projection.configs import ModelConfig, OptimizerKind, ParallelConfig, Workload


@dataclass(frozen=True)
class RankCoord:
    """Per-axis coordinates for a global rank.

    ``tp``, ``cp``, ``dp``, ``pp`` come from Megatron's default
    (non-expert) ``RankGenerator`` (order ``tp-cp-ep-dp-pp`` with ``ep=1``).
    ``ep`` and ``expert_dp`` come from Megatron's **expert**
    ``RankGenerator`` (order ``tp-cp-ep-dp-pp`` with ``cp=1`` and
    ``dp = expert_data_parallel_size``). Both decompose the *same* global rank.
    """

    tp: int
    cp: int
    dp: int
    pp: int
    ep: int
    expert_dp: int


@dataclass(frozen=True)
class ModelPartition:
    """Describes which slices of the model a given PP rank owns."""

    num_layers_on_rank: int
    first_layer_idx: int
    has_embedding: bool
    has_final_norm: bool
    has_output_projection: bool


def validate_parallel_config(model: ModelConfig, parallel: ParallelConfig) -> None:
    """Raise ``ValueError`` with a user-actionable message on any constraint violation."""
    pp = parallel.pipeline_model_parallel_size
    vpp = parallel.virtual_pipeline_model_parallel_size
    layout = parallel.pipeline_model_parallel_layout
    num_layers = model.architecture.num_layers

    if pp < 1 or parallel.tensor_model_parallel_size < 1:
        raise ValueError("pipeline_model_parallel_size and tensor_model_parallel_size must be >= 1")

    if parallel.sequence_parallel and parallel.tensor_model_parallel_size == 1:
        raise ValueError("sequence_parallel requires tensor_model_parallel_size > 1")

    # Optimizer / sharding conflicts (cross-checked against Megatron's arguments.py).
    if parallel.optimizer_kind is OptimizerKind.TORCH_FSDP2:
        if pp > 1:
            raise ValueError(
                "torch_fsdp2 is incompatible with pipeline_model_parallel_size > 1; "
                "either switch optimizer to distributed_optimizer or set pp_size=1"
            )
        if vpp and vpp > 1:
            raise ValueError("torch_fsdp2 is incompatible with virtual_pipeline_model_parallel_size > 1")

    if parallel.optimizer_kind is OptimizerKind.MEGATRON_FSDP:
        if pp > 1:
            raise ValueError("megatron_fsdp is incompatible with pipeline_model_parallel_size > 1 in v1")

    if model.moe.enabled and parallel.expert_model_parallel_size > 1:
        if model.moe.num_routed_experts % parallel.expert_model_parallel_size != 0:
            raise ValueError(
                f"num_routed_experts={model.moe.num_routed_experts} must be divisible by "
                f"expert_model_parallel_size={parallel.expert_model_parallel_size}"
            )
        numer = parallel.context_parallel_size * parallel.data_parallel_size
        if numer % parallel.expert_model_parallel_size != 0:
            raise ValueError(
                f"expert_model_parallel_size={parallel.expert_model_parallel_size} must divide "
                f"cp*dp={numer} (Megatron requires expert_data_parallel_size = cp*dp/ep to be integer)"
            )

    if layout is not None:
        if len(layout) != pp:
            raise ValueError(f"pipeline_model_parallel_layout must have {pp} entries (got {len(layout)})")
        if sum(layout) != num_layers:
            raise ValueError(
                f"pipeline_model_parallel_layout sums to {sum(layout)} but num_layers={num_layers}"
            )
        return

    divisor = pp * (vpp or 1)
    if num_layers % divisor != 0:
        suggestion = (
            "switch to pipeline_model_parallel_layout to specify per-stage layer counts"
            if vpp
            else f"choose a pipeline_model_parallel_size that divides num_layers={num_layers}"
        )
        raise ValueError(
            f"num_layers={num_layers} is not divisible by pp_size*vpp_size={divisor}; {suggestion}"
        )


def validate_workload(model: ModelConfig, parallel: ParallelConfig, workload: Workload) -> None:
    """Raise ``ValueError`` with a user-actionable message on workload constraint violations."""
    if workload.micro_batch_size < 1:
        raise ValueError("micro_batch_size must be >= 1")
    if workload.global_batch_size < 1:
        raise ValueError("global_batch_size must be >= 1")
    if workload.seq_length < 1:
        raise ValueError("seq_length must be >= 1")

    dp = parallel.data_parallel_size
    mbs = workload.micro_batch_size
    gbs = workload.global_batch_size
    per_step = dp * mbs
    if gbs % per_step != 0:
        raise ValueError(
            f"global_batch_size={gbs} must be divisible by data_parallel_size*micro_batch_size={per_step} "
            f"(dp={dp}, mbs={mbs}); adjust gbs, mbs, or dp"
        )

    if workload.recompute_granularity == "full":
        if workload.recompute_method is None:
            raise ValueError("recompute_granularity='full' requires recompute_method ('uniform' or 'block')")
        if workload.recompute_num_layers is None or workload.recompute_num_layers < 1:
            raise ValueError("recompute_granularity='full' requires recompute_num_layers >= 1")
        if workload.recompute_num_layers > model.architecture.num_layers:
            raise ValueError(
                f"recompute_num_layers={workload.recompute_num_layers} must be <= "
                f"num_layers={model.architecture.num_layers}"
            )


def validate_full_config(model: ModelConfig, parallel: ParallelConfig, workload: Workload) -> None:
    validate_parallel_config(model, parallel)
    validate_workload(model, parallel, workload)


def gradient_accumulation_steps(parallel: ParallelConfig, workload: Workload) -> int:
    """``gbs / (dp * mbs)`` — micro-batches per optimizer step."""
    per_step = parallel.data_parallel_size * workload.micro_batch_size
    if per_step <= 0 or workload.global_batch_size % per_step != 0:
        return 0
    return workload.global_batch_size // per_step


def decompose_rank(global_rank: int, parallel: ParallelConfig) -> RankCoord:
    """Decompose a global rank using Megatron's two RankGenerators.

    Default (non-expert): order ``tp-cp-dp-pp`` (``ep=1``). World size is
    ``tp * cp * dp * pp``.

    Expert: order ``tp-ep-edp-pp`` (``cp=1`` and ``dp=edp=cp*dp/ep``) on the
    same world. ``expert_tp == tp`` (Megatron's default).
    """
    tp = parallel.tensor_model_parallel_size
    cp = parallel.context_parallel_size
    dp = parallel.data_parallel_size
    pp = parallel.pipeline_model_parallel_size
    ep = parallel.expert_model_parallel_size

    world = tp * cp * dp * pp
    if not 0 <= global_rank < world:
        raise ValueError(f"global_rank={global_rank} outside world_size={world}")

    # Default decomposition: tp-cp-dp-pp.
    r = global_rank
    tp_r = r % tp
    r //= tp
    cp_r = r % cp
    r //= cp
    dp_r = r % dp
    r //= dp
    pp_r = r

    # Expert decomposition on the same global rank.
    # Within a PP stage there are ``tp * cp * dp`` ranks, which Megatron's
    # expert RankGenerator re-orders as ``tp * ep * edp``. We assume
    # ``expert_tp == tp`` (Megatron's default).
    if ep <= 1 or parallel.expert_data_parallel_size <= 0:
        ep_r, edp_r = 0, 0
    else:
        within_pp = global_rank - pp_r * (tp * cp * dp)
        pos_after_tp = within_pp // tp
        ep_r = pos_after_tp % ep
        edp_r = pos_after_tp // ep
    return RankCoord(tp=tp_r, cp=cp_r, dp=dp_r, pp=pp_r, ep=ep_r, expert_dp=edp_r)


def layers_per_pp_stage(model: ModelConfig, parallel: ParallelConfig) -> list[int]:
    """Returns the layer count for each PP stage (length == pp_size).

    Assumes :func:`validate_parallel_config` has already been called.
    """
    if parallel.pipeline_model_parallel_layout is not None:
        return list(parallel.pipeline_model_parallel_layout)
    pp = parallel.pipeline_model_parallel_size
    per_stage = model.architecture.num_layers // pp
    return [per_stage] * pp


def partition_for_rank(model: ModelConfig, parallel: ParallelConfig, global_rank: int) -> ModelPartition:
    """Compute which model slice a given global rank owns."""
    coord = decompose_rank(global_rank, parallel)
    layer_counts = layers_per_pp_stage(model, parallel)
    first_layer_idx = sum(layer_counts[: coord.pp])
    pp_last = parallel.pipeline_model_parallel_size - 1
    has_output_projection = coord.pp == pp_last and (model.architecture.untie_embeddings_and_output_weights)
    return ModelPartition(
        num_layers_on_rank=layer_counts[coord.pp],
        first_layer_idx=first_layer_idx,
        has_embedding=coord.pp == 0,
        has_final_norm=coord.pp == pp_last,
        has_output_projection=has_output_projection,
    )
