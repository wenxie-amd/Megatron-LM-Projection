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

    if vpp and vpp > 1 and pp <= 1:
        raise ValueError("virtual_pipeline_model_parallel_size > 1 requires pipeline_model_parallel_size > 1")

    # Megatron requires the partitionable dimensions to be divisible by TP / ETP.
    tp = parallel.tensor_model_parallel_size
    att = model.attention
    if att.kv_channels is None and not att.use_mla:
        if model.architecture.hidden_size % att.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={model.architecture.hidden_size} must be divisible by "
                f"num_attention_heads={att.num_attention_heads} (or set kv_channels explicitly)"
            )
    if att.num_attention_heads % tp != 0:
        raise ValueError(
            f"num_attention_heads={att.num_attention_heads} must be divisible by "
            f"tensor_model_parallel_size={tp}"
        )
    if att.num_query_groups is not None and att.num_query_groups % tp != 0:
        raise ValueError(
            f"num_query_groups={att.num_query_groups} (GQA) must be divisible by "
            f"tensor_model_parallel_size={tp}"
        )
    ffn = model.architecture.ffn_hidden_size
    if ffn % tp != 0:
        raise ValueError(f"ffn_hidden_size={ffn} must be divisible by tensor_model_parallel_size={tp}")
    if model.moe.enabled:
        etp = parallel.effective_expert_tensor_parallel_size
        moe_ffn = model.moe.moe_ffn_hidden_size
        if moe_ffn % etp != 0:
            raise ValueError(
                f"moe_ffn_hidden_size={moe_ffn} must be divisible by " f"expert_tensor_parallel_size={etp}"
            )

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
        etp = parallel.effective_expert_tensor_parallel_size
        pp = parallel.pipeline_model_parallel_size
        world = parallel.world_size
        denom = etp * parallel.expert_model_parallel_size * pp
        # EDP must be a clean integer (≥1) or a clean reciprocal (1/n) so each
        # rank covers a whole number of expert slices / copies.
        if denom == 0 or world == 0 or (world % denom != 0 and denom % world != 0):
            raise ValueError(
                f"world_size={world} and ETP*EP*PP={etp}*{parallel.expert_model_parallel_size}*{pp}={denom} must "
                f"divide one another so EDP = world/(ETP*EP*PP) is either an integer (≥1) or a clean reciprocal (1/n)"
            )

    if parallel.moe_folding and not model.moe.enabled:
        raise ValueError("moe_folding can only be enabled for MoE models")

    if layout is not None:
        expected = pp * (vpp or 1)
        if len(layout) != expected:
            raise ValueError(
                f"pipeline_model_parallel_layout must have pp*vpp={expected} entries (got {len(layout)})"
            )
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

    cp = parallel.context_parallel_size
    if cp > 1 and workload.seq_length % (cp * 2) != 0:
        raise ValueError(
            f"seq_length={workload.seq_length} must be a multiple of 2*context_parallel_size={2 * cp} "
            f"(context parallel splits each sequence into 2*CP balanced halves)"
        )

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
        # In ``block`` mode ``recompute_num_layers`` is recomputed per chunk
        # (``num_chunks_per_rank = pp*vpp`` in direct mode, ``len(layout)`` in
        # layout mode). If ``recompute_num_layers * num_chunks_per_rank``
        # exceeds the number of layers on a rank, the activation memory model
        # silently clamps to "recompute every layer on the rank" — same as
        # ``method=uniform`` — so we deliberately *don't* reject that case.


def validate_full_config(model: ModelConfig, parallel: ParallelConfig, workload: Workload) -> None:
    validate_parallel_config(model, parallel)
    validate_workload(model, parallel, workload)


def gradient_accumulation_steps(parallel: ParallelConfig, workload: Workload) -> int:
    """``gbs / (dp * mbs)`` — micro-batches per optimizer step."""
    per_step = parallel.data_parallel_size * workload.micro_batch_size
    if per_step <= 0 or workload.global_batch_size % per_step != 0:
        return 0
    return workload.global_batch_size // per_step


def num_chunks_per_rank(parallel: ParallelConfig) -> int:
    """Number of model chunks a single PP rank owns (== ``vpp``).

    Each ``TransformerBlock`` in Megatron is one virtual chunk, and a PP rank
    holds ``vpp`` of them. With ``recompute_method='block'``,
    ``recompute_num_layers`` is recomputed *per chunk*, so the total recomputed
    layers attributed to one PP rank is
    ``recompute_num_layers * num_chunks_per_rank``.

    - Direct (PP + VPP) mode: ``vpp``.
    - Layout mode: ``len(layout) // pp`` — the layout lists all ``pp * vpp``
      chunks globally and each PP rank owns ``vpp`` consecutive ones.
    """
    pp = parallel.pipeline_model_parallel_size
    if parallel.pipeline_model_parallel_layout is not None:
        return max(1, len(parallel.pipeline_model_parallel_layout) // max(1, pp))
    return parallel.virtual_pipeline_model_parallel_size or 1


def total_recompute_layers_on_rank(
    model: ModelConfig, parallel: ParallelConfig, workload: Workload, pp_rank: int
) -> int:
    """Per-PP-rank recomputed layer count, capped at this rank's layer count.

    ``recompute_method='uniform'`` recomputes every layer (the layer count
    on this rank).

    ``recompute_method='block'`` recomputes ``recompute_num_layers`` per chunk,
    where the chunk multiplier is :func:`num_chunks_per_rank`.
    """
    if workload.recompute_granularity != "full" or not workload.recompute_num_layers:
        return 0
    layer_counts = layers_per_pp_stage(model, parallel)
    if not 0 <= pp_rank < len(layer_counts):
        return 0
    on_rank = layer_counts[pp_rank]
    if workload.recompute_method == "uniform":
        return on_rank
    return min(on_rank, workload.recompute_num_layers * num_chunks_per_rank(parallel))


def total_layers_on_rank(model: ModelConfig, parallel: ParallelConfig, pp_rank: int) -> int:
    layer_counts = layers_per_pp_stage(model, parallel)
    if not 0 <= pp_rank < len(layer_counts):
        return 0
    return layer_counts[pp_rank]


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
        # Expert RankGenerator orders ``tp_e-cp(=1)-ep-edp-pp`` and uses
        # ``tp_e = effective_expert_tensor_parallel_size`` (= TP when folding is off).
        etp = parallel.effective_expert_tensor_parallel_size
        within_pp = global_rank - pp_r * (tp * cp * dp)
        pos_after_tp = within_pp // max(1, etp)
        ep_r = pos_after_tp % ep
        edp_r = pos_after_tp // ep
    return RankCoord(tp=tp_r, cp=cp_r, dp=dp_r, pp=pp_r, ep=ep_r, expert_dp=edp_r)


def layers_per_pp_stage(model: ModelConfig, parallel: ParallelConfig) -> list[int]:
    """Layer count owned by each PP stage (length == pp_size).

    For layout mode with ``vpp > 1``, each PP rank owns ``vpp`` consecutive
    chunks from the layout; the per-stage count is the sum of those chunks.
    Assumes :func:`validate_parallel_config` has already been called.
    """
    pp = parallel.pipeline_model_parallel_size
    vpp = parallel.virtual_pipeline_model_parallel_size or 1
    layout = parallel.pipeline_model_parallel_layout
    if layout is not None:
        if vpp <= 1:
            return list(layout)
        return [sum(layout[i * vpp : (i + 1) * vpp]) for i in range(pp)]
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
