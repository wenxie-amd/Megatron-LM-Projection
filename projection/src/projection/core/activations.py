"""Per-rank activation memory model.

PP / VPP / embedding / output handling follows Megatron-LM's
``megatron/training/theoretical_memory_usage.py``. The per-layer formula goes
beyond what Megatron's helper does so that **selective recompute reduces the
estimate whether or not SP is on** (Megatron's helper bundles them):

============== =============== ==========================================
SP             selective       per-layer formula (bytes)
============== =============== ==========================================
off            off             ``sbh * (10 + 24/TP)``
off            selective       ``sbh * (10 + 13/TP)`` (drops 11sbh/TP)
on             off             ``sbh * 34 / TP``  (Korthikanti SP-only)
on             selective       ``sbh * 18 + 4sb*ffn``, then ``/ TP``
============== =============== ==========================================

Full recomputation overrides chosen layers to ``2 * sbh`` (just the layer
input); partial-full recomputation mixes the two.

- **PP makes ranks unequal**: with non-interleaved 1F1B and
  ``num_microbatches`` in flight, the k-th PP stage holds activations for
  ``min(num_microbatches, pp_size - k)`` microbatches.
- **VPP** adds a uniform penalty
  ``1 + (pp_size - 1) / (pp_size * vpp_size)``.
- **First PP rank** carries embedding + dropout overhead;
  **last PP rank** carries final norm + logits + CE loss.
"""

from __future__ import annotations

import math

from projection.configs import ModelConfig, ParallelConfig, Workload
from projection.core.modules import padded_vocab_size
from projection.parallel.ranks import num_chunks_per_rank


def in_flight_microbatches(parallel: ParallelConfig, pp_rank: int, num_microbatches: int | None) -> int:
    """Microbatches whose activations live on rank ``pp_rank`` at peak.

    Mirrors Megatron's logic in ``compute_activation_memory`` so the worst-case
    rank (PP=0) matches what Megatron reports.
    """
    pp = parallel.pipeline_model_parallel_size
    vpp = parallel.virtual_pipeline_model_parallel_size or 1
    if pp == 1:
        return 1
    if vpp > 1:
        penalty = 1.0 + (pp - 1.0) / (pp * vpp)
        return int(math.ceil(penalty * pp))
    # Non-interleaved 1F1B
    if num_microbatches is None:
        in_flight_at_rank0 = pp
    else:
        in_flight_at_rank0 = min(num_microbatches, pp)
    # Rank 0 holds the most; each subsequent rank holds one fewer.
    return max(1, min(in_flight_at_rank0, pp - pp_rank))


def interleaved_penalty(parallel: ParallelConfig) -> float:
    pp = parallel.pipeline_model_parallel_size
    vpp = parallel.virtual_pipeline_model_parallel_size or 1
    if pp <= 1 or vpp <= 1:
        return 1.0
    return 1.0 + (pp - 1.0) / (pp * vpp)


def _per_layer_full_recompute(model: ModelConfig, workload: Workload, parallel: ParallelConfig) -> int:
    """``2 * sbh`` — just the layer input."""
    s = workload.seq_length
    b = workload.micro_batch_size
    h = model.architecture.hidden_size
    cp = parallel.context_parallel_size
    return (2 * s * b * h) // max(1, cp)


def _per_layer_normal(model: ModelConfig, workload: Workload, parallel: ParallelConfig) -> int:
    """Per-layer activation bytes for granularity 'none' or 'selective'.

    Covers all four combinations of {SP on/off} × {selective on/off}; see the
    table in the module docstring.
    """
    s = workload.seq_length
    b = workload.micro_batch_size
    h = model.architecture.hidden_size
    ffn = model.architecture.ffn_hidden_size
    t = parallel.tensor_model_parallel_size
    cp = max(1, parallel.context_parallel_size)
    sp = parallel.sequence_parallel
    selective = workload.recompute_granularity == "selective"

    if sp and selective:
        return (18 * s * b * h + 4 * s * b * ffn) // (t * cp)
    if sp:
        return (34 * s * b * h) // (t * cp)
    if selective:
        return ((10 * t + 13) * s * b * h) // (t * cp)
    return ((10 * t + 24) * s * b * h) // (t * cp)


def total_activation_bytes_for_rank(
    model: ModelConfig,
    workload: Workload,
    parallel: ParallelConfig,
    num_layers_on_rank: int,
    *,
    pp_rank: int,
    is_first_pp: bool,
    is_last_pp: bool,
    num_microbatches: int | None,
    num_dense_layers_on_rank: int | None = None,
    num_moe_layers_on_rank: int | None = None,
) -> int:
    """Activation bytes for one PP rank.

    When ``parallel.moe_folding`` is on and MoE layers are present, dense layers
    use the attention strategy (TP, SP, CP) while MoE layers use the expert
    strategy (ETP). Without folding (or no MoE), both share the same per-layer
    formula. The split is driven by ``num_dense_layers_on_rank`` /
    ``num_moe_layers_on_rank`` when provided; otherwise everything is treated
    as a homogeneous block of ``num_layers_on_rank`` layers.
    """
    in_flight = in_flight_microbatches(parallel, pp_rank, num_microbatches)
    s = workload.seq_length
    b = workload.micro_batch_size
    h = model.architecture.hidden_size
    t = parallel.tensor_model_parallel_size

    folding_active = (
        parallel.moe_folding
        and model.moe.enabled
        and num_moe_layers_on_rank
        and num_dense_layers_on_rank is not None
    )
    if folding_active:
        nd = num_dense_layers_on_rank or 0
        nm = num_moe_layers_on_rank or 0
    else:
        nd = num_layers_on_rank
        nm = 0

    # Per-layer base bytes — dense uses the attention strategy (TP),
    # MoE uses the expert strategy (ETP). When ETP == TP both are equal.
    dense_per_layer = _per_layer_normal(model, workload, parallel)
    moe_per_layer = (
        _per_layer_normal(
            model, workload, _parallel_swap_tp(parallel, parallel.effective_expert_tensor_parallel_size)
        )
        if folding_active
        else dense_per_layer
    )

    # SP+selective is the only path that needs a final /TP at the end (because
    # the formula uses raw sbh × 18 + 4·sb·ffn without TP baked in).
    final_tp_divide = parallel.sequence_parallel and workload.recompute_granularity == "selective"

    def _layer_bytes(num_layers: int, per_layer: int) -> int:
        if num_layers <= 0:
            return 0
        if workload.recompute_granularity == "full":
            if workload.recompute_method == "uniform":
                local_recomputed = num_layers
            else:
                per_chunk = workload.recompute_num_layers or 0
                local_recomputed = min(num_layers, per_chunk * num_chunks_per_rank(parallel))
            local_kept = num_layers - local_recomputed
            full_per_layer = _per_layer_full_recompute(model, workload, parallel)
            return local_recomputed * full_per_layer + local_kept * per_layer
        return num_layers * per_layer

    layer_bytes = _layer_bytes(nd, dense_per_layer) + _layer_bytes(nm, moe_per_layer)
    total = layer_bytes * in_flight

    if is_first_pp:
        emb_per_mb = 8 * s * b + s * b * h
        total += emb_per_mb * in_flight

    if is_last_pp:
        vocab = padded_vocab_size(model.architecture, t)
        out_bytes = (4 * s * b * h + 4 * s * b * vocab) // t
        total += out_bytes

    if final_tp_divide:
        total = total // t

    return total


def _parallel_swap_tp(parallel: ParallelConfig, new_tp: int) -> ParallelConfig:
    """Return a shallow copy of ``parallel`` with ``tensor_model_parallel_size`` overridden.

    Used so the MoE-side activation can reuse :func:`_per_layer_normal` with the
    expert TP value, without restructuring the formula.
    """
    return parallel.model_copy(update={"tensor_model_parallel_size": max(1, new_tp)})
