"""Per-rank activation memory model.

Formulas are pulled from Megatron-LM's
``megatron/training/theoretical_memory_usage.py`` (the canonical "compute
expected memory before launch" implementation). Key facts mirrored here:

- **PP makes ranks unequal.** With non-interleaved 1F1B and ``num_microbatches``
  in flight, the k-th PP stage (0-indexed) holds activations for
  ``min(num_microbatches, pp_size - k)`` microbatches. Rank 0 is the worst.
- **VPP adds a uniform penalty.** With ``vpp_size`` virtual stages,
  the in-flight count per rank scales by
  ``1 + (pp_size - 1) / (pp_size * vpp_size)`` (Megatron's
  ``interleaved_schedule_memory_penalty``).
- **First PP rank carries the embedding + dropout in-flight overhead** (one
  copy per in-flight microbatch).
- **Last PP rank carries the final norm + output projection + CE loss**.
- **TP / SP / CP** scale the per-layer term.

The per-layer base formula picks between Megatron's two variants:

- SP + selective recompute: ``sbh * 18 + 4 * sb * ffn`` (Megatron's
  ``compute_activation_memory``), then divided by TP at the end.
- Otherwise:                 ``sbh * (10 + 24 / TP)`` (Megatron's
  ``compute_activation_memory_without_sp``); no further TP division.

Full recomputation overrides the chosen layers to ``2 * sbh`` (just the layer
input stashed), partial-full recomputation mixes the two.
"""

from __future__ import annotations

import math

from projection.configs import ModelConfig, ParallelConfig, Workload
from projection.core.modules import padded_vocab_size


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


def _per_layer_sp_selective(model: ModelConfig, workload: Workload, parallel: ParallelConfig) -> int:
    """``sbh * 18 + 4 * sb * ffn``. Will be TP-divided at the end."""
    s = workload.seq_length
    b = workload.micro_batch_size
    h = model.architecture.hidden_size
    ffn = model.architecture.ffn_hidden_size
    cp = parallel.context_parallel_size
    return (18 * s * b * h + 4 * s * b * ffn) // max(1, cp)


def _per_layer_without_sp(model: ModelConfig, workload: Workload, parallel: ParallelConfig) -> int:
    """``sbh * (10 + 24 / t)``. TP factor already baked in."""
    s = workload.seq_length
    b = workload.micro_batch_size
    h = model.architecture.hidden_size
    t = parallel.tensor_model_parallel_size
    cp = parallel.context_parallel_size
    return ((10 * t + 24) * s * b * h) // (t * max(1, cp))


def _per_layer_full_recompute(model: ModelConfig, workload: Workload, parallel: ParallelConfig) -> int:
    """``2 * sbh`` — just the layer input."""
    s = workload.seq_length
    b = workload.micro_batch_size
    h = model.architecture.hidden_size
    cp = parallel.context_parallel_size
    return (2 * s * b * h) // max(1, cp)


def _use_sp_selective(workload: Workload, parallel: ParallelConfig) -> bool:
    return parallel.sequence_parallel and workload.recompute_granularity == "selective"


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
) -> int:
    """Activation bytes for one PP rank, accounting for in-flight microbatches, embedding, and output."""
    in_flight = in_flight_microbatches(parallel, pp_rank, num_microbatches)
    s = workload.seq_length
    b = workload.micro_batch_size
    h = model.architecture.hidden_size
    t = parallel.tensor_model_parallel_size

    use_sp = _use_sp_selective(workload, parallel)
    normal_per_layer = (
        _per_layer_sp_selective(model, workload, parallel)
        if use_sp
        else _per_layer_without_sp(model, workload, parallel)
    )

    if workload.recompute_granularity == "full" and num_layers_on_rank > 0:
        # Recompute knobs are global (``recompute_num_layers`` is across the whole
        # model). Apportion the recomputed layers to this rank proportionally.
        total_recomputed = workload.recompute_num_layers or model.architecture.num_layers
        total_layers = max(1, model.architecture.num_layers)
        local_recomputed = min(
            num_layers_on_rank,
            (total_recomputed * num_layers_on_rank + total_layers - 1) // total_layers,
        )
        local_kept = num_layers_on_rank - local_recomputed
        full_per_layer = _per_layer_full_recompute(model, workload, parallel)
        layer_bytes = local_recomputed * full_per_layer + local_kept * normal_per_layer
    else:
        layer_bytes = num_layers_on_rank * normal_per_layer

    total = layer_bytes * in_flight

    if is_first_pp:
        # Embedding-input + dropout: one copy per in-flight microbatch on rank 0.
        emb_per_mb = 8 * s * b + s * b * h
        total += emb_per_mb * in_flight

    if is_last_pp:
        # Final norm output + logits + CE: one microbatch at peak on the last stage.
        vocab = padded_vocab_size(model.architecture, t)
        out_bytes = (4 * s * b * h + 4 * s * b * vocab) // t
        total += out_bytes

    if use_sp:
        total = total // t

    return total
