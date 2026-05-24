"""Workload validation and derived-value computation."""

from __future__ import annotations

import pytest

from projection import ParallelConfig, Trainer, Workload, load_model_config
from projection.parallel.ranks import (
    gradient_accumulation_steps,
    validate_full_config,
    validate_workload,
)


def test_gbs_must_be_divisible_by_dp_times_mbs() -> None:
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig(data_parallel_size=4)
    workload = Workload(seq_length=2048, micro_batch_size=3, global_batch_size=50)
    with pytest.raises(ValueError, match="global_batch_size=50.*divisible by .*=12"):
        validate_workload(model, parallel, workload)


def test_gbs_valid_division() -> None:
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig(data_parallel_size=4)
    workload = Workload(seq_length=2048, micro_batch_size=2, global_batch_size=64)
    validate_workload(model, parallel, workload)


def test_recompute_full_requires_method_and_num_layers() -> None:
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig()
    bad_no_method = Workload(
        seq_length=2048, micro_batch_size=1, global_batch_size=64, recompute_granularity="full"
    )
    with pytest.raises(ValueError, match="recompute_method"):
        validate_workload(model, parallel, bad_no_method)

    bad_no_num = Workload(
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=64,
        recompute_granularity="full",
        recompute_method="uniform",
    )
    with pytest.raises(ValueError, match="recompute_num_layers"):
        validate_workload(model, parallel, bad_no_num)

    ok = Workload(
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=64,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=8,
    )
    validate_workload(model, parallel, ok)


def test_gradient_accumulation_steps_is_gbs_over_dp_mbs() -> None:
    parallel = ParallelConfig(data_parallel_size=4)
    workload = Workload(seq_length=2048, micro_batch_size=2, global_batch_size=64)
    assert gradient_accumulation_steps(parallel, workload) == 8


def test_expert_data_parallel_size_for_moe() -> None:
    parallel = ParallelConfig(data_parallel_size=32, expert_model_parallel_size=8)
    assert parallel.expert_data_parallel_size == 4


def test_parallel_dims_must_be_positive() -> None:
    with pytest.raises(ValueError, match="tensor_model_parallel_size.*>= 1"):
        ParallelConfig(tensor_model_parallel_size=0)


def test_recompute_block_caps_at_layers_per_pp_rank() -> None:
    """``block`` mode: recompute_num_layers * num_chunks_per_rank must fit per rank."""
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig(pipeline_model_parallel_size=4)
    # num_chunks_per_rank = pp * vpp = 4 * 1 = 4. recompute_num_layers=4 * 4 = 16 > 8.
    bad = Workload(
        seq_length=1024,
        micro_batch_size=1,
        global_batch_size=64,
        recompute_granularity="full",
        recompute_method="block",
        recompute_num_layers=4,
    )
    with pytest.raises(ValueError, match="exceeds layers per PP rank"):
        validate_workload(model, parallel, bad)


def test_recompute_block_with_vpp_uses_pp_times_vpp_chunks() -> None:
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig(pipeline_model_parallel_size=4, virtual_pipeline_model_parallel_size=2)
    # num_chunks_per_rank = pp*vpp = 8. recompute_num_layers=2 → 2*8=16 > 8 layers/rank.
    bad = Workload(
        seq_length=1024,
        micro_batch_size=1,
        global_batch_size=64,
        recompute_granularity="full",
        recompute_method="block",
        recompute_num_layers=2,
    )
    with pytest.raises(ValueError, match="exceeds layers per PP rank"):
        validate_workload(model, parallel, bad)


def test_tp_must_divide_num_query_groups_gqa() -> None:
    """Llama 3.1 8B has 32 heads, 8 KV groups → TP=16 fails on GQA group count (8 % 16 != 0)."""
    from projection.parallel.ranks import validate_parallel_config

    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig(tensor_model_parallel_size=16)
    with pytest.raises(
        ValueError,
        match=r"num_query_groups=8 \(GQA\) must be divisible by tensor_model_parallel_size=16",
    ):
        validate_parallel_config(model, parallel)


def test_tp_must_divide_num_attention_heads_no_gqa() -> None:
    """DSV2-Lite has 16 heads, no GQA → TP=8 fine, TP=32 fails on head count."""
    from projection.parallel.ranks import validate_parallel_config

    model = load_model_config("deepseek_v2_lite")
    validate_parallel_config(model, ParallelConfig(tensor_model_parallel_size=8))

    parallel_bad = ParallelConfig(tensor_model_parallel_size=32)
    with pytest.raises(ValueError, match="num_attention_heads=16"):
        validate_parallel_config(model, parallel_bad)


def test_etp_must_divide_moe_ffn_hidden_size() -> None:
    """DSV2-Lite has moe_ffn_hidden_size=1408 (not divisible by 3) → ETP=3 rejected."""
    from projection.parallel.ranks import validate_parallel_config

    model = load_model_config("deepseek_v2_lite")
    parallel = ParallelConfig(
        tensor_model_parallel_size=1,
        moe_folding=True,
        expert_tensor_parallel_size=3,
        expert_model_parallel_size=1,
    )
    with pytest.raises(ValueError, match="moe_ffn_hidden_size=1408"):
        validate_parallel_config(model, parallel)


def test_seq_length_must_be_multiple_of_2_cp() -> None:
    """CP > 1 requires seq_length % (2*CP) == 0 (ring attention splits into 2*CP halves)."""
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig(context_parallel_size=4)
    bad = Workload(seq_length=2050, micro_batch_size=1, global_batch_size=64)
    with pytest.raises(ValueError, match=r"multiple of 2\*context_parallel_size=8"):
        validate_workload(model, parallel, bad)

    ok = Workload(seq_length=2048, micro_batch_size=1, global_batch_size=64)
    validate_workload(model, parallel, ok)


def test_vpp_requires_pp_greater_than_1() -> None:
    from projection.parallel.ranks import validate_parallel_config

    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig(pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=2)
    with pytest.raises(ValueError, match="requires pipeline_model_parallel_size > 1"):
        validate_parallel_config(model, parallel)


def test_recompute_uniform_does_not_check_recompute_num_layers_bound() -> None:
    """``uniform`` mode recomputes everything; no per-rank cap check."""
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig(pipeline_model_parallel_size=4)
    ok = Workload(
        seq_length=1024,
        micro_batch_size=1,
        global_batch_size=64,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=8,
    )
    validate_workload(model, parallel, ok)


def test_selective_recompute_reduces_activation_bytes_without_sp() -> None:
    """Selective should save the attention-block activations even when SP is off."""
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig()
    no_re = Workload(seq_length=2048, micro_batch_size=1, global_batch_size=64)
    sel = Workload(
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=64,
        recompute_granularity="selective",
    )
    no_re_bytes = Trainer(model, parallel, no_re).report().memory.activation_bytes
    sel_bytes = Trainer(model, parallel, sel).report().memory.activation_bytes
    assert sel_bytes < no_re_bytes


def test_selective_recompute_reduces_activation_bytes_with_sp() -> None:
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig(tensor_model_parallel_size=2, sequence_parallel=True)
    no_re = Workload(seq_length=2048, micro_batch_size=1, global_batch_size=64)
    sel = Workload(
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=64,
        recompute_granularity="selective",
    )
    no_re_bytes = Trainer(model, parallel, no_re).report().memory.activation_bytes
    sel_bytes = Trainer(model, parallel, sel).report().memory.activation_bytes
    assert sel_bytes < no_re_bytes


def test_sp_alone_reduces_activation_bytes() -> None:
    """SP alone (no selective) should already reduce activations vs. no-SP baseline."""
    model = load_model_config("llama3.1_8B")
    no_sp = ParallelConfig(tensor_model_parallel_size=2, sequence_parallel=False)
    sp_on = ParallelConfig(tensor_model_parallel_size=2, sequence_parallel=True)
    w = Workload(seq_length=2048, micro_batch_size=1, global_batch_size=64)
    a = Trainer(model, no_sp, w).report().memory.activation_bytes
    b = Trainer(model, sp_on, w).report().memory.activation_bytes
    assert b < a


def test_full_recompute_reduces_activation_bytes() -> None:
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig()
    no_re = Workload(seq_length=2048, micro_batch_size=1, global_batch_size=64)
    full_re = Workload(
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=64,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )
    no_re_bytes = Trainer(model, parallel, no_re).report().memory.activation_bytes
    full_re_bytes = Trainer(model, parallel, full_re).report().memory.activation_bytes
    assert full_re_bytes < no_re_bytes


def test_partial_full_recompute_is_in_between() -> None:
    """Use ``block`` method to control partial recompute (uniform always recomputes all)."""
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig()
    no_re = Workload(seq_length=2048, micro_batch_size=1, global_batch_size=64)
    partial = Workload(
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=64,
        recompute_granularity="full",
        recompute_method="block",
        recompute_num_layers=16,
    )
    full = Workload(
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=64,
        recompute_granularity="full",
        recompute_method="block",
        recompute_num_layers=32,
    )
    no_re_bytes = Trainer(model, parallel, no_re).report().memory.activation_bytes
    partial_bytes = Trainer(model, parallel, partial).report().memory.activation_bytes
    full_bytes = Trainer(model, parallel, full).report().memory.activation_bytes
    assert full_bytes < partial_bytes < no_re_bytes


def test_validate_full_config_combines_both() -> None:
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig(pipeline_model_parallel_size=5)
    workload = Workload(seq_length=2048, micro_batch_size=1, global_batch_size=64)
    with pytest.raises(ValueError, match="not divisible"):
        validate_full_config(model, parallel, workload)
