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


def test_recompute_num_layers_caps_at_num_layers() -> None:
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig()
    bad = Workload(
        seq_length=1024,
        micro_batch_size=1,
        global_batch_size=64,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=100,
    )
    with pytest.raises(ValueError, match="<= num_layers=32"):
        validate_workload(model, parallel, bad)


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
        recompute_num_layers=32,
    )
    no_re_bytes = Trainer(model, parallel, no_re).report().memory.activation_bytes
    full_re_bytes = Trainer(model, parallel, full_re).report().memory.activation_bytes
    assert full_re_bytes < no_re_bytes


def test_partial_full_recompute_is_in_between() -> None:
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig()
    no_re = Workload(seq_length=2048, micro_batch_size=1, global_batch_size=64)
    partial = Workload(
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=64,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=16,
    )
    full = Workload(
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=64,
        recompute_granularity="full",
        recompute_method="uniform",
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
