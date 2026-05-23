"""Conflict detection for the optimizer / sharding single-select."""

from __future__ import annotations

import pytest

from projection import ParallelConfig, Trainer, Workload, load_model_config
from projection.configs import OptimizerKind
from projection.parallel.ranks import validate_parallel_config


def test_torch_fsdp2_incompatible_with_pp() -> None:
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig(optimizer_kind=OptimizerKind.TORCH_FSDP2, pipeline_model_parallel_size=2)
    with pytest.raises(ValueError, match="torch_fsdp2.*pipeline"):
        validate_parallel_config(model, parallel)


def test_megatron_fsdp_incompatible_with_pp() -> None:
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig(optimizer_kind=OptimizerKind.MEGATRON_FSDP, pipeline_model_parallel_size=4)
    with pytest.raises(ValueError, match="megatron_fsdp.*pipeline"):
        validate_parallel_config(model, parallel)


def test_torch_fsdp2_incompatible_with_vpp() -> None:
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig(
        optimizer_kind=OptimizerKind.TORCH_FSDP2, virtual_pipeline_model_parallel_size=2
    )
    with pytest.raises(ValueError, match="torch_fsdp2.*virtual"):
        validate_parallel_config(model, parallel)


def test_fsdp_shards_params_across_dp() -> None:
    model = load_model_config("llama3.1_8B")
    workload = Workload(seq_length=2048, micro_batch_size=1, global_batch_size=64)

    no_dp = ParallelConfig(optimizer_kind=OptimizerKind.TORCH_FSDP2, data_parallel_size=1)
    with_dp = ParallelConfig(optimizer_kind=OptimizerKind.TORCH_FSDP2, data_parallel_size=4)

    no_dp_bytes = Trainer(model, no_dp, workload).report().memory.param_bytes
    with_dp_bytes = Trainer(model, with_dp, workload).report().memory.param_bytes
    assert with_dp_bytes == no_dp_bytes // 4


def test_distributed_optimizer_does_not_shard_params_by_dp() -> None:
    model = load_model_config("llama3.1_8B")
    workload = Workload(seq_length=2048, micro_batch_size=1, global_batch_size=64)

    no_dp = ParallelConfig(optimizer_kind=OptimizerKind.DISTRIBUTED_OPTIMIZER, data_parallel_size=1)
    with_dp = ParallelConfig(optimizer_kind=OptimizerKind.DISTRIBUTED_OPTIMIZER, data_parallel_size=4)

    assert (
        Trainer(model, no_dp, workload).report().memory.param_bytes
        == Trainer(model, with_dp, workload).report().memory.param_bytes
    )


def test_moe_ep_must_divide_num_routed_experts() -> None:
    model = load_model_config("deepseek_v2_lite")
    parallel = ParallelConfig(expert_model_parallel_size=7)
    with pytest.raises(ValueError, match="num_routed_experts.*divisible"):
        validate_parallel_config(model, parallel)
