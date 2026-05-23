"""Gold-standard tests for the MoE / MLA path (DeepSeek-V2-Lite)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from projection import ParallelConfig, Precision, Trainer, Workload, load_model_config
from projection.core.modules import MoEModule

FIXTURE = (
    Path(__file__).resolve().parent.parent / "fixtures" / "deepseek_v2_lite" / "default_bf16_tp1_pp1_ep1.json"
)


def _setup() -> tuple[Trainer, dict]:
    with open(FIXTURE, "r", encoding="utf-8") as f:
        fixture = json.load(f)
    model = load_model_config(fixture["model"])
    parallel = ParallelConfig(
        precision=Precision(fixture["parallel"]["precision"]),
        tensor_model_parallel_size=fixture["parallel"]["tensor_model_parallel_size"],
        pipeline_model_parallel_size=fixture["parallel"]["pipeline_model_parallel_size"],
        data_parallel_size=fixture["parallel"]["data_parallel_size"],
        context_parallel_size=fixture["parallel"]["context_parallel_size"],
        expert_model_parallel_size=fixture["parallel"]["expert_model_parallel_size"],
        sequence_parallel=fixture["parallel"]["sequence_parallel"],
    )
    workload = Workload(**fixture["workload"])
    return Trainer(model, parallel, workload, global_rank=0), fixture


def test_deepseek_v2_lite_total_param_count() -> None:
    trainer, fixture = _setup()
    assert trainer.report().param_count == fixture["expected"]["param_count_total"]


def test_deepseek_v2_lite_param_breakdown() -> None:
    trainer, fixture = _setup()
    breakdown = {pb.name: pb.count for pb in trainer.report().param_breakdown}
    assert breakdown == fixture["expected"]["param_breakdown"]


def test_deepseek_v2_lite_moe_layer_components() -> None:
    """The routed expert size and MoE block size are derivable from the YAML alone."""
    model = load_model_config("deepseek_v2_lite")
    moe_block = MoEModule(model)
    assert moe_block.routed_expert_param_count() == 8_650_752
    assert moe_block.total_full_param_count() == 571_080_704


def test_ep_shards_routed_experts() -> None:
    model = load_model_config("deepseek_v2_lite")
    moe_block = MoEModule(model)
    full = moe_block.param_count(ep_size=1)
    sharded = moe_block.param_count(ep_size=8)
    # Routed: 64 experts × 8.65M ÷ 8 = 8 experts × 8.65M = 69M
    # Shared: 2 × 8.65M = 17.3M; Gate: 131K (replicated on every EP rank)
    expected_routed_local = 8 * 8_650_752
    expected_shared = 2 * 8_650_752
    expected_gate = 2048 * 64
    assert sharded == expected_routed_local + expected_shared + expected_gate
    assert full > sharded


def test_ep_must_divide_num_experts() -> None:
    model = load_model_config("deepseek_v2_lite")
    moe = MoEModule(model)
    with pytest.raises(ValueError, match="divisible"):
        moe.param_count(ep_size=7)


def test_pp_splits_layers_with_dense_first() -> None:
    """With PP=3 and 27 layers, the first PP rank owns the dense layer + 8 MoE layers."""
    model = load_model_config("deepseek_v2_lite")
    parallel = ParallelConfig(pipeline_model_parallel_size=3, expert_model_parallel_size=1)
    workload = Workload(seq_length=4096, micro_batch_size=1, global_batch_size=64)

    # PP rank 0: layers [0..8], includes the 1 dense + 8 MoE
    rank0 = Trainer(model, parallel, workload, global_rank=0)
    assert rank0.transformer_model.block.num_dense_on_rank == 1
    assert rank0.transformer_model.block.num_moe_on_rank == 8

    # PP rank 1: layers [9..17], all MoE
    rank1 = Trainer(model, parallel, workload, global_rank=parallel.tensor_model_parallel_size)
    assert rank1.transformer_model.block.num_dense_on_rank == 0
    assert rank1.transformer_model.block.num_moe_on_rank == 9

    # PP rank 2 (last): layers [18..26], all MoE, has output_projection
    rank2 = Trainer(model, parallel, workload, global_rank=2 * parallel.tensor_model_parallel_size)
    assert rank2.transformer_model.block.num_dense_on_rank == 0
    assert rank2.transformer_model.block.num_moe_on_rank == 9
    assert rank2.partition.has_output_projection
