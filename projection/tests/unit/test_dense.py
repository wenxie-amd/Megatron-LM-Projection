"""Gold-standard tests for the dense Llama-style path.

The total parameter count for Llama 3.1 8B (8,030,261,248) is taken from Meta's
public spec and is independently verifiable. The other expected numbers in the
fixture are derived from documented Megatron formulas and will be regenerated
from real ``megatron.core`` by ``tools/gen_fixtures.py`` once that's run on a
real Megatron environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from projection import ParallelConfig, Precision, Trainer, Workload, load_model_config
from projection.parallel.ranks import (
    decompose_rank,
    layers_per_pp_stage,
    partition_for_rank,
    validate_parallel_config,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _load_fixture(name: str) -> dict:
    with open(FIXTURES / name, "r", encoding="utf-8") as f:
        return json.load(f)


def _default_setup() -> tuple[Trainer, dict]:
    fixture = _load_fixture("llama3_1_8b/default_bf16_tp1_pp1.json")
    model = load_model_config(fixture["model"])
    parallel = ParallelConfig(
        precision=Precision(fixture["parallel"]["precision"]),
        tensor_model_parallel_size=fixture["parallel"]["tensor_model_parallel_size"],
        pipeline_model_parallel_size=fixture["parallel"]["pipeline_model_parallel_size"],
        data_parallel_size=fixture["parallel"]["data_parallel_size"],
        context_parallel_size=fixture["parallel"]["context_parallel_size"],
        sequence_parallel=fixture["parallel"]["sequence_parallel"],
    )
    workload = Workload(**fixture["workload"])
    return Trainer(model, parallel, workload, global_rank=0), fixture


def test_llama3_1_8b_total_param_count_matches_meta_spec() -> None:
    """Pinned to Meta's published 8.03B parameter count."""
    trainer, _ = _default_setup()
    assert trainer.report().param_count == 8_030_261_248


def test_llama3_1_8b_param_breakdown() -> None:
    trainer, fixture = _default_setup()
    breakdown = {pb.name: pb.count for pb in trainer.report().param_breakdown}
    assert breakdown == fixture["expected"]["param_breakdown"]


def test_llama3_1_8b_rank_0_memory_matches_fixture() -> None:
    trainer, fixture = _default_setup()
    expected = fixture["expected"]["rank_0_memory_bytes"]
    mem = trainer.report().memory
    assert mem.param_bytes == expected["param_bytes"]
    assert mem.activation_bytes == expected["activation_bytes"]
    assert mem.optimizer.grad_buffer_bytes == expected["optimizer_grad_buffer_bytes"]
    assert mem.optimizer.main_param_bytes == expected["optimizer_main_param_bytes"]
    assert mem.optimizer.state_bytes == expected["optimizer_state_bytes"]
    assert mem.total_bytes == expected["total_bytes"]


def test_pp_partition_splits_layers_evenly() -> None:
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig(pipeline_model_parallel_size=4)
    validate_parallel_config(model, parallel)
    layer_counts = layers_per_pp_stage(model, parallel)
    assert layer_counts == [8, 8, 8, 8]
    p0 = partition_for_rank(model, parallel, 0)
    p_last = partition_for_rank(model, parallel, 3)
    assert p0.has_embedding and not p0.has_final_norm
    assert p_last.has_final_norm and p_last.has_output_projection
    assert p0.num_layers_on_rank == 8


def test_pp_layout_overrides_partition() -> None:
    model = load_model_config("llama3.1_8B")
    layout = [9, 8, 8, 7]
    parallel = ParallelConfig(pipeline_model_parallel_size=4, pipeline_model_parallel_layout=layout)
    validate_parallel_config(model, parallel)
    assert layers_per_pp_stage(model, parallel) == layout


def test_pp_non_divisible_raises_with_helpful_message() -> None:
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig(pipeline_model_parallel_size=5)
    with pytest.raises(ValueError, match="not divisible"):
        validate_parallel_config(model, parallel)


def test_pp_vpp_non_divisible_suggests_layout() -> None:
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig(pipeline_model_parallel_size=4, virtual_pipeline_model_parallel_size=3)
    with pytest.raises(ValueError, match="pipeline_model_parallel_layout"):
        validate_parallel_config(model, parallel)


def test_pp_layout_wrong_sum_raises() -> None:
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig(pipeline_model_parallel_size=4, pipeline_model_parallel_layout=[9, 8, 8, 8])
    with pytest.raises(ValueError, match="sums to 33"):
        validate_parallel_config(model, parallel)


def test_layout_and_vpp_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        ParallelConfig(
            pipeline_model_parallel_size=4,
            virtual_pipeline_model_parallel_size=2,
            pipeline_model_parallel_layout=[8, 8, 8, 8],
        )


def test_sequence_parallel_requires_tp_gt_1() -> None:
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig(sequence_parallel=True, tensor_model_parallel_size=1)
    with pytest.raises(ValueError, match="sequence_parallel requires"):
        validate_parallel_config(model, parallel)


def test_decompose_rank_default_ordering() -> None:
    parallel = ParallelConfig(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        data_parallel_size=2,
    )
    coord_0 = decompose_rank(0, parallel)
    coord_1 = decompose_rank(1, parallel)
    coord_2 = decompose_rank(2, parallel)
    coord_4 = decompose_rank(4, parallel)
    assert (coord_0.tp, coord_0.cp, coord_0.dp, coord_0.pp) == (0, 0, 0, 0)
    assert (coord_1.tp, coord_1.cp, coord_1.dp, coord_1.pp) == (1, 0, 0, 0)
    assert (coord_2.tp, coord_2.cp, coord_2.dp, coord_2.pp) == (0, 0, 1, 0)
    assert (coord_4.tp, coord_4.cp, coord_4.dp, coord_4.pp) == (0, 0, 0, 1)


def test_decompose_rank_world_excludes_ep() -> None:
    """EP is *not* a separate axis of world size — it shards inside (cp*dp).

    Scenario: world=32, dp=8, ep=8, pp=4, tp=1, cp=1. Then EDP=1, world=tp*cp*dp*pp=32.
    """
    parallel = ParallelConfig(
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        data_parallel_size=8,
        pipeline_model_parallel_size=4,
        expert_model_parallel_size=8,
    )
    assert parallel.world_size == 32
    assert parallel.expert_data_parallel_size == 1
    # Rank 8: dp_r=0, pp_r=1 (since rank 8 // (tp*cp*dp) = 8 // 8 = 1)
    c8 = decompose_rank(8, parallel)
    assert (c8.tp, c8.cp, c8.dp, c8.pp) == (0, 0, 0, 1)
    # In the expert decomposition (cp=1, dp=edp=1, ep=8 on the same world):
    # within-pp position = 0, so ep_r = 0, edp_r = 0.
    assert (c8.ep, c8.expert_dp) == (0, 0)


def test_decompose_rank_ep_shifts_within_pp_stage() -> None:
    """Within a PP stage, EP shards across the (cp*dp) ranks."""
    parallel = ParallelConfig(
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        data_parallel_size=8,
        pipeline_model_parallel_size=4,
        expert_model_parallel_size=8,
    )
    # Ranks 0..7 share PP stage 0; their ep_r runs 0..7, all with edp_r=0.
    for r in range(8):
        c = decompose_rank(r, parallel)
        assert c.pp == 0
        assert (c.ep, c.expert_dp) == (r, 0)
    # Ranks 8..15 share PP stage 1; same pattern.
    for r in range(8, 16):
        c = decompose_rank(r, parallel)
        assert c.pp == 1
        assert (c.ep, c.expert_dp) == (r - 8, 0)


def test_decompose_rank_with_edp_greater_than_1() -> None:
    """With ep=2, edp=4 (dp=8): ep_r alternates and edp_r grows."""
    parallel = ParallelConfig(
        tensor_model_parallel_size=1,
        data_parallel_size=8,
        pipeline_model_parallel_size=2,
        expert_model_parallel_size=2,
    )
    assert parallel.expert_data_parallel_size == 4
    # Within pp stage 0 (ranks 0..7): tp_r=0, then ep×edp = 2×4 ordered as (ep_r, edp_r).
    coords = [(decompose_rank(r, parallel).ep, decompose_rank(r, parallel).expert_dp) for r in range(8)]
    assert coords == [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3)]


def test_invalid_ep_when_not_dividing_cp_dp() -> None:
    from projection import load_model_config
    from projection.parallel.ranks import validate_parallel_config

    model = load_model_config("deepseek_v2_lite")
    parallel = ParallelConfig(data_parallel_size=3, expert_model_parallel_size=2)
    with pytest.raises(ValueError, match="cp\\*dp"):
        validate_parallel_config(model, parallel)


def test_rank_outside_world_raises() -> None:
    parallel = ParallelConfig(tensor_model_parallel_size=2)
    with pytest.raises(ValueError, match="outside world_size"):
        decompose_rank(2, parallel)


def test_per_rank_param_count_halves_with_tp2() -> None:
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig(tensor_model_parallel_size=2)
    workload = Workload(seq_length=8192, micro_batch_size=1, global_batch_size=64)
    trainer = Trainer(model, parallel, workload, global_rank=0)
    mem = trainer.report().memory
    full_total = 8_030_261_248
    assert mem.param_bytes == (full_total // 2) * 2
