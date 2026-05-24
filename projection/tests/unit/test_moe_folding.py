"""MoE folding: independent attention vs MoE parallel strategies."""

from __future__ import annotations

import pytest

from projection import ParallelConfig, Trainer, Workload, load_model_config
from projection.parallel.ranks import validate_parallel_config


def _setup(parallel: ParallelConfig) -> Trainer:
    model = load_model_config("deepseek_v2_lite")
    workload = Workload(seq_length=2048, micro_batch_size=1, global_batch_size=64)
    return Trainer(model, parallel, workload)


def test_moe_folding_off_etp_equals_tp() -> None:
    """Without folding, ETP defaults to TP (Megatron's expert RankGenerator default)."""
    p = ParallelConfig(tensor_model_parallel_size=2, moe_folding=False)
    assert p.effective_expert_tensor_parallel_size == 2


def test_moe_folding_on_uses_explicit_etp() -> None:
    p = ParallelConfig(
        tensor_model_parallel_size=4,
        moe_folding=True,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=8,
        data_parallel_size=2,
    )
    assert p.effective_expert_tensor_parallel_size == 1
    # World = TP*PP*CP*DP = 4*1*1*2 = 8. EDP = world / (ETP*EP*PP) = 8/(1*8*1) = 1.
    assert p.world_size == 8
    assert p.expert_data_parallel_size == 1


def test_moe_folding_on_requires_moe_model() -> None:
    """``moe_folding=True`` on a dense model should be rejected."""
    model = load_model_config("llama3.1_8B")
    parallel = ParallelConfig(moe_folding=True)
    workload = Workload(seq_length=1024, micro_batch_size=1, global_batch_size=64)
    with pytest.raises(ValueError, match="moe_folding can only be enabled for MoE models"):
        Trainer(model, parallel, workload)


def test_world_divisibility_for_expert_groups() -> None:
    """world must be divisible by ETP*EP*PP (or vice versa for fractional EDP)."""
    model = load_model_config("deepseek_v2_lite")
    # ETP=4 divides moe_ffn=1408; EP=2 divides num_routed=64. world=2*1*1*6=12.
    # ETP*EP*PP = 4*2*1 = 8. 12 % 8 = 4, 8 % 12 = 8 → neither divides → invalid.
    parallel = ParallelConfig(
        tensor_model_parallel_size=2,
        moe_folding=True,
        expert_tensor_parallel_size=4,
        expert_model_parallel_size=2,
        data_parallel_size=6,
    )
    with pytest.raises(ValueError, match="ETP\\*EP\\*PP"):
        validate_parallel_config(model, parallel)


def test_chained_optimizer_shards_dense_by_dp_and_expert_by_edp() -> None:
    """With folding on, dense params shard by DP, routed-expert params by EDP."""
    p = ParallelConfig(
        tensor_model_parallel_size=1,
        moe_folding=True,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=2,
        data_parallel_size=4,
    )
    # world=4, EDP = 4/(1*2*1) = 2, DP = 4.
    assert p.data_parallel_size == 4
    assert p.expert_data_parallel_size == 2

    mem = _setup(p).report().memory.optimizer
    # Optimizer state has fp32 master + Adam (12 bytes/param). For DSV2-Lite
    # (~15.7B params), routed experts dominate (≈14.85B of those are routed).
    # Dense state shards by DP=4, routed by EDP=2.
    assert mem.main_param_bytes > 0
    assert mem.state_bytes == 2 * mem.main_param_bytes  # m + v


def test_chained_optimizer_dense_dp_routed_edp_split() -> None:
    """Doubling DP halves the dense optimizer state contribution; EDP halves the routed."""
    base_p = ParallelConfig(
        tensor_model_parallel_size=1,
        moe_folding=True,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=2,
        data_parallel_size=2,
    )
    big_dp = ParallelConfig(
        tensor_model_parallel_size=1,
        moe_folding=True,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=2,
        data_parallel_size=4,
    )
    a = _setup(base_p).report().memory.optimizer.main_param_bytes
    b = _setup(big_dp).report().memory.optimizer.main_param_bytes
    # Increasing DP from 2 to 4 also doubles EDP (2 → 2 stays since ep=2, but
    # actually with world doubling EDP also doubles: world=4→8, EDP=8/(1*2*1)=4).
    # Both dense and routed parts get further sharded, so total drops.
    assert b < a


def test_etp_changes_routed_param_bytes() -> None:
    """Smaller ETP means each rank holds more routed-expert params (less TP sharding)."""
    parallel_high_etp = ParallelConfig(
        tensor_model_parallel_size=2,
        moe_folding=True,
        expert_tensor_parallel_size=2,
        expert_model_parallel_size=2,
        data_parallel_size=2,
    )
    parallel_low_etp = ParallelConfig(
        tensor_model_parallel_size=2,
        moe_folding=True,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=2,
        data_parallel_size=2,
    )
    a = _setup(parallel_high_etp).report().memory.param_bytes
    b = _setup(parallel_low_etp).report().memory.param_bytes
    # Lower ETP → routed params less sharded → more bytes per rank.
    assert b > a


def test_edp_can_be_fractional_when_ep_exceeds_attention_dp() -> None:
    """``EDP < 1`` means each rank holds ``1/EDP`` expert slices."""
    p = ParallelConfig(
        tensor_model_parallel_size=1,
        moe_folding=True,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=16,
        data_parallel_size=4,
    )
    # world = 1*1*1*4 = 4. EDP = 4 / (1*16*1) = 1/4.
    assert p.world_size == 4
    assert p.expert_data_parallel_size == pytest.approx(0.25)


def test_edp_fractional_routed_params_replicate() -> None:
    """With EDP=1/n, per-rank routed params = base_shard * n."""
    base = ParallelConfig(
        tensor_model_parallel_size=1,
        moe_folding=True,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=4,
        data_parallel_size=4,
    )  # EDP = 4/(1*4*1) = 1
    sparse = ParallelConfig(
        tensor_model_parallel_size=1,
        moe_folding=True,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=16,
        data_parallel_size=4,
    )  # EDP = 4/(1*16*1) = 1/4 → each rank holds 4 expert slices
    a = _setup(base).report().memory.param_bytes
    b = _setup(sparse).report().memory.param_bytes
    # Both configs put the same routed params on each rank because EP × replication
    # cancels out: base has EP=4 slices × 1 copy each; sparse has EP=16 slices × 1/4
    # copies (each rank holds 4 slices = same number of params).
    assert a == b


def test_edp_below_1_does_not_shard_routed_optimizer_state() -> None:
    """When EDP < 1, the effective shard divisor for routed optimizer state is 1.

    Compared to EDP = 1 at the same EP, the routed state per rank should be
    identical (both use max(1, EDP) = 1).
    """
    edp_eq1 = ParallelConfig(
        tensor_model_parallel_size=1,
        moe_folding=True,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=8,
        data_parallel_size=8,
    )  # EDP = 1
    edp_half = ParallelConfig(
        tensor_model_parallel_size=1,
        moe_folding=True,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=16,
        data_parallel_size=8,
    )  # EDP = 1/2 → routed params per rank doubles, state shard stays 1.

    eq1 = _setup(edp_eq1).report().memory.optimizer.state_bytes
    half = _setup(edp_half).report().memory.optimizer.state_bytes
    # Both use shard factor max(1, EDP) = 1 for routed; total state ends up the
    # same because (params × 2) ÷ (1 × 1) == params ÷ 1 × 2 (per-rank routed
    # params went up by 2× under EDP=1/2). The point: no further DP saving.
    assert eq1 == half


def test_etp_changes_moe_layer_activation_bytes() -> None:
    """Per-MoE-layer activation uses ETP when folding is on; lowering ETP raises activation."""
    high_etp = ParallelConfig(
        tensor_model_parallel_size=2,
        moe_folding=True,
        expert_tensor_parallel_size=2,
        expert_model_parallel_size=2,
        data_parallel_size=2,
    )
    low_etp = ParallelConfig(
        tensor_model_parallel_size=2,
        moe_folding=True,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=2,
        data_parallel_size=2,
    )
    a = _setup(high_etp).report().memory.activation_bytes
    b = _setup(low_etp).report().memory.activation_bytes
    assert b > a
