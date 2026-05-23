"""Precision-aware distributed optimizer dtype handling."""

from __future__ import annotations

from projection import ParallelConfig, Trainer, Workload, load_model_config


def _trainer(parallel: ParallelConfig) -> Trainer:
    model = load_model_config("llama3.1_8B")
    workload = Workload(seq_length=2048, micro_batch_size=1, global_batch_size=64)
    return Trainer(model, parallel, workload)


def test_default_optimizer_uses_fp32() -> None:
    mem = _trainer(ParallelConfig()).report().memory.optimizer
    # 8.03B params × 4 bytes ≈ 32 GB grad_buffer, 32 GB main_param, 64 GB state (m+v).
    assert mem.grad_buffer_bytes == mem.main_param_bytes
    assert mem.state_bytes == mem.main_param_bytes * 2


def test_precision_aware_bf16_master_halves_main_param_bytes() -> None:
    base = _trainer(ParallelConfig()).report().memory.optimizer
    par = (
        _trainer(
            ParallelConfig(
                use_precision_aware_optimizer=True,
                optimizer_main_param_dtype="bf16",
                optimizer_exp_avg_dtype="fp32",
                optimizer_exp_avg_sq_dtype="fp32",
                optimizer_main_grad_dtype="fp32",
            )
        )
        .report()
        .memory.optimizer
    )
    assert par.main_param_bytes == base.main_param_bytes // 2
    assert par.state_bytes == base.state_bytes


def test_precision_aware_bf16_adam_states_halves_them() -> None:
    base = _trainer(ParallelConfig()).report().memory.optimizer
    par = (
        _trainer(
            ParallelConfig(
                use_precision_aware_optimizer=True,
                optimizer_exp_avg_dtype="bf16",
                optimizer_exp_avg_sq_dtype="bf16",
            )
        )
        .report()
        .memory.optimizer
    )
    assert par.state_bytes == base.state_bytes // 2


def test_grad_buffer_is_not_sharded_by_dp() -> None:
    """The main_grad buffer is allocated full-numel on every DP rank (see Megatron's DDP)."""
    base = _trainer(ParallelConfig(data_parallel_size=1)).report().memory.optimizer
    sharded = _trainer(ParallelConfig(data_parallel_size=4)).report().memory.optimizer
    assert sharded.grad_buffer_bytes == base.grad_buffer_bytes
    # But main_param and state ARE sharded.
    assert sharded.main_param_bytes == base.main_param_bytes // 4
    assert sharded.state_bytes == base.state_bytes // 4
