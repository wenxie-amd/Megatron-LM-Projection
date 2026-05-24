"""Optimizer memory model.

For v1 we model only Megatron's **distributed optimizer** with Adam:

- Parameter copy:   ``param_dtype`` bytes per param (unsharded by DP; sharded by TP+PP+CP+EP)
- Gradient buffer:  fp32 by default (``--accumulate-allreduce-grads-in-fp32``); else ``param_dtype``
- Master param:     fp32, sharded across the DP group
- Adam momentum:    fp32, sharded across the DP group
- Adam variance:    fp32, sharded across the DP group

Torch FSDP2 and Megatron FSDP are deferred to M5.
"""

from __future__ import annotations

from dataclasses import dataclass

from projection.configs import (
    BF16_BYTES,
    FP32_BYTES,
    OptimizerDtype,
    OptimizerKind,
    ParallelConfig,
    Precision,
)

_DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2}


def _dtype_bytes(dtype: OptimizerDtype) -> int:
    return _DTYPE_BYTES[dtype]


@dataclass(frozen=True)
class OptimizerMemory:
    """Per-rank optimizer memory.

    Three independent buckets:

    - ``grad_buffer_bytes`` — the single contiguous ``main_grad`` buffer. In
      Megatron, each ``param.main_grad`` is a slice/view into one big
      DDP-allocated tensor whose dtype is fp32 by default (controlled by
      ``--accumulate-allreduce-grads-in-fp32`` or the precision-aware
      ``--main-grads-dtype``). The optimizer reads from the same buffer; there
      is no separate bf16 model-side gradient.
    - ``main_param_bytes`` — fp32 master copy of the params (sharded by DP for
      the distributed optimizer; for FSDP, also sharded by DP).
    - ``state_bytes`` — Adam ``m`` + ``v`` (each fp32 by default).
    """

    grad_buffer_bytes: int
    main_param_bytes: int
    state_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.grad_buffer_bytes + self.main_param_bytes + self.state_bytes


class DistributedOptimizer:
    """Megatron distributed Adam. Sharded across the DP group.

    With ``use_precision_aware_optimizer``, the grad buffer / master param /
    Adam momentum / Adam variance dtypes can each be set independently
    (matching Megatron's ``--main-grads-dtype`` family of flags).
    """

    def __init__(self, parallel: ParallelConfig, accumulate_grads_in_fp32: bool = True):
        if parallel.optimizer_kind is not OptimizerKind.DISTRIBUTED_OPTIMIZER:
            raise ValueError(
                f"DistributedOptimizer requires optimizer_kind=DISTRIBUTED_OPTIMIZER, "
                f"got {parallel.optimizer_kind}"
            )
        self._parallel = parallel
        self._accumulate_grads_in_fp32 = accumulate_grads_in_fp32

    def grad_dtype_bytes(self, precision: Precision) -> int:
        if self._parallel.use_precision_aware_optimizer:
            return _dtype_bytes(self._parallel.optimizer_main_grad_dtype)
        return FP32_BYTES if self._accumulate_grads_in_fp32 else precision.bytes

    def memory_for(self, params_on_rank: int, precision: Precision) -> OptimizerMemory:
        return self.memory_for_split(params_on_rank, 0, precision)

    def memory_for_split(
        self, dense_params_on_rank: int, routed_params_on_rank: int, precision: Precision
    ) -> OptimizerMemory:
        """Chained optimizer: dense uses ADP, routed-experts use EDP.

        Matches Megatron's ``ChainedOptimizer`` setup with separate expert /
        non-expert param groups — dense main_param + Adam state shard by ADP;
        routed-expert main_param + Adam state shard by EDP. When ``EDP < 1``
        (each rank holds ``1/EDP`` expert slices) the routed-expert optimizer
        state is *not* further sharded: ``max(1, EDP)`` is the effective shard
        divisor.
        """
        dp = self._parallel.data_parallel_size
        edp_raw = self._parallel.expert_data_parallel_size  # float (may be < 1)
        edp_effective = max(1.0, edp_raw) if edp_raw > 0 else max(1.0, float(dp))
        if self._parallel.use_precision_aware_optimizer:
            master_bytes_per = _dtype_bytes(self._parallel.optimizer_main_param_dtype)
            m_bytes_per = _dtype_bytes(self._parallel.optimizer_exp_avg_dtype)
            v_bytes_per = _dtype_bytes(self._parallel.optimizer_exp_avg_sq_dtype)
        else:
            master_bytes_per = m_bytes_per = v_bytes_per = FP32_BYTES

        # Grad buffer is full-numel on every DP rank (and full on every EDP
        # rank for the expert sub-buffer): not sharded.
        grad_bytes_per = self.grad_dtype_bytes(precision)
        grad_bytes = (dense_params_on_rank + routed_params_on_rank) * grad_bytes_per

        adam_bytes = m_bytes_per + v_bytes_per
        dense_main = (dense_params_on_rank * master_bytes_per) // max(1, dp)
        dense_state = (dense_params_on_rank * adam_bytes) // max(1, dp)
        routed_main = int((routed_params_on_rank * master_bytes_per) // edp_effective)
        routed_state = int((routed_params_on_rank * adam_bytes) // edp_effective)

        return OptimizerMemory(
            grad_buffer_bytes=grad_bytes,
            main_param_bytes=dense_main + routed_main,
            state_bytes=dense_state + routed_state,
        )


class FSDPOptimizer:
    """Torch FSDP2 / Megatron FSDP. Params + grads + optimizer state all sharded across DP.

    Simplified model: under FSDP, *every* tensor (params, grads, master, m, v) is
    DP-sharded. We expose this through the same :class:`OptimizerMemory` shape
    for consistency with the distributed-optimizer case.
    """

    def __init__(self, parallel: ParallelConfig):
        if parallel.optimizer_kind not in (OptimizerKind.TORCH_FSDP2, OptimizerKind.MEGATRON_FSDP):
            raise ValueError(
                f"FSDPOptimizer requires optimizer_kind in (TORCH_FSDP2, MEGATRON_FSDP), "
                f"got {parallel.optimizer_kind}"
            )
        self._parallel = parallel

    def memory_for(self, params_on_rank: int, precision: Precision) -> OptimizerMemory:
        return self.memory_for_split(params_on_rank, 0, precision)

    def memory_for_split(
        self, dense_params_on_rank: int, routed_params_on_rank: int, precision: Precision
    ) -> OptimizerMemory:
        """FSDP: every tensor lives only as a local DP shard (already accounted for in caller)."""
        _ = precision
        total = dense_params_on_rank + routed_params_on_rank
        return OptimizerMemory(
            grad_buffer_bytes=total * FP32_BYTES,
            main_param_bytes=total * FP32_BYTES,
            state_bytes=total * (FP32_BYTES + FP32_BYTES),
        )


def model_param_bytes(params_on_rank: int, precision: Precision) -> int:
    """Live model param bytes (sharded by TP+PP+CP+EP, *not* by DP for distributed optimizer)."""
    return params_on_rank * precision.bytes


__all__ = [
    "BF16_BYTES",
    "DistributedOptimizer",
    "FP32_BYTES",
    "FSDPOptimizer",
    "OptimizerMemory",
    "model_param_bytes",
]
