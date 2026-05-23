"""Stable JSON-in / JSON-out API consumed by the Pyodide bridge.

The bridge passes a single JSON-serializable dict in and gets a single
JSON-serializable dict back. Keeping the surface area small here makes the
TypeScript side trivial and decouples the UI from the internal class layout.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from projection.configs import (
    GPUSpec,
    ModelConfig,
    ParallelConfig,
    TrainingHyperparameters,
    Workload,
)
from projection.core.trainer import Trainer, TrainerReport
from projection.loader import (
    list_builtin_gpus,
    list_builtin_models,
    load_gpu_spec,
    load_model_config,
)
from projection.parallel.ranks import gradient_accumulation_steps
from projection.script_gen import generate_megatron_script

MAX_RANKS = 8


def list_models() -> list[str]:
    """List built-in model names exactly as the dropdown should show them."""
    return [load_model_config(name).name for name in list_builtin_models()]


def list_gpus() -> list[str]:
    return [load_gpu_spec(name).name for name in list_builtin_gpus()]


def get_model_config(name: str) -> dict[str, Any]:
    return load_model_config(name).model_dump()


def get_gpu_spec(name: str) -> dict[str, Any]:
    return load_gpu_spec(name).model_dump()


def get_model_breakdown(model: str | dict[str, Any]) -> dict[str, Any]:
    """Return full-model param count + per-module breakdown (no parallelism).

    Also returns a ``ffn_breakdown`` for whichever block dominates per layer:

    - For dense MLP (SwiGLU): gate / up / down.
    - For MoE: router + routed experts (one slice per routed expert group)
      + shared experts.
    """
    if isinstance(model, str):
        model_config = load_model_config(model)
    else:
        model_config = ModelConfig.model_validate(model)
    parallel = ParallelConfig()
    workload = Workload(seq_length=1, micro_batch_size=1, global_batch_size=1)
    trainer = Trainer(model_config, parallel, workload, global_rank=0)
    report = trainer.report()
    return {
        "param_count": report.param_count,
        "param_breakdown": [{"name": p.name, "count": p.count} for p in report.param_breakdown],
        "ffn_breakdown": _ffn_breakdown(model_config),
    }


def _ffn_breakdown(model: ModelConfig) -> dict[str, Any]:
    h = model.architecture.hidden_size

    if model.moe.enabled:
        moe = model.moe
        from projection.core.modules import MoEModule

        block = MoEModule(model)
        per_expert = block.routed_expert_param_count()
        return {
            "kind": "moe",
            "entries": [
                {"name": "router (gate)", "count": block.gate_param_count()},
                {
                    "name": f"routed experts ×{moe.num_routed_experts}",
                    "count": moe.num_routed_experts * per_expert,
                },
                {
                    "name": f"shared experts ×{moe.num_shared_experts}",
                    "count": moe.num_shared_experts * block.shared_expert_param_count(),
                },
            ],
        }

    ffn = model.architecture.ffn_hidden_size
    if model.mlp.swiglu:
        return {
            "kind": "mlp",
            "entries": [
                {"name": "gate_proj", "count": h * ffn},
                {"name": "up_proj", "count": h * ffn},
                {"name": "down_proj", "count": ffn * h},
            ],
        }
    return {
        "kind": "mlp",
        "entries": [
            {"name": "fc1", "count": h * ffn},
            {"name": "fc2", "count": ffn * h},
        ],
    }


def run_projection(payload: dict[str, Any]) -> dict[str, Any]:
    """Build trainers for the requested ranks and return memory + structure for each.

    Expected payload shape::

        {
          "model": "llama3.1_8B" | <inline model dict>,
          "parallel": { ...ParallelConfig fields... },
          "workload": { ...Workload fields... },
          "ranks": [0, 1, 4],          # up to MAX_RANKS
          "hyperparameters": { ... }    # optional
        }
    """
    model_field = payload["model"]
    if isinstance(model_field, str):
        model_config = load_model_config(model_field)
    elif isinstance(model_field, dict):
        model_config = ModelConfig.model_validate(model_field)
    else:
        raise TypeError("'model' must be a string name or an inline model dict")

    parallel = ParallelConfig.model_validate(payload["parallel"])
    workload = Workload.model_validate(payload["workload"])
    hyperparameters = TrainingHyperparameters.model_validate(payload.get("hyperparameters") or {})

    ranks: list[int] = list(payload.get("ranks", [0]))
    if len(ranks) == 0:
        raise ValueError("ranks must contain at least 1 entry")
    if len(ranks) > MAX_RANKS:
        raise ValueError(f"ranks may contain at most {MAX_RANKS} entries (got {len(ranks)})")
    if len(set(ranks)) != len(ranks):
        raise ValueError("ranks must be unique")

    rank_reports = []
    for r in ranks:
        trainer = Trainer(model_config, parallel, workload, global_rank=r, hyperparameters=hyperparameters)
        rank_reports.append(_report_to_dict(trainer.report()))

    return {
        "model_config": model_config.model_dump(),
        "parallel": parallel.model_dump(),
        "workload": workload.model_dump(),
        "derived": {
            "world_size": parallel.world_size,
            "data_parallel_size": parallel.data_parallel_size,
            "expert_data_parallel_size": parallel.expert_data_parallel_size,
            "gradient_accumulation_steps": gradient_accumulation_steps(parallel, workload),
        },
        "rank_reports": rank_reports,
    }


def _report_to_dict(report: TrainerReport) -> dict[str, Any]:
    return {
        "global_rank": report.global_rank,
        "rank_coord": _asdict_safe(report.rank_coord),
        "partition": _asdict_safe(report.partition),
        "param_count": report.param_count,
        "param_breakdown": [{"name": p.name, "count": p.count} for p in report.param_breakdown],
        "memory": {
            "param_bytes": report.memory.param_bytes,
            "activation_bytes": report.memory.activation_bytes,
            "optimizer": {
                "grad_buffer_bytes": report.memory.optimizer.grad_buffer_bytes,
                "main_param_bytes": report.memory.optimizer.main_param_bytes,
                "state_bytes": report.memory.optimizer.state_bytes,
                "total_bytes": report.memory.optimizer.total_bytes,
            },
            "total_bytes": report.memory.total_bytes,
            "precision": report.memory.precision.value,
        },
    }


def compute_derived(payload: dict[str, Any]) -> dict[str, Any]:
    """Return derived training-loop sizes (world_size, dp, edp, ga) + any validation error.

    Lighter than :func:`run_projection`: skips per-rank instantiation. The UI
    calls this in Step 3 to show derived values live as the user edits.
    """
    parallel = ParallelConfig.model_validate(payload["parallel"])
    workload = Workload.model_validate(payload["workload"])
    return {
        "world_size": parallel.world_size,
        "data_parallel_size": parallel.data_parallel_size,
        "expert_data_parallel_size": parallel.expert_data_parallel_size,
        "gradient_accumulation_steps": gradient_accumulation_steps(parallel, workload),
    }


def validate_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate without running the projection. Returns {valid: bool, errors: [str]}."""
    from projection.parallel.ranks import validate_full_config

    errors: list[str] = []
    try:
        model_field = payload["model"]
        model_config = (
            load_model_config(model_field)
            if isinstance(model_field, str)
            else ModelConfig.model_validate(model_field)
        )
        parallel = ParallelConfig.model_validate(payload["parallel"])
        workload = Workload.model_validate(payload["workload"])
    except Exception as exc:
        return {"valid": False, "errors": [str(exc)]}

    try:
        validate_full_config(model_config, parallel, workload)
    except ValueError as exc:
        errors.append(str(exc))

    return {"valid": len(errors) == 0, "errors": errors}


def generate_script(payload: dict[str, Any]) -> str:
    """Return a Megatron-LM launch shell script for the given configuration.

    Same input shape as :func:`run_projection`, plus an optional ``num_gpus`` and
    ``nproc_per_node`` field (defaults: ``num_gpus`` from parallel world_size,
    ``nproc_per_node=8``).
    """
    model_field = payload["model"]
    if isinstance(model_field, str):
        model_config = load_model_config(model_field)
    else:
        model_config = ModelConfig.model_validate(model_field)

    parallel = ParallelConfig.model_validate(payload["parallel"])
    workload = Workload.model_validate(payload["workload"])
    hyperparameters = TrainingHyperparameters.model_validate(payload.get("hyperparameters") or {})
    num_gpus = int(payload.get("num_gpus") or parallel.world_size)
    nproc_per_node = int(payload.get("nproc_per_node") or 8)

    return generate_megatron_script(
        model_config,
        parallel,
        workload,
        hyperparameters,
        num_gpus=num_gpus,
        nproc_per_node=nproc_per_node,
    )


def _asdict_safe(obj: Any) -> dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (ModelConfig, ParallelConfig, Workload, GPUSpec, TrainingHyperparameters)):
        return obj.model_dump()
    raise TypeError(f"don't know how to serialize {type(obj)!r}")


__all__ = [
    "MAX_RANKS",
    "compute_derived",
    "generate_script",
    "get_gpu_spec",
    "get_model_breakdown",
    "get_model_config",
    "list_gpus",
    "list_models",
    "run_projection",
    "validate_config",
]
