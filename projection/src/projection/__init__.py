"""Memory and structure projection for Megatron-LM training jobs."""

from projection.configs import (
    ArchitectureConfig,
    AttentionConfig,
    GPUSpec,
    MLPConfig,
    ModelConfig,
    NormConfig,
    OptimizerKind,
    ParallelConfig,
    PositionEmbeddingConfig,
    Precision,
    TrainingHyperparameters,
    Workload,
)
from projection.core.trainer import MemoryBreakdown, Trainer, TrainerReport
from projection.loader import (
    list_builtin_gpus,
    list_builtin_models,
    load_gpu_spec,
    load_model_config,
)

__version__ = "0.1.0"


def hello() -> str:
    return "Hello from projection!"


__all__ = [
    "__version__",
    "hello",
    "ArchitectureConfig",
    "AttentionConfig",
    "GPUSpec",
    "MLPConfig",
    "MemoryBreakdown",
    "ModelConfig",
    "NormConfig",
    "OptimizerKind",
    "ParallelConfig",
    "PositionEmbeddingConfig",
    "Precision",
    "Trainer",
    "TrainerReport",
    "TrainingHyperparameters",
    "Workload",
    "list_builtin_gpus",
    "list_builtin_models",
    "load_gpu_spec",
    "load_model_config",
]
