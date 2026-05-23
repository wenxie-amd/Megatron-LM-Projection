"""Config schemas for models, GPUs, parallelism, and workload.

Field names mirror Megatron's argument names exactly (e.g. ``num_layers``,
``hidden_size``) so users can map them 1:1 to a Megatron run.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Precision(str, Enum):
    """Training precision for compute / activations / weights / grads."""

    BF16 = "bf16"
    FP8 = "fp8"

    @property
    def bytes(self) -> int:
        return {Precision.BF16: 2, Precision.FP8: 1}[self]


FP32_BYTES = 4
BF16_BYTES = 2
FP8_BYTES = 1


class ArchitectureConfig(BaseModel):
    """Top-level model shape."""

    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    vocab_size: int
    max_position_embeddings: int
    untie_embeddings_and_output_weights: bool = False
    make_vocab_size_divisible_by: int = 128


class AttentionConfig(BaseModel):
    """Multi-head attention block shape.

    Set ``use_mla=true`` to switch to Multi-head Latent Attention (DeepSeek-V2/V3),
    in which case the MLA-specific fields take over and ``kv_channels`` /
    ``num_query_groups`` are ignored.
    """

    num_attention_heads: int
    num_query_groups: int | None = None
    kv_channels: int | None = None
    attention_dropout: float = 0.0
    add_qkv_bias: bool = False

    use_mla: bool = False
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None

    def num_kv_heads(self) -> int:
        return self.num_query_groups or self.num_attention_heads

    def head_dim(self, hidden_size: int) -> int:
        return self.kv_channels or (hidden_size // self.num_attention_heads)


class MLPConfig(BaseModel):
    """Dense MLP block shape."""

    swiglu: bool = False
    add_bias_linear: bool = True


class MoEConfig(BaseModel):
    """Optional MoE block. Layers ``[first_k_dense_replace, num_layers)`` are MoE."""

    enabled: bool = False
    moe_ffn_hidden_size: int = 0
    num_routed_experts: int = 0
    num_shared_experts: int = 0
    moe_router_topk: int = 1
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 0
    add_router_bias: bool = False


class NormConfig(BaseModel):
    normalization: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    layernorm_epsilon: float = 1.0e-5


class PositionEmbeddingConfig(BaseModel):
    position_embedding_type: Literal["learned_absolute", "rope", "none"] = "learned_absolute"
    rotary_base: float = 10_000.0
    rotary_percent: float = 1.0


class ModelConfig(BaseModel):
    """The full Megatron ``TransformerConfig`` subset we track."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    architecture: ArchitectureConfig
    attention: AttentionConfig
    mlp: MLPConfig = Field(default_factory=MLPConfig)
    moe: MoEConfig = Field(default_factory=MoEConfig)
    norm: NormConfig = Field(default_factory=NormConfig)
    position_embedding: PositionEmbeddingConfig = Field(default_factory=PositionEmbeddingConfig)

    @property
    def is_proxy(self) -> bool:
        return self.name.endswith("(proxy)")


class GPUSpec(BaseModel):
    """A single GPU's hardware spec, used for spec display and headroom checks."""

    model_config = ConfigDict(extra="forbid")

    name: str
    vendor: Literal["nvidia", "amd"]
    memory_gb: int
    bf16_tflops: float
    fp8_tflops: float
    bandwidth_gbps: float


class Workload(BaseModel):
    """Per-step training workload knobs (memory- and throughput-sensitive).

    Structural construction only — semantic checks (e.g. ``recompute_granularity='full'``
    requires ``recompute_method`` + ``recompute_num_layers``) live in
    :func:`projection.parallel.ranks.validate_workload` so the UI can hold
    transient incomplete state.
    """

    model_config = ConfigDict(extra="forbid")

    seq_length: int
    micro_batch_size: int
    global_batch_size: int
    recompute_granularity: Literal["none", "selective", "full"] = "none"
    recompute_method: Literal["uniform", "block"] | None = None
    recompute_num_layers: int | None = None
    sequence_parallel: bool = False


class OptimizerKind(str, Enum):
    DISTRIBUTED_OPTIMIZER = "distributed_optimizer"
    TORCH_FSDP2 = "torch_fsdp2"
    MEGATRON_FSDP = "megatron_fsdp"


OptimizerDtype = Literal["fp32", "bf16", "fp16"]


class ParallelConfig(BaseModel):
    """``ModelParallelConfig`` subset.

    PP can be set either via ``pipeline_model_parallel_size`` + ``virtual_pipeline_model_parallel_size``
    or via ``pipeline_model_parallel_layout`` (mutually exclusive).

    ``data_parallel_size`` is treated as derived in the UI (``world_size /
    (TP * PP * CP)``) but accepted here as an explicit field for API stability
    and to keep the projection runnable as a CLI without a UI.
    """

    model_config = ConfigDict(extra="forbid")

    precision: Precision = Precision.BF16

    tensor_model_parallel_size: int = 1
    sequence_parallel: bool = False
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int | None = None
    pipeline_model_parallel_layout: list[int] | None = None
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    data_parallel_size: int = 1

    optimizer_kind: OptimizerKind = OptimizerKind.DISTRIBUTED_OPTIMIZER

    use_precision_aware_optimizer: bool = False
    optimizer_main_param_dtype: OptimizerDtype = "fp32"
    optimizer_main_grad_dtype: OptimizerDtype = "fp32"
    optimizer_exp_avg_dtype: OptimizerDtype = "fp32"
    optimizer_exp_avg_sq_dtype: OptimizerDtype = "fp32"

    @model_validator(mode="after")
    def _check_pp(self) -> "ParallelConfig":
        if (
            self.pipeline_model_parallel_layout is not None
            and self.virtual_pipeline_model_parallel_size is not None
        ):
            raise ValueError(
                "pipeline_model_parallel_layout and virtual_pipeline_model_parallel_size are mutually exclusive"
            )
        for name, value in {
            "tensor_model_parallel_size": self.tensor_model_parallel_size,
            "pipeline_model_parallel_size": self.pipeline_model_parallel_size,
            "context_parallel_size": self.context_parallel_size,
            "expert_model_parallel_size": self.expert_model_parallel_size,
            "data_parallel_size": self.data_parallel_size,
        }.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1 (got {value})")
        return self

    @property
    def world_size(self) -> int:
        """``world = tp * cp * dp * pp``.

        EP is **not** a separate axis multiplying world size: it lives inside
        the ``cp * dp`` axis (Megatron's expert RankGenerator sets ``cp=1`` and
        ``dp = expert_data_parallel_size = cp * dp / ep`` on the same world).
        """
        return (
            self.tensor_model_parallel_size
            * self.pipeline_model_parallel_size
            * self.context_parallel_size
            * self.data_parallel_size
        )

    @property
    def expert_data_parallel_size(self) -> int:
        """``EDP = cp * dp / ep`` (matches Megatron's expert RankGenerator)."""
        if self.expert_model_parallel_size <= 0:
            return 0
        numer = self.context_parallel_size * self.data_parallel_size
        if numer % self.expert_model_parallel_size != 0:
            return 0
        return max(1, numer // self.expert_model_parallel_size)


class TrainingHyperparameters(BaseModel):
    """No memory/throughput impact; defaults are fine."""

    model_config = ConfigDict(extra="forbid")

    lr: float = 3.0e-4
    min_lr: float = 3.0e-5
    train_iters: int = 100_000
    weight_decay: float = 0.1
    clip_grad: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1.0e-8
