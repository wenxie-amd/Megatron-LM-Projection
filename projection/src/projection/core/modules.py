"""Leaf modules: Embedding, Attention, MLP, Norm.

Each module knows how to compute its parameter count and (for non-trivial ones)
its per-microbatch activation byte count. Memory math mirrors Megatron's
formulas but is implemented independently. The values are pinned against real
``megatron.core`` in ``tests/fixtures/`` as a gold standard.
"""

from __future__ import annotations

from dataclasses import dataclass

from projection.configs import (
    ArchitectureConfig,
    AttentionConfig,
    MLPConfig,
    ModelConfig,
    MoEConfig,
    NormConfig,
)


def padded_vocab_size(architecture: ArchitectureConfig, tensor_parallel_size: int) -> int:
    """Megatron pads vocab to a multiple of ``make_vocab_size_divisible_by * TP``."""
    multiple = architecture.make_vocab_size_divisible_by * tensor_parallel_size
    return ((architecture.vocab_size + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class ModuleParams:
    """Per-module parameter breakdown for pie-chart / debugging."""

    name: str
    count: int


class EmbeddingModule:
    """Token embedding (and, when untied, the output projection lives here too).

    The output projection is modeled separately by :class:`TransformerModel` so
    this class only owns the token embedding.
    """

    def __init__(self, model: ModelConfig):
        self._model = model

    def param_count(self, tensor_parallel_size: int = 1) -> int:
        vocab = padded_vocab_size(self._model.architecture, tensor_parallel_size)
        return vocab * self._model.architecture.hidden_size


class AttentionModule:
    """Multi-head / grouped-query attention (Llama-style), Multi-head Latent
    Attention (DeepSeek-V2/V3), or DeepSeek-V4 hybrid attention.

    For V4 the call site must pass ``compress_ratio`` (0 / 4 / 128) because
    every layer has a different per-layer cost (Compressor / Indexer).
    """

    def __init__(self, model: ModelConfig):
        self._model = model

    def param_count(self, *, compress_ratio: int | None = None) -> int:
        cfg: AttentionConfig = self._model.attention
        if cfg.use_deepseek_v4:
            from projection.core.deepseek_v4 import v4_attention_param_count_per_layer

            cr = 0 if compress_ratio is None else int(compress_ratio)
            return v4_attention_param_count_per_layer(self._model, cr)
        if cfg.use_mla:
            return _mla_param_count(self._model)
        return _mha_param_count(self._model)


def _mha_param_count(model: ModelConfig) -> int:
    cfg = model.attention
    h = model.architecture.hidden_size
    head_dim = cfg.head_dim(h)
    q_dim = cfg.num_attention_heads * head_dim
    kv_dim = cfg.num_kv_heads() * head_dim
    weights = h * q_dim + h * kv_dim + h * kv_dim + q_dim * h
    bias = (q_dim + kv_dim + kv_dim) if cfg.add_qkv_bias else 0
    return weights + bias


def _mla_param_count(model: ModelConfig) -> int:
    """MLA from DeepSeek-V2.

    Required fields when ``use_mla=true``: ``kv_lora_rank``, ``qk_nope_head_dim``,
    ``qk_rope_head_dim``, ``v_head_dim``. ``q_lora_rank`` is optional (None => direct Q).
    """
    cfg = model.attention
    h = model.architecture.hidden_size
    num_heads = cfg.num_attention_heads
    nope = _required(cfg.qk_nope_head_dim, "qk_nope_head_dim")
    rope = _required(cfg.qk_rope_head_dim, "qk_rope_head_dim")
    v_dim = _required(cfg.v_head_dim, "v_head_dim")
    kv_lora = _required(cfg.kv_lora_rank, "kv_lora_rank")

    qk_head = nope + rope
    if cfg.q_lora_rank is None:
        q_path = h * num_heads * qk_head
    else:
        q_a = h * cfg.q_lora_rank
        q_a_norm = cfg.q_lora_rank
        q_b = cfg.q_lora_rank * num_heads * qk_head
        q_path = q_a + q_a_norm + q_b

    kv_a = h * (kv_lora + rope)
    kv_a_norm = kv_lora
    kv_b = kv_lora * num_heads * (nope + v_dim)
    o_proj = num_heads * v_dim * h
    return q_path + kv_a + kv_a_norm + kv_b + o_proj


def _required(value: int | None, field_name: str) -> int:
    if value is None:
        raise ValueError(f"MLA requires {field_name!r} to be set")
    return value


class MLPModule:
    """SwiGLU or vanilla MLP."""

    def __init__(self, model: ModelConfig):
        self._model = model

    def param_count(self) -> int:
        cfg: MLPConfig = self._model.mlp
        h = self._model.architecture.hidden_size
        ffn = self._model.architecture.ffn_hidden_size

        if cfg.swiglu:
            # SwiGLU: gate + up + down, all of size (h, ffn) or (ffn, h)
            weights = (h * ffn) + (h * ffn) + (ffn * h)
            bias = (2 * ffn + h) if cfg.add_bias_linear else 0
        else:
            # MLP: fc1(h->ffn) + fc2(ffn->h)
            weights = (h * ffn) + (ffn * h)
            bias = (ffn + h) if cfg.add_bias_linear else 0
        return weights + bias


class NormModule:
    """RMSNorm has 1 vector of weights of size ``hidden_size``; LayerNorm has 2 (gain + bias)."""

    def __init__(self, model: ModelConfig):
        self._model = model

    def param_count(self) -> int:
        cfg: NormConfig = self._model.norm
        h = self._model.architecture.hidden_size
        return h if cfg.normalization == "RMSNorm" else 2 * h


def _expert_mlp_param_count(hidden_size: int, ffn_size: int, swiglu: bool, add_bias_linear: bool) -> int:
    if swiglu:
        weights = (hidden_size * ffn_size) * 2 + (ffn_size * hidden_size)
        bias = (2 * ffn_size + hidden_size) if add_bias_linear else 0
    else:
        weights = (hidden_size * ffn_size) + (ffn_size * hidden_size)
        bias = (ffn_size + hidden_size) if add_bias_linear else 0
    return weights + bias


class MoEModule:
    """MoE block: router + ``num_routed_experts`` experts + ``num_shared_experts`` shared experts.

    Routed experts are sharded across the EP group via ``param_count(ep_size=...)``.
    Shared experts and the router are replicated on every EP rank.
    """

    def __init__(self, model: ModelConfig):
        self._model = model

    @property
    def cfg(self) -> MoEConfig:
        return self._model.moe

    def gate_param_count(self) -> int:
        moe = self.cfg
        h = self._model.architecture.hidden_size
        router_bias = moe.num_routed_experts if moe.add_router_bias else 0
        return h * moe.num_routed_experts + router_bias

    def routed_expert_param_count(self) -> int:
        moe = self.cfg
        mlp: MLPConfig = self._model.mlp
        h = self._model.architecture.hidden_size
        return _expert_mlp_param_count(h, moe.moe_ffn_hidden_size, mlp.swiglu, mlp.add_bias_linear)

    def shared_expert_param_count(self) -> int:
        moe = self.cfg
        mlp: MLPConfig = self._model.mlp
        h = self._model.architecture.hidden_size
        return _expert_mlp_param_count(h, moe.moe_ffn_hidden_size, mlp.swiglu, mlp.add_bias_linear)

    def param_count(self, ep_size: int = 1) -> int:
        moe = self.cfg
        if ep_size < 1 or moe.num_routed_experts % ep_size != 0:
            raise ValueError(
                f"num_routed_experts={moe.num_routed_experts} must be divisible by ep_size={ep_size}"
            )
        routed_local = (moe.num_routed_experts // ep_size) * self.routed_expert_param_count()
        shared_total = moe.num_shared_experts * self.shared_expert_param_count()
        return self.gate_param_count() + routed_local + shared_total

    def total_full_param_count(self) -> int:
        """Param count assuming ep_size=1 (i.e. global, no sharding)."""
        return self.param_count(ep_size=1)
