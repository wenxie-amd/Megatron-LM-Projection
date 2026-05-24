"""DeepSeek-V4 specific parameter-count helpers.

V4 introduces three things that don't fit the GPT / MLA / GQA mold and
therefore live in their own module:

1. **Hybrid attention** — per layer one of CSA (``cr=4``), HCA (``cr=128``),
   or dense + SWA (``cr=0``). All three share a single-latent KV
   (``K = V = head_dim`` broadcast to all query heads), a Q LoRA path, and a
   grouped low-rank O projection. The compressed branches additionally own a
   :class:`Compressor` (``cr ∈ {4, 128}``) and CSA layers own an
   :class:`Indexer`.
2. **mHC** — every transformer layer carries 2 small ``HyperMixer`` modules
   (attention sub-block + FFN sub-block). A single ``HyperHead`` collapses
   the K streams at the trunk end, and one more lives in each MTP depth
   (``use_separate_hc_head=True``).
3. **Hash routing** — the first ``num_hash_layers`` MoE layers replace the
   topk router with a non-trainable ``tid2eid`` int32 buffer (``vocab_size
   × moe_router_topk``). Returned separately so callers can flag it as
   buffer memory rather than trainable parameter memory.

Formulas mirror :mod:`primus.backends.megatron.patches.deepseek_v4_flops_patches`
(see ``third_party/Primus``); per-layer accounting in
``deepseek-v4/develop/techblog/01-deepseek-v4-architecture-deep-dive.md``.

All counts here are **element counts** (not bytes). Precision-aware byte
multipliers are applied by the caller (:class:`~projection.core.optimizer`).
HC parameters are fp32 on the real model; that delta is small and is left to
the optimizer / activation accounting layers — here we just return the
element count.
"""

from __future__ import annotations

from dataclasses import dataclass

from projection.configs import HybridAttentionConfig, ModelConfig


def _required(value: int | None, field_name: str) -> int:
    if value is None:
        raise ValueError(f"DeepSeek-V4 attention requires {field_name!r} to be set")
    return value


# ---------------------------------------------------------------------------
# Attention: per-layer Q LoRA + single-latent KV + grouped O
# ---------------------------------------------------------------------------


def v4_attention_base_param_count(model: ModelConfig) -> int:
    """V4 attention projections (independent of ``compress_ratio``).

    * Q LoRA: ``linear_q_down_proj`` (``h -> q_lora_rank``) +
      ``linear_q_up_proj`` (``q_lora_rank -> n*d``) + ``q_norm`` (RMSNorm).
    * Single-latent KV: ``linear_kv`` (``h -> head_dim``) + ``kv_norm``.
      One projection produces both K and V, broadcast across all query heads.
    * Grouped low-rank O: ``linear_o_a`` (``(n*d/o_groups) -> o_groups*o_lora_rank``)
      + ``linear_o_b`` (``o_groups*o_lora_rank -> h``). Falls back to a flat
      ``(n*d) -> h`` projection when ``o_lora_rank`` is unset.
    * ``attn_sink``: per-head learnable scalar (``n`` params).
    """
    att = model.attention
    h = model.architecture.hidden_size
    n = att.num_attention_heads
    d = att.head_dim(h)
    q_rank = _required(att.q_lora_rank, "q_lora_rank")
    nd = n * d

    q_path = h * q_rank + q_rank * nd + q_rank  # wq_a + wq_b + q_norm
    kv_path = h * d + d  # wkv + kv_norm

    if att.o_lora_rank and att.o_lora_rank > 0:
        og = max(1, att.o_groups)
        o_path = nd * att.o_lora_rank + og * att.o_lora_rank * h
    else:
        o_path = nd * h

    hyb = model.hybrid_attention
    sink = n if (hyb is not None and hyb.attn_sink) else 0

    return q_path + kv_path + o_path + sink


def _compressor_param_count(*, hidden: int, head_dim: int, ratio: int) -> int:
    """Compressor (cr ∈ {4, 128}).

    Layout per :mod:`primus.backends.megatron.core.transformer.compressor`:

    * ``wkv``: ``h -> coff*head_dim`` (coff=2 for cr=4 overlap, 1 for cr=128).
    * ``wgate``: same shape.
    * ``ape``: ``[coff*ratio, head_dim]`` learnable absolute position embedding.
    * ``kv_norm``: RMSNorm over ``head_dim``.
    """
    if ratio <= 0:
        return 0
    coff = 2 if ratio == 4 else 1
    proj_out = coff * head_dim
    wkv = hidden * proj_out
    wgate = hidden * proj_out
    ape_len = 2 * ratio if coff == 2 else ratio
    ape = ape_len * head_dim
    kv_norm = head_dim
    return wkv + wgate + ape + kv_norm


def _indexer_param_count(*, hidden: int, hyb: HybridAttentionConfig) -> int:
    """Indexer (cr=4 only). Includes the mini-Compressor at index_head_dim, ratio=4."""
    ihd = hyb.index_head_dim
    inh = hyb.index_n_heads
    if ihd <= 0 or inh <= 0:
        return 0

    dq_rank = ihd
    w_dq = hidden * dq_rank
    w_iuq = dq_rank * inh * ihd
    w_w = hidden * inh

    mini = _compressor_param_count(hidden=hidden, head_dim=ihd, ratio=4)
    return w_dq + w_iuq + w_w + mini


def v4_attention_param_count_per_layer(model: ModelConfig, compress_ratio: int) -> int:
    """Total V4 attention param count for one layer of branch ``compress_ratio``."""
    if not model.is_v4:
        raise ValueError("v4_attention_param_count_per_layer requires a V4 model")
    hyb = model.hybrid_attention
    assert hyb is not None  # for type checker
    h = model.architecture.hidden_size
    d = model.attention.head_dim(h)

    base = v4_attention_base_param_count(model)
    extra = 0
    if compress_ratio != 0:
        extra += _compressor_param_count(hidden=h, head_dim=d, ratio=compress_ratio)
    if compress_ratio == 4:
        extra += _indexer_param_count(hidden=h, hyb=hyb)
    return base + extra


# ---------------------------------------------------------------------------
# mHC: per-layer HyperMixer + trunk / MTP HyperHead
# ---------------------------------------------------------------------------


def hyper_mixer_param_count(hidden: int, hc_mult: int) -> int:
    """One ``HyperMixer``: ``fn`` (K*D → (2+K)*K) + ``scale[3]`` + ``base[out_dim]``."""
    if hc_mult < 2:
        return 0
    out_dim = (2 + hc_mult) * hc_mult
    fn = hc_mult * hidden * out_dim
    scale = 3
    base = out_dim
    return fn + scale + base


def hyper_head_param_count(hidden: int, hc_mult: int) -> int:
    """One ``HyperHead``: ``fn`` (K*D → K) + ``scale`` (scalar) + ``base[K]``."""
    if hc_mult < 2:
        return 0
    fn = hc_mult * hidden * hc_mult
    scale = 1
    base = hc_mult
    return fn + scale + base


def hc_per_layer_param_count(model: ModelConfig) -> int:
    """Two ``HyperMixer`` instances per V4 transformer layer (attn + FFN)."""
    hc = model.hyper_connection
    if not hc.enabled:
        return 0
    h = model.architecture.hidden_size
    return 2 * hyper_mixer_param_count(h, hc.hc_mult)


# ---------------------------------------------------------------------------
# Hash routing buffer (non-trainable int32)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HashRoutingBuffer:
    """The ``tid2eid`` lookup table for one hash-routed MoE layer.

    Element count and byte count are reported separately because the buffer is
    int32 (4 bytes) and has no gradient / optimizer state — callers should
    attribute it to ``params`` byte memory but NOT to gradient or optimizer
    state memory.
    """

    elements: int
    bytes: int  # int32 = 4 bytes / element


def hash_routing_buffer_per_layer(model: ModelConfig) -> HashRoutingBuffer:
    """``[vocab_size, moe_router_topk]`` int32 lookup buffer."""
    if model.moe.num_hash_layers <= 0:
        return HashRoutingBuffer(elements=0, bytes=0)
    n = model.architecture.vocab_size * max(1, model.moe.moe_router_topk)
    return HashRoutingBuffer(elements=n, bytes=4 * n)


# ---------------------------------------------------------------------------
# Layer-schedule helper
# ---------------------------------------------------------------------------


def normalize_compress_ratios(model: ModelConfig) -> tuple[list[int], list[int]]:
    """Return ``(decoder_ratios, mtp_ratios)`` each padded to expected length.

    Mirrors :func:`primus.backends.megatron.patches.deepseek_v4_flops_patches._normalize_layer_ratios`.
    YAML ``compress_ratios`` may carry ``num_layers``, ``num_layers + mtp_num_layers``,
    or be shorter (in which case the trailing slot is padded with the last value,
    or ``0`` if empty).
    """
    if not model.is_v4:
        raise ValueError("normalize_compress_ratios requires a V4 model")
    hyb = model.hybrid_attention
    assert hyb is not None
    n = model.architecture.num_layers
    m = model.mtp.num_layers
    parsed = list(hyb.compress_ratios)
    if len(parsed) == n + m:
        return parsed[:n], parsed[n:]
    if len(parsed) == n:
        return parsed, [0] * m
    if len(parsed) > n:
        return parsed[:n], [0] * m
    pad = parsed[-1] if parsed else 0
    return parsed + [pad] * (n - len(parsed)), [0] * m
