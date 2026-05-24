"""DeepSeek-V4 specific tests.

Covers:

* YAML loadability for both V4 variants.
* Per-layer V4 attention parameter count agrees with hand-computed formula
  (Q LoRA + single-latent KV + grouped O + Compressor + Indexer).
* mHC HyperMixer / HyperHead element counts match the Primus module layout.
* Hash routing buffer has the right shape.
* Total parameter count for V4-Flash and V4-Pro is within 5% of the official
  Total parameter announcement (284B / 1.6T) when run with PP=1 / TP=1.
* Generated Megatron script contains V4-specific flags.
"""

from __future__ import annotations

import pytest

from projection.configs import (
    ParallelConfig,
    Precision,
    TrainingHyperparameters,
    Workload,
)
from projection.core.deepseek_v4 import (
    hash_routing_buffer_per_layer,
    hyper_head_param_count,
    hyper_mixer_param_count,
    normalize_compress_ratios,
    v4_attention_base_param_count,
    v4_attention_param_count_per_layer,
)
from projection.core.trainer import Trainer
from projection.loader import load_model_config
from projection.script_gen.megatron import generate_megatron_script

V4_MODELS = ["deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Pro"]


@pytest.mark.parametrize("name", V4_MODELS)
def test_loadable(name: str) -> None:
    m = load_model_config(name)
    assert m.is_v4
    assert m.attention.use_deepseek_v4
    assert m.hybrid_attention is not None
    assert m.hyper_connection.enabled
    assert m.hyper_connection.hc_mult == 4
    assert m.mtp.num_layers == 1


@pytest.mark.parametrize("name", V4_MODELS)
def test_compress_ratios_length(name: str) -> None:
    """Schedule must cover all decoder layers plus optionally the MTP depths."""
    m = load_model_config(name)
    assert m.hybrid_attention is not None
    expected = m.architecture.num_layers + m.mtp.num_layers
    assert len(m.hybrid_attention.compress_ratios) == expected

    decoder, mtp = normalize_compress_ratios(m)
    assert len(decoder) == m.architecture.num_layers
    assert len(mtp) == m.mtp.num_layers
    assert all(r in (0, 4, 128) for r in decoder)


def test_v4_flash_attention_base_param_count() -> None:
    """Per techblog §1.2 + the V4 FLOPs patch: Q LoRA + single-latent KV +
    grouped low-rank O + per-head attn_sink."""
    m = load_model_config("deepseek-ai/DeepSeek-V4-Flash")
    h, n, d = 4096, 64, 512
    q_lora = 1024
    o_lora = 1024
    og = 8
    expected = (
        h * q_lora  # wq_a
        + q_lora * n * d  # wq_b
        + q_lora  # q_norm
        + h * d  # wkv (single latent)
        + d  # kv_norm
        + n * d * o_lora  # wo_a (grouped low-rank)
        + og * o_lora * h  # wo_b
        + n  # attn_sink (per-head scalar)
    )
    assert v4_attention_base_param_count(m) == expected


def test_v4_flash_compressor_indexer_extras() -> None:
    """CSA layer has Compressor + Indexer; HCA layer has Compressor only; dense
    layer has neither."""
    m = load_model_config("deepseek-ai/DeepSeek-V4-Flash")
    base = v4_attention_base_param_count(m)

    dense = v4_attention_param_count_per_layer(m, 0)
    hca = v4_attention_param_count_per_layer(m, 128)
    csa = v4_attention_param_count_per_layer(m, 4)

    assert dense == base
    assert hca > dense
    assert csa > hca
    # CSA extras include the Indexer; HCA does not.
    assert (csa - dense) > (hca - dense)


def test_hyper_mixer_and_head_shapes() -> None:
    """Match :class:`primus.backends.megatron.core.transformer.hyper_connection.HyperMixer`
    and ``HyperHead`` exactly."""
    h = 4096
    k = 4
    # HyperMixer: fn (k*h -> (2+k)*k) + scale[3] + base[(2+k)*k]
    out_dim = (2 + k) * k
    expected_mixer = k * h * out_dim + 3 + out_dim
    assert hyper_mixer_param_count(h, k) == expected_mixer

    # HyperHead: fn (k*h -> k) + scale[1] + base[k]
    expected_head = k * h * k + 1 + k
    assert hyper_head_param_count(h, k) == expected_head


def test_hash_routing_buffer_size() -> None:
    """``tid2eid`` is ``[vocab_size, moe_router_topk]`` int32."""
    m = load_model_config("deepseek-ai/DeepSeek-V4-Flash")
    buf = hash_routing_buffer_per_layer(m)
    assert buf.elements == m.architecture.vocab_size * m.moe.moe_router_topk
    assert buf.bytes == 4 * buf.elements


@pytest.mark.parametrize(
    "name,expected_total_b,tolerance_b",
    [
        # Tolerances are wider than the announced totals because the announced
        # 284B / 1.6T may or may not include MTP; our model always counts MTP and
        # the trunk HyperHead. We expect to land within 3% of the announced
        # active+inactive total.
        ("deepseek-ai/DeepSeek-V4-Flash", 284e9, 15e9),
        ("deepseek-ai/DeepSeek-V4-Pro", 1.6e12, 60e9),
    ],
)
def test_total_param_count_matches_announcement(
    name: str, expected_total_b: float, tolerance_b: float
) -> None:
    m = load_model_config(name)
    p = ParallelConfig(
        precision=Precision.BF16,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        data_parallel_size=1,
    )
    w = Workload(seq_length=4096, micro_batch_size=1, global_batch_size=1)
    t = Trainer(m, p, w, global_rank=0)
    r = t.report()
    assert abs(r.param_count - expected_total_b) < tolerance_b, (
        f"{name}: got {r.param_count/1e9:.2f}B, expected ~{expected_total_b/1e9:.0f}B"
        f" (tolerance {tolerance_b/1e9:.0f}B)"
    )


def test_v4_script_emits_v4_flags() -> None:
    """Script generator must include the V4-specific flags (compress-ratios,
    hc-mult, attn-sink, etc.). The generated script is a blueprint — not
    expected to run unmodified on stock Megatron-LM today."""
    m = load_model_config("deepseek-ai/DeepSeek-V4-Flash")
    p = ParallelConfig(
        precision=Precision.BF16,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=8,
        expert_model_parallel_size=8,
        data_parallel_size=1,
    )
    w = Workload(
        seq_length=4096,
        micro_batch_size=1,
        global_batch_size=256,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )
    script = generate_megatron_script(m, p, w, TrainingHyperparameters(), num_gpus=8)
    for flag in (
        "--hybrid-attention-enabled",
        "--use-deepseek-v4",
        "--q-lora-rank 1024",
        "--o-lora-rank 1024",
        "--o-groups 8",
        "--compress-ratios",
        "--attn-sliding-window 128",
        "--attn-sink",
        "--index-topk 512",
        "--hc-mult 4",
        "--mtp-num-layers 1",
        "--num-hash-layers 3",
        "--moe-router-score-function sqrtsoftplus",
        "--swiglu-limit 10.0",
    ):
        assert flag in script, f"missing flag in V4 script: {flag!r}"

    # Ensure we did NOT emit --group-query-attention (V4 uses single-latent KV,
    # which is not the same as GQA).
    assert "--group-query-attention" not in script


def test_v4_activation_scales_with_hc_mult() -> None:
    """Per techblog §2: V4's mHC packs ``hc_mult`` parallel streams into the
    sequence axis, so per-layer activation roughly scales with hc_mult."""
    from projection.core.activations import total_activation_bytes_for_rank

    m = load_model_config("deepseek-ai/DeepSeek-V4-Flash")
    p = ParallelConfig(precision=Precision.BF16, tensor_model_parallel_size=1, data_parallel_size=1)
    w = Workload(seq_length=4096, micro_batch_size=1, global_batch_size=1)
    bytes_v4 = total_activation_bytes_for_rank(
        m,
        w,
        p,
        num_layers_on_rank=4,
        pp_rank=0,
        is_first_pp=True,
        is_last_pp=True,
        num_microbatches=1,
        num_dense_layers_on_rank=0,
        num_moe_layers_on_rank=4,
    )
    # A non-V4 model with same shape would not get the hc_mult multiplier.
    # We pick V3 (MLA) as a control: same MoE family, same hidden size order.
    m3 = load_model_config("deepseek-ai/DeepSeek-V3")
    bytes_v3 = total_activation_bytes_for_rank(
        m3,
        w,
        p,
        num_layers_on_rank=4,
        pp_rank=0,
        is_first_pp=True,
        is_last_pp=True,
        num_microbatches=1,
    )
    # V4 activations should be substantially larger than V3 for the same layer
    # count at the same hidden_size, primarily because of the hc_mult=4 axis.
    assert bytes_v4 > bytes_v3
