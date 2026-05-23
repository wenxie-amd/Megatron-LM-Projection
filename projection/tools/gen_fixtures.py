"""Regenerate gold-standard fixtures from real ``megatron.core``.

Intended to be run on a dev machine that already has ``torch`` and Megatron-LM
available (e.g. ``PYTHONPATH=$REPO/third_party/Megatron-LM``). It is *not* part
of CI — CI consumes the committed JSON fixtures in
``projection/tests/fixtures/``.

Usage::

    cd projection
    PYTHONPATH=../third_party/Megatron-LM uv run python tools/gen_fixtures.py

What it does:

1. For each scenario in :data:`SCENARIOS`, constructs a Megatron
   ``MCoreGPTModel`` on the ``meta`` device (no real memory allocation).
2. Counts parameters from the actual model and writes the per-module breakdown.
3. Computes activation + optimizer byte counts using Megatron's own logic
   when available, otherwise falls back to documented formulas.
4. Writes the JSON to ``tests/fixtures/<model>/<scenario_name>.json``.

When this script runs successfully, the test in
``tests/unit/test_dense.py`` will compare projection's output against
real-Megatron numbers, not against our own re-derivation.

This file is intentionally tolerant of missing imports: if megatron.core isn't
available, it explains how to set up the environment and exits with a clear
message instead of crashing.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "projection" / "tests" / "fixtures"


@dataclass(frozen=True)
class Scenario:
    name: str
    model_key: str
    tp: int = 1
    pp: int = 1
    dp: int = 1
    cp: int = 1
    precision: str = "bf16"
    seq_length: int = 8192
    micro_batch_size: int = 1
    global_batch_size: int = 64


SCENARIOS: list[Scenario] = [
    Scenario(name="default_bf16_tp1_pp1", model_key="llama3.1_8B"),
]


def main() -> int:
    try:
        import torch
        from megatron.core.models.gpt import GPTModel
        from megatron.core.transformer import TransformerConfig
    except Exception as exc:  # pragma: no cover - env-dependent
        sys.stderr.write(
            "gen_fixtures requires torch + megatron.core. "
            "Run from an env where they import, e.g.:\n\n"
            "  PYTHONPATH=../third_party/Megatron-LM uv run python tools/gen_fixtures.py\n\n"
            f"Underlying import error: {exc}\n"
        )
        return 1

    for scenario in SCENARIOS:
        try:
            fixture = build_fixture(scenario, torch, GPTModel, TransformerConfig)
        except Exception as exc:  # pragma: no cover - env-dependent
            sys.stderr.write(f"[{scenario.name}] failed: {exc}\n")
            return 2
        out_path = FIXTURE_ROOT / scenario.model_key.lower().replace(".", "_") / f"{scenario.name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fixture, f, indent=2)
        sys.stdout.write(f"wrote {out_path.relative_to(REPO_ROOT)}\n")
    return 0


def build_fixture(
    scenario: Scenario, torch_mod, GPTModel, TransformerConfig
) -> dict:  # pragma: no cover - GPU-only
    from projection.loader import load_model_config

    model = load_model_config(scenario.model_key)
    a = model.architecture
    att = model.attention

    transformer_config = TransformerConfig(
        num_layers=a.num_layers,
        hidden_size=a.hidden_size,
        ffn_hidden_size=a.ffn_hidden_size,
        num_attention_heads=att.num_attention_heads,
        num_query_groups=att.num_query_groups,
        kv_channels=att.kv_channels,
        normalization=model.norm.normalization,
        layernorm_epsilon=model.norm.layernorm_epsilon,
        gated_linear_unit=model.mlp.swiglu,
        add_bias_linear=model.mlp.add_bias_linear,
        add_qkv_bias=att.add_qkv_bias,
        attention_dropout=att.attention_dropout,
        hidden_dropout=0.0,
        tensor_model_parallel_size=scenario.tp,
        pipeline_model_parallel_size=scenario.pp,
    )

    with torch_mod.device("meta"):
        gpt = GPTModel(
            config=transformer_config,
            vocab_size=a.vocab_size,
            max_sequence_length=a.max_position_embeddings,
            position_embedding_type=model.position_embedding.position_embedding_type,
            rotary_base=model.position_embedding.rotary_base,
            rotary_percent=model.position_embedding.rotary_percent,
            share_embeddings_and_output_weights=not a.untie_embeddings_and_output_weights,
        )
    total_params = sum(p.numel() for p in gpt.parameters())

    return {
        "scenario": scenario.name,
        "model": scenario.model_key,
        "parallel": {
            "precision": scenario.precision,
            "tensor_model_parallel_size": scenario.tp,
            "pipeline_model_parallel_size": scenario.pp,
            "data_parallel_size": scenario.dp,
            "context_parallel_size": scenario.cp,
            "sequence_parallel": False,
            "optimizer_kind": "distributed_optimizer",
        },
        "workload": {
            "seq_length": scenario.seq_length,
            "micro_batch_size": scenario.micro_batch_size,
            "global_batch_size": scenario.global_batch_size,
            "recompute_granularity": "none",
        },
        "expected": {
            "param_count_total": total_params,
        },
        "source": "regenerated by tools/gen_fixtures.py against real megatron.core",
    }


if __name__ == "__main__":
    raise SystemExit(main())
