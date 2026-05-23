"""Tests for the JSON-in / JSON-out bridge API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from projection.api import (
    MAX_RANKS,
    get_gpu_spec,
    get_model_config,
    list_gpus,
    list_models,
    run_projection,
)

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "llama3_1_8b" / "default_bf16_tp1_pp1.json"


def _default_payload(ranks: list[int] | None = None) -> dict:
    with open(FIXTURE, "r", encoding="utf-8") as f:
        fixture = json.load(f)
    return {
        "model": fixture["model"],
        "parallel": fixture["parallel"],
        "workload": fixture["workload"],
        "ranks": ranks or [0],
    }


def test_list_models_includes_llama() -> None:
    assert "llama3.1_8B" in list_models()


def test_list_gpus_includes_h100() -> None:
    assert "H100" in list_gpus()


def test_get_model_config_round_trips() -> None:
    cfg = get_model_config("llama3.1_8B")
    assert cfg["name"] == "llama3.1_8B"
    assert cfg["architecture"]["num_layers"] == 32


def test_get_gpu_spec_h100_memory() -> None:
    spec = get_gpu_spec("h100")
    assert spec["memory_gb"] == 80
    assert spec["vendor"] == "nvidia"


def test_run_projection_matches_fixture() -> None:
    with open(FIXTURE, "r", encoding="utf-8") as f:
        fixture = json.load(f)
    out = run_projection(_default_payload())
    rank0 = out["rank_reports"][0]
    assert rank0["param_count"] == fixture["expected"]["param_count_total"]
    expected_mem = fixture["expected"]["rank_0_memory_bytes"]
    mem = rank0["memory"]
    assert mem["param_bytes"] == expected_mem["param_bytes"]
    assert mem["activation_bytes"] == expected_mem["activation_bytes"]
    assert mem["optimizer"]["grad_buffer_bytes"] == expected_mem["optimizer_grad_buffer_bytes"]
    assert mem["optimizer"]["main_param_bytes"] == expected_mem["optimizer_main_param_bytes"]
    assert mem["optimizer"]["state_bytes"] == expected_mem["optimizer_state_bytes"]
    assert mem["total_bytes"] == expected_mem["total_bytes"]


def test_run_projection_rejects_too_many_ranks() -> None:
    with pytest.raises(ValueError, match="at most"):
        run_projection(_default_payload(list(range(MAX_RANKS + 1))))


def test_run_projection_rejects_duplicate_ranks() -> None:
    with pytest.raises(ValueError, match="unique"):
        run_projection(_default_payload([0, 0, 1]))


def test_run_projection_supports_inline_model_dict() -> None:
    payload = _default_payload()
    payload["model"] = get_model_config("llama3.1_8B")
    out = run_projection(payload)
    assert out["rank_reports"][0]["param_count"] == 8_030_261_248


def test_run_projection_multi_rank_pp() -> None:
    payload = _default_payload(ranks=[0, 1, 2, 3])
    payload["parallel"] = dict(payload["parallel"])
    payload["parallel"]["pipeline_model_parallel_size"] = 4
    out = run_projection(payload)
    assert [r["partition"]["num_layers_on_rank"] for r in out["rank_reports"]] == [8, 8, 8, 8]
    assert out["rank_reports"][0]["partition"]["has_embedding"] is True
    assert out["rank_reports"][3]["partition"]["has_final_norm"] is True
