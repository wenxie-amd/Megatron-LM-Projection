# M1 — Python core (dense path)

## What landed

- `projection/src/projection/configs.py`: pydantic models. Field names mirror Megatron's argument names 1:1 (`num_layers`, `hidden_size`, …). Configs include `Architecture`, `Attention`, `MLP`, `Norm`, `PositionEmbedding`, `Workload`, `ParallelConfig`, `TrainingHyperparameters`, `GPUSpec`.
- `loader.py`: loads YAMLs either from bundled package data (`projection.model_configs`, `projection.gpu_specs`) or arbitrary paths.
- `core/modules.py`: leaf classes (`EmbeddingModule`, `AttentionModule`, `MLPModule`, `NormModule`) with `param_count()`. Attention supports both MHA/GQA and MLA (MLA wired in M4 but the formulas live here).
- `core/layer.py`, `core/block.py`, `core/model.py`: `TransformerLayer` → `TransformerBlock` (a PP-rank slice of layers) → `TransformerModel`.
- `core/activations.py`: per-layer activation byte formulas from Korthikanti et al. 2022, with TP/SP/CP scaling and recompute granularity.
- `core/optimizer.py`: `DistributedOptimizer` (distributed Adam, fp32 grad accumulation by default).
- `core/trainer.py`: `Trainer(global_rank=…)` composes the above and returns a `TrainerReport` (param count, breakdown, per-rank memory).
- `parallel/ranks.py`: `decompose_rank` (default `tp-cp-ep-dp-pp` ordering), `validate_parallel_config`, `layers_per_pp_stage`, `partition_for_rank`, plus the `RankCoord` and `ModelPartition` dataclasses.
- Initial YAMLs: `model_configs/llama3_1_8b.yaml`, `gpu_specs/h100.yaml`.
- Tests: 13 in `tests/unit/test_dense.py` covering param-count exactness, PP/VPP/layout validation, and rank decomposition.

## The Llama 3.1 8B pin

The total parameter count is pinned to **8,030,261,248** — Meta's published value for Llama 3.1 8B. This is independently verifiable from the public spec, so the test is a real check, not a self-reference.

## `tools/gen_fixtures.py`

Regenerates the JSON fixtures by constructing a real `megatron.core` model on the meta device. Gated behind a `try / except ImportError` so the file is always parseable; intended for occasional dev-box runs when the Megatron submodule bumps. CI does not run it.

## Known limitations

- Activation memory uses the simplified formula (no per-microbatch-vs-pipeline-bubble adjustment, no FP8-activation savings); refined as fixtures land.
- `vocab_size` padding is per Megatron's `make_vocab_size_divisible_by * TP`; we don't yet model `tensor-model-parallel-output-grad-allreduce` style transforms.
