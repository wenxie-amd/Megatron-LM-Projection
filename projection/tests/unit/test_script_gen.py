"""Tests for the Megatron launch-script generator."""

from __future__ import annotations

from projection import ParallelConfig, Precision, Workload, load_model_config
from projection.configs import OptimizerKind, TrainingHyperparameters
from projection.script_gen import generate_megatron_script


def _common_args():
    return load_model_config("llama3.1_8B"), TrainingHyperparameters()


def test_llama_script_contains_core_flags() -> None:
    model, hp = _common_args()
    parallel = ParallelConfig(
        precision=Precision.BF16, tensor_model_parallel_size=2, pipeline_model_parallel_size=2
    )
    workload = Workload(seq_length=8192, micro_batch_size=1, global_batch_size=64)
    script = generate_megatron_script(model, parallel, workload, hp, num_gpus=16)

    for flag in [
        "torchrun",
        "pretrain_gpt.py",
        "--num-layers 32",
        "--hidden-size 4096",
        "--ffn-hidden-size 14336",
        "--num-attention-heads 32",
        "--seq-length 8192",
        "--micro-batch-size 1",
        "--global-batch-size 64",
        "--tensor-model-parallel-size 2",
        "--pipeline-model-parallel-size 2",
        "--bf16",
        "--swiglu",
        "--normalization RMSNorm",
        "--position-embedding-type rope",
        "--rotary-base 500000",
        "--use-rotary-position-embeddings",
        "--use-distributed-optimizer",
        "--untie-embeddings-and-output-weights",
        "--disable-bias-linear",
        "--group-query-attention",
        "--num-query-groups 8",
    ]:
        assert flag in script, f"expected {flag!r} in generated script"


def test_fp8_emits_fp8_format() -> None:
    model, hp = _common_args()
    parallel = ParallelConfig(precision=Precision.FP8)
    workload = Workload(seq_length=4096, micro_batch_size=1, global_batch_size=16)
    script = generate_megatron_script(model, parallel, workload, hp, num_gpus=8)
    assert "--fp8-format hybrid" in script
    assert "--bf16" in script


def test_mla_script_contains_mla_flags() -> None:
    model = load_model_config("deepseek_v2_lite")
    hp = TrainingHyperparameters()
    parallel = ParallelConfig(expert_model_parallel_size=8)
    workload = Workload(seq_length=4096, micro_batch_size=1, global_batch_size=64)
    script = generate_megatron_script(model, parallel, workload, hp, num_gpus=8)
    assert "--multi-latent-attention" in script
    assert "--kv-lora-rank 512" in script
    assert "--qk-head-dim 128" in script
    assert "--qk-pos-emb-head-dim 64" in script
    assert "--v-head-dim 128" in script
    assert "--num-experts 64" in script
    assert "--moe-router-topk 6" in script
    assert "--moe-ffn-hidden-size 1408" in script
    assert "--expert-model-parallel-size 8" in script


def test_fsdp_emits_fsdp_flag() -> None:
    model, hp = _common_args()
    parallel = ParallelConfig(optimizer_kind=OptimizerKind.TORCH_FSDP2)
    workload = Workload(seq_length=4096, micro_batch_size=1, global_batch_size=64)
    script = generate_megatron_script(model, parallel, workload, hp, num_gpus=8)
    assert "--use-torch-fsdp2" in script
    assert "--use-distributed-optimizer" not in script


def test_script_defaults_to_mock_data_and_no_save() -> None:
    model, hp = _common_args()
    parallel = ParallelConfig()
    workload = Workload(seq_length=4096, micro_batch_size=1, global_batch_size=64)
    script = generate_megatron_script(model, parallel, workload, hp, num_gpus=8)
    assert "--mock-data" in script
    assert "--log-interval 1" in script
    assert "--no-save-optim" in script
    assert "--tensorboard-dir" not in script
    assert "--save " not in script
    assert "wandb" not in script.lower()


def test_script_lines_have_consistent_indentation() -> None:
    """All ARGS lines should share the same leading whitespace."""
    model, hp = _common_args()
    parallel = ParallelConfig(tensor_model_parallel_size=2)
    workload = Workload(seq_length=4096, micro_batch_size=1, global_batch_size=64)
    script = generate_megatron_script(model, parallel, workload, hp, num_gpus=8)
    lines = script.splitlines()
    args_open = lines.index("ARGS=(")
    args_close = lines.index(")", args_open)
    inner = lines[args_open + 1 : args_close]
    indents = {len(line) - len(line.lstrip(" ")) for line in inner if line.strip()}
    assert indents == {4}, f"expected all ARGS lines indented by 4 spaces, got {indents}"


def test_precision_aware_optimizer_flags() -> None:
    model, hp = _common_args()
    parallel = ParallelConfig(
        use_precision_aware_optimizer=True,
        optimizer_main_param_dtype="bf16",
        optimizer_exp_avg_dtype="bf16",
        optimizer_exp_avg_sq_dtype="fp32",
        optimizer_main_grad_dtype="fp32",
    )
    workload = Workload(seq_length=4096, micro_batch_size=1, global_batch_size=64)
    script = generate_megatron_script(model, parallel, workload, hp, num_gpus=8)
    assert "--use-precision-aware-optimizer" in script
    assert "--main-params-dtype bf16" in script
    assert "--exp-avg-dtype bf16" in script
    assert "--exp-avg-sq-dtype fp32" in script
    assert "--main-grads-dtype fp32" in script


def test_recompute_full_emits_method_and_num_layers() -> None:
    model, hp = _common_args()
    parallel = ParallelConfig()
    workload = Workload(
        seq_length=4096,
        micro_batch_size=1,
        global_batch_size=64,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=8,
    )
    script = generate_megatron_script(model, parallel, workload, hp, num_gpus=8)
    assert "--recompute-granularity full" in script
    assert "--recompute-method uniform" in script
    assert "--recompute-num-layers 8" in script
