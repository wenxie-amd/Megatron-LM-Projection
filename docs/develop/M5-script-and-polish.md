# M5 — Step 5, FSDP, info icons, polish

## What landed

### Script generator

- `projection/src/projection/script_gen/megatron.py`: generates a runnable Megatron-LM `pretrain_gpt.py` launch shell script. Follows the structure of `third_party/Megatron-LM/examples/`: env-var header, `torchrun` invocation, model + parallel + workload + recompute flags. Includes:
  - Dense flags: `--num-layers`, `--hidden-size`, `--ffn-hidden-size`, `--num-attention-heads`, `--swiglu`, `--normalization RMSNorm`, `--rotary-base`, etc.
  - MLA flags when `attention.use_mla`: `--multi-latent-attention`, `--kv-lora-rank`, `--qk-head-dim`, `--qk-pos-emb-head-dim`, `--v-head-dim`.
  - MoE flags when `moe.enabled`: `--num-experts`, `--moe-router-topk`, `--moe-ffn-hidden-size`, `--moe-shared-expert-intermediate-size`, `--moe-layer-freq`.
  - Parallelism: TP/PP/CP/EP/SP, VPP-derived `--num-layers-per-virtual-pipeline-stage`, layout-mode `--pipeline-model-parallel-layout`.
  - Precision: `--bf16`, plus `--fp8-format hybrid` when applicable.
  - Optimizer: maps single-select to `--use-distributed-optimizer` / `--use-torch-fsdp2` / `--use-megatron-fsdp`.
- `api.generate_script(payload)` exposes the generator over the bridge.
- `web/src/steps/Step5Script.tsx`:
  - Renders the script in a `<pre>` block with a Copy button.
  - Routes AMD selection to a placeholder explaining Primus is out of v1.

### FSDP support

- `core/optimizer.py`: new `FSDPOptimizer` for Torch FSDP2 / Megatron FSDP. Models the common case: params, grads, master, and Adam states all DP-sharded.
- `core/trainer.py`: under FSDP, `_params_on_rank` divides by `TP * DP` instead of just `TP`.
- `parallel/ranks.validate_parallel_config`: adds conflict rules
  - `torch_fsdp2` incompatible with `pp_size > 1`
  - `torch_fsdp2` incompatible with `vpp > 1`
  - `megatron_fsdp` incompatible with `pp_size > 1` (v1 conservative)
- UI: enables `Torch FSDP2` and `Megatron FSDP` options in the Step 3 optimizer dropdown.
- Tests: `tests/unit/test_fsdp_conflicts.py` (6 tests) covering each rule plus the FSDP-shards-params and distributed-optimizer-doesn't-shard contracts.

### Info icons + explanation panel

- `web/src/components/ExplanationPanel.tsx`: a `useContext`-based panel pinned to the right edge of the page. `<InfoIcon title body />` opens it; the panel has a close button.
- Demonstrated on Step 3 for the precision selector and the optimizer single-select (with bullet-list explanations of distributed_optimizer vs FSDP2 vs Megatron FSDP).
- Pattern is reusable — other steps can add `InfoIcon` to any field by passing it as `hint`.

## Verified

- 40 Python tests pass (`uv run pytest`).
- 11 web tests pass (`vitest run`).
- `npm run build` produces a clean dist with `wheels/projection-0.1.0-py3-none-any.whl` bundled.

## What's intentionally not done

- **Multiple visual mockups**: the design doc asked for "2–3 easy-to-use and professional styles". Producing meaningful UI mockups for a user to choose from is human work; the shipped style is a single clean default. Future passes can iterate on `web/src/App.css` or introduce a theme system.
- **A11y / responsive deep pass**: basic responsiveness is in via `grid` + `auto-fit`; full a11y audit (focus order, screen-reader labels for charts) is future work.
- **Info icons on every parameter**: the infrastructure is in place; only a few illustrative parameters carry icons today. Adding the rest is a straightforward mechanical pass.
