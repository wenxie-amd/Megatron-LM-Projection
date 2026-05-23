# Project Design

## Background

When training models with Megatron-LM, you typically need to analyze the model's GPU memory footprint before you can design a distributed strategy and figure out how many GPUs are required to start training. Megatron-LM exposes a large number of parameters, and different model configurations and training hyperparameters can have a significant impact on runtime memory usage. We therefore want a website that helps users quickly understand memory usage and the memory breakdown for different models under different training configurations, so they can more clearly decide how to design their distributed strategy.

## Goal

Build a Megatron-LM-based projection website that can be deployed on GitHub Pages.

To stay fully static while still running the Python core in the browser, the site loads our Python package via Pyodide (WebAssembly). The same Python code is also usable as a CLI and from unit tests.

## Scope (v1)

- **Training only** (no inference analysis).
- **Precisions**: BF16 and FP8 only. (No FP16, MXFP4, etc.)
- **Models**: `llama3.1_8B` (dense) and `deepseek_v2_lite` (MoE). More models are added by dropping in new YAMLs.
- **Launch script generation**: NVIDIA + Megatron-LM only. AMD + Primus is planned but out of v1.

## Website Functionality

The site is a step-by-step guided flow that walks the user through configuration and selection.

### Step 1: Model Selection

- Provide a dropdown of commonly used models. v1 ships with `llama3.1_8B` and `deepseek_v2_lite`.
- Each model is described by a YAML in `projection/model_configs/`. **Parameter names in YAML mirror Megatron's argument names exactly** (e.g. `num_layers`, `hidden_size`, `num_attention_heads`). Related parameters are grouped together for readability (e.g. attention-related, MoE-related, MLA-related).
- After a model is selected, load that YAML and list its configuration below the dropdown. The "configuration" here refers to Megatron's `TransformerConfig` parameters — list the important ones that describe how the model is composed. This does **not** include `ModelParallelConfig` settings.
- A small subset of these parameters is **user-editable**. In v1 only `num_layers` is editable; more will be opened up later. As soon as the user edits any value, the model is no longer the canonical one and must be visibly labeled as a **proxy model** (e.g. `deepseek_v2_lite (proxy)`).
- Analyze the model based on those parameters and render the overall model structure as well as the structure of a single layer inside the model.
- Analyze per-module parameter counts for the model and visualize them with a pie chart. When the user hovers over a slice, show more detailed information for that module.

### Step 2: Machine Selection

- Show the GPU vendor: NVIDIA or AMD.
- Select the specific GPU model:
  - NVIDIA: `H100`, `H200`, `B200`, `GB200`, `GB300`, `Rubin`.
  - AMD: `MI300X`, `MI325X`, `MI350X`, `MI355X`, `MI455X`.
- Then let the user set the number of GPUs.
- List the detailed spec of the selected GPU, e.g. memory size, BF16 TFLOPS, FP8 TFLOPS, etc.
- After picking a primary GPU, allow the user to optionally pick a second GPU for spec comparison (e.g. a table with one column for the primary GPU and one for the secondary GPU). The primary GPU is still the one used for the subsequent projection.

### Step 3: Training Parameters

This step has three parts.

**1. Distributed strategy** (mostly `ModelParallelConfig`)

- Training precision: BF16 or FP8.
- Common distributed strategies: TP, SP, PP, EP, CP, DP. Expose each as a configurable knob.
- **Pipeline parallelism** supports two mutually exclusive modes:
  - **Layout mode** — set `pipeline-model-parallel-layout` directly. In this mode the user does **not** also set virtual pipeline size; it is implied by the layout.
  - **PP + VPP mode** — set PP size and (optionally) virtual pipeline size. In this mode `num_layers` must be divisible by `pp_size * vpp_size`; otherwise show an error and suggest switching to layout mode.
- **Optimizer / sharding** — single-select among:
  - distributed optimizer
  - Torch FSDP2
  - Megatron FSDP

  Once chosen, list its specific parameters for configuration. These three choices have known conflicts with other distributed strategy parameters (see Megatron's `arguments.py`). The UI must detect and surface those conflicts (e.g. red highlight + explanation of which conflicting flags need to change).

**2. Workload** (directly affects memory and/or throughput)

- `seq_length`
- `micro_batch_size`, `global_batch_size`
- Recompute knobs: `recompute_granularity` (none / selective / full), `recompute_method`, `recompute_num_layers`
- `sequence_parallel` toggle
- MoE-specific knobs that affect memory or throughput (e.g. `moe_grouped_gemm`, `moe_token_dispatcher_type`, `moe_pad_expert_input_to_capacity`)

**3. Hyperparameters** (no effect on memory or throughput; defaults are fine)

- E.g. `lr`, `min_lr`, `train_iters`.
- Provide reasonable defaults; the user can override.

### Step 4: Memory Analysis

- Based on everything from the first three steps, run a memory-usage analysis and breakdown.
- The breakdown should include parameters, activations, gradients, and distributed optimizer state (including main param and optimizer states), and must label the specific datatype for each (fp32, bf16, fp8).
- The actual memory math is implemented from scratch in `projection/core/`, mirroring Megatron's formulas. Unit tests pin the results against the real Megatron-LM (loaded from `third_party/Megatron-LM`) as a gold standard, so the in-browser implementation stays accurate without importing Megatron into the runtime.
- In multi-GPU training, different ranks can have different memory footprints (e.g. with PP enabled, earlier pipeline ranks usually consume more memory). The user provides a **rank list** (e.g. `0, 4, 10, 63`) — up to **8 ranks** — and the page renders a side-by-side comparison of those specific ranks (table + bar chart).

### Step 5: Generate a Runnable Training Script

- **v1**: NVIDIA + Megatron-LM only. Generate a shell launch script that follows Megatron-LM's launch-script conventions, using the configured model and all the choices from Steps 2–3.
- **Future**: when the user picks an AMD GPU in Step 2, generate Primus configuration + launch script instead. Out of scope for v1.

## Website Layout and Style

- Based on the functionality above, propose several easy-to-use and professional styles, with corresponding UI mockups for me to choose from.
- Small requirement: place an explanation panel in the top-right of the page. Every parameter on the page has an info icon; clicking it surfaces details or an answer in that panel — e.g. what the parameter means, and how its derived value was computed.

## Code Layout

### Code style

- Static descriptive data (model configs, GPU specs, default hyperparameters) lives in YAML. The YAML keys mirror Megatron's argument names exactly so users can map them 1:1 to a Megatron run.
- All derivations and auto-generated outputs are produced by Python code in `projection/`. The Python core is a **thin mirror** of Megatron-LM's memory/parameter formulas — independent code, but pinned against the real `third_party/Megatron-LM` in unit tests as a gold standard.
- The same Python package is loaded in the browser via Pyodide, so the site stays static (deployable to GitHub Pages) while still doing all the real computation in Python.

### Directory structure

- Place the Python core code under a `projection/` directory. It can include, for example:
  - `model_configs/` — YAML files describing each model. v1 ships:
    - `llama3_1_8b.yaml` (dense)
    - `deepseek_v2_lite.yaml` (MoE)

    Additional models (`deepseek_v2.yaml`, `deepseek_v3.yaml`, `llama3_1_70b.yaml`, …) can be added by dropping in a new YAML.
  - `gpu_specs/` — YAML files describing GPU hardware specs (memory size, BF16/FP8 TFLOPS, etc.).
  - `core/` — core training class definitions: from `trainer`, `optimizer`, `transformer model`, to `block`, to `layer`, to the smaller modules inside a layer (e.g. attention, MLP, MoE, etc.).
    - `trainer` is composed of a transformer model and an optimizer. The trainer returns the final parameter count, memory total, memory breakdown, etc. The trainer is therefore bound to a specific rank, because different ranks can produce different results.
    - `transformer model` is composed of an embedding, transformer blocks, etc.
    - `transformer block` is in turn composed of layers.
    - `layer` is composed of smaller modules.
    - These Python classes nest layer by layer. Each level owns its own parameters.
    - Each class exposes a common set of methods — e.g. get parameter count, get activation memory, etc.
    - Higher-level classes are computed by composing lower-level ones, which makes breakdown analysis easy. For example, the trainer's memory is the combination of the transformer model and the optimizer.
- The web frontend code can live wherever you think makes the most sense.
