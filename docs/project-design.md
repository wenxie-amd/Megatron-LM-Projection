# Project Design

## Background

When training models with Megatron-LM, you typically need to analyze the model's GPU memory footprint before you can design a distributed strategy and figure out how many GPUs are required to start training. Megatron-LM exposes a large number of parameters, and different model configurations and training hyperparameters can have a significant impact on runtime memory usage. We therefore want a website that helps users quickly understand memory usage and the memory breakdown for different models under different training configurations, so they can more clearly decide how to design their distributed strategy.

## Goal

Build a Megatron-LM-based projection website that can be deployed on GitHub.

## Website Functionality

The site is a step-by-step guided flow that walks the user through configuration and selection.

### Step 1: Model Selection

- Provide a dropdown that includes several commonly used models, e.g. `llama3.1_8B`, `deepseek_v2_lite`, `deepseek_v3`.
- After a model is selected, automatically list its basic configuration below the dropdown. The "configuration" here refers to Megatron's `TransformerConfig` parameters — list the important ones that describe how the model is composed. This does **not** include `ModelParallelConfig` settings.
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

This step has two parts.

**Distributed strategy**

- Mainly the parameters in `ModelParallelConfig`. Start by letting the user choose the training precision: BF16 or FP8 training. Then expose the common distributed strategies, e.g. TP, SP, PP, EP, CP, DP, etc. PP supports two modes: direct setting, or setting via a layout.
- Let the user choose between distributed optimizer, Torch FSDP2, or Megatron FSDP. Once chosen, list the corresponding parameters so the user can configure them.

**Hyperparameters**

- E.g. `lr`, `min_lr`, `train_iters`, etc.
- Provide reasonable defaults for these — they generally have no effect on training throughput.

### Step 4: Memory Analysis

- Based on everything from the first three steps, run a memory-usage analysis and breakdown.
- The breakdown should include parameters, activations, gradients, and distributed optimizer state (including main param and optimizer states), and must label the specific datatype for each (fp32, bf16, fp16, etc.).
- The actual memory math should reference and learn from Megatron's implementation.
- Note: in multi-GPU training, different ranks can have different memory footprints. For example, with PP enabled, earlier pipeline ranks consume more memory. We therefore also need a per-rank memory comparison view — a table, bar chart, or any other suitable visualization.

### Step 5: Generate a Runnable Training Script

- Following Megatron-LM's launch-script conventions, generate a shell launch script for the currently configured model.

## Website Layout and Style

- Based on the functionality above, propose several easy-to-use and professional styles, with corresponding UI mockups for me to choose from.
- Small requirement: place an explanation panel in the top-right of the page. Every parameter on the page has an info icon; clicking it surfaces details or an answer in that panel — e.g. what the parameter means, and how its derived value was computed.

## Code Layout

### Code style

- Define models, hardware specs, and training hyperparameters in Python.
- Derivations and auto-generated outputs are produced by invoking Python.

### Directory structure

- Place the Python core code under a `projection/` directory. It can include, for example:
  - `model_configs/` — YAML files for common models:
    - `deepseek_v2.yaml`
    - `deepseek_v2_lite.yaml`
    - `deepseek_v3.yaml`
    - `llama3_1_8b.yaml`
    - `llama3_1_70b.yaml`
  - `core/` — core training class definitions: from `trainer`, `optimizer`, `transformer model`, to `block`, to `layer`, to the smaller modules inside a layer (e.g. attention, MLP, MoE, etc.).
    - `trainer` is composed of a transformer model and an optimizer. The trainer returns the final parameter count, memory total, memory breakdown, etc. The trainer is therefore bound to a specific rank, because different ranks can produce different results.
    - `transformer model` is composed of an embedding, transformer blocks, etc.
    - `transformer block` is in turn composed of layers.
    - `layer` is composed of smaller modules.
    - These Python classes nest layer by layer. Each level owns its own parameters.
    - Each class exposes a common set of methods — e.g. get parameter count, get activation memory, etc.
    - Higher-level classes are computed by composing lower-level ones, which makes breakdown analysis easy. For example, the trainer's memory is the combination of the transformer model and the optimizer.
- The web frontend code can live wherever you think makes the most sense.
