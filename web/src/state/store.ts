/**
 * Global wizard state.
 *
 * Step components read slices and dispatch updates. Step 4 lazily kicks off a
 * projection through the Pyodide bridge. Keeping the state in one reducer
 * makes the data flow easy to follow and easy to test.
 *
 * Convention: DP, EDP, and GA are *derived* from world_size, TP, PP, CP, EP,
 * MBS, GBS — never user-set. The reducer keeps a denormalised
 * ``data_parallel_size`` inside ``parallel`` so the Python side (which still
 * accepts an explicit DP) gets consistent input.
 */

import type {
  OptimizerDtype,
  OptimizerKind,
  ParallelConfig,
  Precision,
  ProjectionOutput,
  Workload,
} from "../pyodide/types";

export interface ModelConfigView {
  name: string;
  description?: string | null;
  architecture: {
    num_layers: number;
    hidden_size: number;
    ffn_hidden_size: number;
    vocab_size: number;
    max_position_embeddings: number;
    untie_embeddings_and_output_weights: boolean;
    make_vocab_size_divisible_by: number;
  };
  attention: {
    num_attention_heads: number;
    num_query_groups?: number | null;
    kv_channels?: number | null;
    attention_dropout: number;
    add_qkv_bias: boolean;
    use_mla: boolean;
    q_lora_rank?: number | null;
    kv_lora_rank?: number | null;
    qk_nope_head_dim?: number | null;
    qk_rope_head_dim?: number | null;
    v_head_dim?: number | null;
  };
  mlp: { swiglu: boolean; add_bias_linear: boolean };
  moe: {
    enabled: boolean;
    moe_ffn_hidden_size: number;
    num_routed_experts: number;
    num_shared_experts: number;
    moe_router_topk: number;
    moe_layer_freq: number;
    first_k_dense_replace: number;
    add_router_bias: boolean;
  };
  norm: { normalization: "LayerNorm" | "RMSNorm"; layernorm_epsilon: number };
  position_embedding: {
    position_embedding_type: "learned_absolute" | "rope" | "none";
    rotary_base: number;
    rotary_percent: number;
  };
}

export interface GpuSpecView {
  name: string;
  vendor: "nvidia" | "amd";
  memory_gb: number;
  bf16_tflops: number;
  fp8_tflops: number;
  bandwidth_gbps: number;
}

export type PpMode = "direct" | "layout";

export interface State {
  bridgeStatus: "loading" | "ready" | "error";
  bridgeError?: string;
  availableModels: string[];
  availableGpus: string[];

  selectedModel: string | null;
  modelConfig: ModelConfigView | null;
  numLayersOverride: number | null;

  primaryGpuName: string | null;
  primaryGpu: GpuSpecView | null;
  secondaryGpuName: string | null;
  secondaryGpu: GpuSpecView | null;
  numGpus: number;

  parallel: ParallelConfig;
  ppMode: PpMode;
  layoutText: string;
  workload: Workload;

  ranks: number[];
  projection: ProjectionOutput | null;
  projectionError: string | null;
  projectionRunning: boolean;
}

export const DEFAULT_PARALLEL: ParallelConfig = {
  precision: "bf16",
  tensor_model_parallel_size: 1,
  sequence_parallel: false,
  pipeline_model_parallel_size: 1,
  virtual_pipeline_model_parallel_size: null,
  pipeline_model_parallel_layout: null,
  context_parallel_size: 1,
  expert_model_parallel_size: 1,
  data_parallel_size: 1,
  optimizer_kind: "distributed_optimizer",
  use_precision_aware_optimizer: false,
  optimizer_main_param_dtype: "fp32",
  optimizer_main_grad_dtype: "fp32",
  optimizer_exp_avg_dtype: "fp32",
  optimizer_exp_avg_sq_dtype: "fp32",
};

export const DEFAULT_WORKLOAD: Workload = {
  seq_length: 8192,
  micro_batch_size: 1,
  global_batch_size: 64,
  recompute_granularity: "none",
};

export const INITIAL_STATE: State = {
  bridgeStatus: "loading",
  availableModels: [],
  availableGpus: [],
  selectedModel: null,
  modelConfig: null,
  numLayersOverride: null,
  primaryGpuName: null,
  primaryGpu: null,
  secondaryGpuName: null,
  secondaryGpu: null,
  numGpus: 8,
  parallel: DEFAULT_PARALLEL,
  ppMode: "direct",
  layoutText: "",
  workload: DEFAULT_WORKLOAD,
  ranks: [0],
  projection: null,
  projectionError: null,
  projectionRunning: false,
};

export type Action =
  | { type: "BRIDGE_READY"; models: string[]; gpus: string[] }
  | { type: "BRIDGE_ERROR"; error: string }
  | { type: "SELECT_MODEL"; name: string; config: ModelConfigView }
  | { type: "OVERRIDE_NUM_LAYERS"; value: number | null }
  | { type: "SELECT_GPU_PRIMARY"; name: string; spec: GpuSpecView }
  | { type: "SELECT_GPU_SECONDARY"; name: string | null; spec: GpuSpecView | null }
  | { type: "SET_NUM_GPUS"; value: number }
  | { type: "SET_PARALLEL"; patch: Partial<ParallelConfig> }
  | { type: "SET_PP_MODE"; mode: PpMode }
  | { type: "SET_LAYOUT_TEXT"; value: string }
  | { type: "SET_WORKLOAD"; patch: Partial<Workload> }
  | { type: "SET_RANKS"; ranks: number[] }
  | { type: "PROJECTION_START" }
  | { type: "PROJECTION_SUCCESS"; output: ProjectionOutput }
  | { type: "PROJECTION_ERROR"; error: string };

function _deriveDP(world: number, parallel: ParallelConfig): number {
  const tp = Math.max(1, parallel.tensor_model_parallel_size);
  const pp = Math.max(1, parallel.pipeline_model_parallel_size);
  const cp = Math.max(1, parallel.context_parallel_size);
  const denom = tp * pp * cp;
  if (world <= 0 || denom <= 0 || world % denom !== 0) return 0;
  return Math.max(1, world / denom);
}

function _withDerivedDP(state: State, patch?: { numGpus?: number; parallel?: Partial<ParallelConfig> }): State {
  const numGpus = patch?.numGpus ?? state.numGpus;
  const merged: ParallelConfig = { ...state.parallel, ...(patch?.parallel ?? {}) };
  const dp = _deriveDP(numGpus, merged);
  // We keep ``data_parallel_size`` in parallel so the Python side gets a
  // self-consistent dict; when world/TPP/CP don't cleanly divide, it stays 0
  // and validation surfaces the error.
  return {
    ...state,
    numGpus,
    parallel: { ...merged, data_parallel_size: dp || 1 },
  };
}

export function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "BRIDGE_READY":
      return {
        ...state,
        bridgeStatus: "ready",
        availableModels: action.models,
        availableGpus: action.gpus,
      };
    case "BRIDGE_ERROR":
      return { ...state, bridgeStatus: "error", bridgeError: action.error };
    case "SELECT_MODEL":
      return {
        ...state,
        selectedModel: action.name,
        modelConfig: action.config,
        numLayersOverride: null,
      };
    case "OVERRIDE_NUM_LAYERS":
      return { ...state, numLayersOverride: action.value };
    case "SELECT_GPU_PRIMARY":
      return { ...state, primaryGpuName: action.name, primaryGpu: action.spec };
    case "SELECT_GPU_SECONDARY":
      return { ...state, secondaryGpuName: action.name, secondaryGpu: action.spec };
    case "SET_NUM_GPUS":
      return _withDerivedDP(state, { numGpus: Math.max(1, action.value) });
    case "SET_PARALLEL":
      return _withDerivedDP(state, { parallel: action.patch });
    case "SET_PP_MODE": {
      const next = _withDerivedDP(state, {
        parallel: {
          virtual_pipeline_model_parallel_size:
            action.mode === "direct" ? state.parallel.virtual_pipeline_model_parallel_size : null,
          pipeline_model_parallel_layout:
            action.mode === "layout" ? state.parallel.pipeline_model_parallel_layout : null,
        },
      });
      return { ...next, ppMode: action.mode };
    }
    case "SET_LAYOUT_TEXT":
      return { ...state, layoutText: action.value };
    case "SET_WORKLOAD":
      return { ...state, workload: { ...state.workload, ...action.patch } };
    case "SET_RANKS":
      return { ...state, ranks: action.ranks };
    case "PROJECTION_START":
      return { ...state, projectionRunning: true, projectionError: null };
    case "PROJECTION_SUCCESS":
      return { ...state, projectionRunning: false, projection: action.output };
    case "PROJECTION_ERROR":
      return { ...state, projectionRunning: false, projectionError: action.error };
    default:
      return state;
  }
}

export function isProxyModel(state: State): boolean {
  return state.numLayersOverride !== null && state.numLayersOverride !== state.modelConfig?.architecture.num_layers;
}

export function effectiveModelConfig(state: State): ModelConfigView | null {
  if (!state.modelConfig) return null;
  if (state.numLayersOverride === null) return state.modelConfig;
  return {
    ...state.modelConfig,
    architecture: { ...state.modelConfig.architecture, num_layers: state.numLayersOverride },
  };
}

export function effectiveModelName(state: State): string | null {
  if (!state.modelConfig) return null;
  return isProxyModel(state) ? `${state.modelConfig.name} (proxy)` : state.modelConfig.name;
}

export function parseRankList(text: string, max: number): { ranks: number[]; error: string | null } {
  const raw = text
    .split(/[\s,]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  const ranks: number[] = [];
  for (const item of raw) {
    const n = Number(item);
    if (!Number.isInteger(n) || n < 0) {
      return { ranks: [], error: `'${item}' is not a non-negative integer` };
    }
    ranks.push(n);
  }
  if (ranks.length > max) {
    return { ranks: [], error: `at most ${max} ranks are supported` };
  }
  if (new Set(ranks).size !== ranks.length) {
    return { ranks: [], error: "rank list must be unique" };
  }
  return { ranks, error: null };
}

export function parseLayoutList(text: string): number[] | null {
  const raw = text
    .split(/[\s,]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  if (raw.length === 0) return null;
  const out: number[] = [];
  for (const item of raw) {
    const n = Number(item);
    if (!Number.isInteger(n) || n <= 0) return null;
    out.push(n);
  }
  return out;
}

export const PRECISIONS: Precision[] = ["bf16", "fp8"];
export const OPTIMIZER_KINDS: { value: OptimizerKind; label: string; enabled: boolean }[] = [
  { value: "distributed_optimizer", label: "Distributed Optimizer", enabled: true },
  { value: "torch_fsdp2", label: "Torch FSDP2", enabled: true },
  { value: "megatron_fsdp", label: "Megatron FSDP", enabled: true },
];

export const OPTIMIZER_DTYPES: OptimizerDtype[] = ["fp32", "bf16", "fp16"];

export interface DerivedView {
  world_size: number;
  data_parallel_size: number;
  expert_data_parallel_size: number;
  gradient_accumulation_steps: number;
}

export function deriveView(state: State): DerivedView {
  const tp = state.parallel.tensor_model_parallel_size;
  const pp = state.parallel.pipeline_model_parallel_size;
  const cp = state.parallel.context_parallel_size;
  const ep = state.parallel.expert_model_parallel_size;
  const denom = tp * pp * cp;
  const dpRaw = state.numGpus > 0 && denom > 0 && state.numGpus % denom === 0 ? state.numGpus / denom : 0;
  const dp = dpRaw || 0;
  // EDP = cp * dp / ep (Megatron's expert RankGenerator with cp=1 and dp=edp on the same world).
  const edpNumer = cp * dp;
  const edp = ep > 0 && edpNumer % ep === 0 ? edpNumer / ep : 0;
  const perStep = dp * state.workload.micro_batch_size;
  const ga = perStep > 0 && state.workload.global_batch_size % perStep === 0 ? state.workload.global_batch_size / perStep : 0;
  return {
    world_size: state.numGpus,
    data_parallel_size: dp,
    expert_data_parallel_size: edp,
    gradient_accumulation_steps: ga,
  };
}

export function clientValidate(state: State): string[] {
  const errors: string[] = [];
  const tp = state.parallel.tensor_model_parallel_size;
  const pp = state.parallel.pipeline_model_parallel_size;
  const cp = state.parallel.context_parallel_size;
  const ep = state.parallel.expert_model_parallel_size;
  const denom = tp * pp * cp;
  if (state.numGpus < 1) errors.push("World size (number of GPUs) must be >= 1.");
  if (denom > 0 && state.numGpus % denom !== 0) {
    errors.push(
      `world_size=${state.numGpus} is not divisible by TP*PP*CP=${denom}; ` +
        `adjust TP, PP, CP, or the number of GPUs so DP comes out to a positive integer.`,
    );
  }
  const dp = deriveView(state).data_parallel_size;
  if (state.workload.global_batch_size > 0 && dp > 0) {
    const perStep = dp * state.workload.micro_batch_size;
    if (state.workload.global_batch_size % perStep !== 0) {
      errors.push(
        `global_batch_size=${state.workload.global_batch_size} must be divisible by ` +
          `data_parallel_size*micro_batch_size=${perStep}; adjust gbs, mbs, or DP (via world_size / TP / PP / CP).`,
      );
    }
  }
  if (state.modelConfig?.moe?.enabled && state.parallel.expert_model_parallel_size > 1) {
    if (state.modelConfig.moe.num_routed_experts % ep !== 0) {
      errors.push(
        `num_routed_experts=${state.modelConfig.moe.num_routed_experts} must be divisible by ` +
          `expert_model_parallel_size=${ep}.`,
      );
    }
    const dpForEp = deriveView(state).data_parallel_size;
    const edpNumer = cp * dpForEp;
    if (dpForEp > 0 && edpNumer % ep !== 0) {
      errors.push(
        `expert_model_parallel_size=${ep} must divide cp*dp=${edpNumer} so EDP = cp*dp/ep is an integer.`,
      );
    }
  }
  if (
    state.parallel.optimizer_kind === "torch_fsdp2" &&
    state.parallel.pipeline_model_parallel_size > 1
  ) {
    errors.push("torch_fsdp2 is incompatible with pipeline_model_parallel_size > 1.");
  }
  if (
    state.parallel.optimizer_kind === "megatron_fsdp" &&
    state.parallel.pipeline_model_parallel_size > 1
  ) {
    errors.push("megatron_fsdp is incompatible with pipeline_model_parallel_size > 1 in v1.");
  }
  if (
    state.parallel.sequence_parallel &&
    state.parallel.tensor_model_parallel_size === 1
  ) {
    errors.push("sequence_parallel requires tensor_model_parallel_size > 1.");
  }
  if (state.workload.recompute_granularity === "full") {
    if (!state.workload.recompute_method) {
      errors.push("recompute_granularity='full' requires recompute_method (uniform / block).");
    }
    if (!state.workload.recompute_num_layers || state.workload.recompute_num_layers < 1) {
      errors.push("recompute_granularity='full' requires recompute_num_layers >= 1.");
    } else if (
      state.modelConfig &&
      state.workload.recompute_num_layers > state.modelConfig.architecture.num_layers
    ) {
      errors.push(
        `recompute_num_layers=${state.workload.recompute_num_layers} must be <= num_layers=${state.modelConfig.architecture.num_layers}.`,
      );
    }
  }
  return errors;
}
