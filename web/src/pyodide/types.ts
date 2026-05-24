/**
 * Typed contract with `projection.api`. Mirrors the Python side; if either
 * changes, both must change. Tests verify they stay in sync.
 */

export type Precision = "bf16" | "fp8";

export type OptimizerKind =
  | "distributed_optimizer"
  | "torch_fsdp2"
  | "megatron_fsdp";

export type OptimizerDtype = "fp32" | "bf16" | "fp16";

export interface ParallelConfig {
  precision: Precision;
  tensor_model_parallel_size: number;
  sequence_parallel: boolean;
  pipeline_model_parallel_size: number;
  virtual_pipeline_model_parallel_size?: number | null;
  pipeline_model_parallel_layout?: number[] | null;
  context_parallel_size: number;
  expert_model_parallel_size: number;
  data_parallel_size: number;
  optimizer_kind: OptimizerKind;
  use_precision_aware_optimizer?: boolean;
  optimizer_main_param_dtype?: OptimizerDtype;
  optimizer_main_grad_dtype?: OptimizerDtype;
  optimizer_exp_avg_dtype?: OptimizerDtype;
  optimizer_exp_avg_sq_dtype?: OptimizerDtype;
  moe_folding?: boolean;
  expert_tensor_parallel_size?: number | null;
}

export interface DerivedConfig {
  world_size: number;
  data_parallel_size: number;
  expert_data_parallel_size: number;
  expert_tensor_parallel_size: number;
  gradient_accumulation_steps: number;
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

export interface Workload {
  seq_length: number;
  micro_batch_size: number;
  global_batch_size: number;
  recompute_granularity: "none" | "selective" | "full";
  recompute_method?: "uniform" | "block" | null;
  recompute_num_layers?: number | null;
  sequence_parallel?: boolean;
}

export interface ProjectionInput {
  model: string | Record<string, unknown>;
  parallel: Partial<ParallelConfig>;
  workload: Workload;
  ranks: number[];
  hyperparameters?: Record<string, unknown>;
}

export interface ParamBreakdownEntry {
  name: string;
  count: number;
}

export interface RankCoord {
  tp: number;
  cp: number;
  dp: number;
  pp: number;
  ep: number;
  expert_dp: number;
}

export interface ModelPartition {
  num_layers_on_rank: number;
  has_embedding: boolean;
  has_final_norm: boolean;
  has_output_projection: boolean;
}

export interface OptimizerMemoryBytes {
  grad_buffer_bytes: number;
  main_param_bytes: number;
  state_bytes: number;
  total_bytes: number;
}

export interface MemoryBytes {
  param_bytes: number;
  activation_bytes: number;
  optimizer: OptimizerMemoryBytes;
  total_bytes: number;
  precision: Precision;
}

export interface RankReport {
  global_rank: number;
  rank_coord: RankCoord;
  partition: ModelPartition;
  param_count: number;
  param_breakdown: ParamBreakdownEntry[];
  memory: MemoryBytes;
}

export interface ProjectionOutput {
  model_config: Record<string, unknown>;
  parallel: ParallelConfig;
  workload: Workload;
  derived: DerivedConfig;
  rank_reports: RankReport[];
}
