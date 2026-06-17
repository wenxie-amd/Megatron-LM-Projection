import { useEffect, useState } from "react";

import { InfoIcon } from "../components/ExplanationPanel";
import { CheckboxField, NumberField, SelectField, TextField } from "../components/Field";
import { getBridge } from "../pyodide/bridge";
import { useProjection } from "../state/context";
import {
  OPTIMIZER_DTYPES,
  OPTIMIZER_KINDS,
  PRECISIONS,
  clientValidate,
  deriveView,
  estimateLayerCosts,
  formatEdp,
  parseLayoutList,
  suggestLayout,
  suggestLayoutBalanced,
} from "../state/store";
import type { OptimizerDtype, Precision, Workload } from "../pyodide/types";

export function Step3Training() {
  const { state, dispatch } = useProjection();
  const { parallel, workload, ppMode } = state;
  const derived = deriveView(state);
  const errors = clientValidate(state);

  const layoutParsed = parseLayoutList(state.layoutText);
  const layoutSum = layoutParsed?.reduce((a, b) => a + b, 0) ?? null;
  const numLayers = state.modelConfig?.architecture.num_layers ?? null;
  const layoutVpp = parallel.virtual_pipeline_model_parallel_size ?? 1;
  const expectedLayoutLen = parallel.pipeline_model_parallel_size * Math.max(1, layoutVpp);

  let layoutError: string | null = null;
  if (ppMode === "layout") {
    if (!layoutParsed) {
      layoutError = "Enter positive integers separated by commas (e.g. 9, 8, 8, 7)";
    } else if (layoutParsed.length !== expectedLayoutLen) {
      layoutError = `layout must have pp*vpp=${expectedLayoutLen} entries (got ${layoutParsed.length})`;
    } else if (numLayers !== null && layoutSum !== numLayers) {
      layoutError = `layout sums to ${layoutSum} but num_layers=${numLayers}`;
    }
  }

  const effectiveLayout = ppMode === "layout" && layoutParsed && !layoutError ? layoutParsed : null;
  useEffect(() => {
    if (ppMode !== "layout") return;
    const current = JSON.stringify(parallel.pipeline_model_parallel_layout);
    const next = JSON.stringify(effectiveLayout);
    if (current !== next) {
      dispatch({ type: "SET_PARALLEL", patch: { pipeline_model_parallel_layout: effectiveLayout } });
    }
  }, [ppMode, effectiveLayout, parallel.pipeline_model_parallel_layout, dispatch]);

  // Auto-fill the layout textarea whenever the user is in layout mode and
  // (PP, VPP, num_layers, EP) changes. We re-fill even after user edits so
  // changing knobs invalidates a now-stale layout. For MoE models with
  // ``first_k_dense_replace > 0`` we use a balanced layout that absorbs the
  // cheap dense layers onto chunk 0.
  const firstKDense = state.modelConfig?.moe?.first_k_dense_replace ?? 0;
  useEffect(() => {
    if (ppMode !== "layout" || numLayers === null) return;
    const costs = estimateLayerCosts(state);
    const suggestion =
      state.modelConfig?.moe?.enabled && firstKDense > 0
        ? suggestLayoutBalanced(numLayers, firstKDense, expectedLayoutLen, costs)
        : suggestLayout(numLayers, expectedLayoutLen);
    const text = suggestion.join(", ");
    if (state.layoutText !== text) {
      dispatch({ type: "SET_LAYOUT_TEXT", value: text });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    ppMode,
    parallel.pipeline_model_parallel_size,
    layoutVpp,
    numLayers,
    parallel.expert_model_parallel_size,
    parallel.tensor_model_parallel_size,
    parallel.expert_tensor_parallel_size,
    parallel.moe_folding,
    firstKDense,
  ]);

  const isMoE = !!state.modelConfig?.moe.enabled;

  return (
    <section className="step">
      <h2>Step 3 · Training Parameters</h2>

      {errors.length > 0 && (
        <div className="validation-banner" role="alert">
          <strong>Fix the following before running the projection:</strong>
          <ul>
            {errors.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </div>
      )}

      <fieldset>
        <legend>Derived</legend>
        <div className="derived-grid">
          <Derived label="World size" value={derived.world_size} hint="from Step 2 (number of GPUs)" />
          <Derived
            label={parallel.moe_folding ? "ADP (attention DP)" : "DP"}
            value={derived.data_parallel_size || "—"}
            hint="world_size / (TP × PP × CP)"
          />
          {isMoE && (
            <Derived
              label="EDP (expert DP)"
              value={formatEdp(derived.expert_data_parallel_size)}
              hint={
                derived.expert_data_parallel_size > 0 && derived.expert_data_parallel_size < 1
                  ? "world / (ETP × EP × PP); < 1 means each rank holds 1/EDP expert slices, optimizer state is not further sharded"
                  : "world / (ETP × EP × PP)"
              }
            />
          )}
          {isMoE && parallel.moe_folding && (
            <Derived
              label="ETP (expert TP)"
              value={derived.expert_tensor_parallel_size}
              hint="from MoE column (defaults to TP)"
            />
          )}
          <Derived
            label="GA"
            value={derived.gradient_accumulation_steps || "—"}
            hint="gbs / (DP × mbs)"
          />
        </div>
      </fieldset>

      <fieldset>
        <legend>
          Distributed strategy{" "}
          <InfoIcon
            title="Distributed strategy"
            body={
              <>
                <p>Knobs from Megatron's <code>ModelParallelConfig</code>.</p>
                <p>
                  Rank ordering is Megatron's default <code>tp-cp-ep-dp-pp</code>, so rank 0 holds
                  (tp=0, cp=0, ep=0, dp=0, pp=0).
                </p>
                <p>DP and EDP are derived from world_size and the other parallel sizes.</p>
                <p>
                  <b>MoE folding</b> lets attention and MoE use independent strategies
                  (attention: TP/CP/ADP; MoE: ETP/EP/EDP). PP is shared. Megatron implements this
                  via a <code>ChainedOptimizer</code> with separate dense / expert sub-optimizers.
                </p>
              </>
            }
          />
        </legend>
        <div className="grid">
          <SelectField
            label="Precision"
            value={parallel.precision}
            options={PRECISIONS.map((p) => ({ value: p, label: p.toUpperCase() }))}
            onChange={(v: Precision) => dispatch({ type: "SET_PARALLEL", patch: { precision: v } })}
          />
          <NumberField
            label="PP (pipeline_model_parallel_size, shared)"
            value={parallel.pipeline_model_parallel_size}
            min={1}
            onChange={(v) =>
              dispatch({ type: "SET_PARALLEL", patch: { pipeline_model_parallel_size: Math.max(1, v) } })
            }
          />
          {isMoE && (
            <CheckboxField
              label="MoE folding"
              checked={!!parallel.moe_folding}
              onChange={(v) => dispatch({ type: "SET_PARALLEL", patch: { moe_folding: v } })}
              hint="Independent strategies for attention and MoE (Megatron ChainedOptimizer)"
            />
          )}
        </div>

        {isMoE && parallel.moe_folding ? (
          <div className="folded-strategy">
            <div className="folded-side">
              <h4>Attention side</h4>
              <div className="grid">
                <NumberField
                  label="TP (tensor_model_parallel_size)"
                  value={parallel.tensor_model_parallel_size}
                  min={1}
                  onChange={(v) =>
                    dispatch({ type: "SET_PARALLEL", patch: { tensor_model_parallel_size: Math.max(1, v) } })
                  }
                />
                <CheckboxField
                  label="Sequence parallel"
                  checked={parallel.sequence_parallel}
                  onChange={(v) => dispatch({ type: "SET_PARALLEL", patch: { sequence_parallel: v } })}
                  disabled={parallel.tensor_model_parallel_size === 1}
                />
                <NumberField
                  label="CP (context_parallel_size)"
                  value={parallel.context_parallel_size}
                  min={1}
                  onChange={(v) =>
                    dispatch({ type: "SET_PARALLEL", patch: { context_parallel_size: Math.max(1, v) } })
                  }
                />
                <div className="derived-cell inline">
                  <div className="derived-label">ADP (attention DP)</div>
                  <div className="derived-value">{derived.data_parallel_size || "—"}</div>
                  <div className="derived-hint">world / (TP × CP × PP)</div>
                </div>
              </div>
            </div>
            <div className="folded-side">
              <h4>MoE side</h4>
              <div className="grid">
                <NumberField
                  label="ETP (expert_tensor_parallel_size)"
                  value={parallel.expert_tensor_parallel_size ?? parallel.tensor_model_parallel_size}
                  min={1}
                  onChange={(v) =>
                    dispatch({
                      type: "SET_PARALLEL",
                      patch: { expert_tensor_parallel_size: Math.max(1, v) },
                    })
                  }
                  hint="Independent TP for routed experts"
                />
                <NumberField
                  label="EP (expert_model_parallel_size)"
                  value={parallel.expert_model_parallel_size}
                  min={1}
                  onChange={(v) =>
                    dispatch({ type: "SET_PARALLEL", patch: { expert_model_parallel_size: Math.max(1, v) } })
                  }
                />
                <div className="derived-cell inline">
                  <div className="derived-label">EDP (expert DP)</div>
                  <div className="derived-value">{formatEdp(derived.expert_data_parallel_size)}</div>
                  <div className="derived-hint">
                    world / (ETP × EP × PP){" "}
                    {derived.expert_data_parallel_size > 0 && derived.expert_data_parallel_size < 1
                      ? `· 1/EDP=${Math.round(1 / derived.expert_data_parallel_size)} expert slices per rank, no DP shard for routed`
                      : ""}
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="grid">
            <NumberField
              label="TP (tensor_model_parallel_size)"
              value={parallel.tensor_model_parallel_size}
              min={1}
              onChange={(v) =>
                dispatch({ type: "SET_PARALLEL", patch: { tensor_model_parallel_size: Math.max(1, v) } })
              }
            />
            <CheckboxField
              label="Sequence parallel"
              checked={parallel.sequence_parallel}
              onChange={(v) => dispatch({ type: "SET_PARALLEL", patch: { sequence_parallel: v } })}
              disabled={parallel.tensor_model_parallel_size === 1}
            />
            <NumberField
              label="CP (context_parallel_size)"
              value={parallel.context_parallel_size}
              min={1}
              onChange={(v) =>
                dispatch({ type: "SET_PARALLEL", patch: { context_parallel_size: Math.max(1, v) } })
              }
            />
            {isMoE && (
              <NumberField
                label="EP (expert_model_parallel_size)"
                value={parallel.expert_model_parallel_size}
                min={1}
                onChange={(v) =>
                  dispatch({ type: "SET_PARALLEL", patch: { expert_model_parallel_size: Math.max(1, v) } })
                }
                hint="Routed experts split across this group"
              />
            )}
          </div>
        )}

        <div className="pp-mode">
          <span className="field-label">Pipeline mode</span>
          <label>
            <input
              type="radio"
              name="ppMode"
              checked={ppMode === "direct"}
              onChange={() => dispatch({ type: "SET_PP_MODE", mode: "direct" })}
            />
            Direct (PP + VPP)
          </label>
          <label>
            <input
              type="radio"
              name="ppMode"
              checked={ppMode === "layout"}
              onChange={() => dispatch({ type: "SET_PP_MODE", mode: "layout" })}
            />
            Layout (per-stage layer counts)
          </label>
        </div>
        <NumberField
          label="VPP (virtual_pipeline_model_parallel_size)"
          value={parallel.virtual_pipeline_model_parallel_size ?? 1}
          min={1}
          onChange={(v) =>
            dispatch({
              type: "SET_PARALLEL",
              patch: { virtual_pipeline_model_parallel_size: v === 1 ? null : Math.max(1, v) },
            })
          }
          hint={
            ppMode === "layout"
              ? `Layout will have pp*vpp = ${expectedLayoutLen} chunks`
              : "1 means no interleaved schedule"
          }
        />
        {ppMode === "layout" && (
          <TextField
            label={`Layout (${expectedLayoutLen} layer counts per chunk, comma-separated)`}
            value={state.layoutText}
            onChange={(v) => dispatch({ type: "SET_LAYOUT_TEXT", value: v })}
            placeholder={
              numLayers !== null ? suggestLayout(numLayers, expectedLayoutLen).join(", ") : "e.g. 9, 8, 8, 7"
            }
            hint={
              layoutError ??
              (state.modelConfig?.moe?.enabled && firstKDense > 0
                ? `Chunk 0 absorbs ${firstKDense} dense + extra MoE layers to balance against the heavier MoE chunks; you can edit freely.`
                : "Front + back chunks get the +1 surplus. You can edit freely.")
            }
          />
        )}

        <SelectField
          label="Optimizer / sharding"
          value={parallel.optimizer_kind}
          options={OPTIMIZER_KINDS.map((o) => ({ value: o.value, label: o.label, disabled: !o.enabled }))}
          onChange={(v) => dispatch({ type: "SET_PARALLEL", patch: { optimizer_kind: v } })}
          hint={
            <InfoIcon
              title="Optimizer / sharding"
              body={
                <>
                  <p><b>distributed_optimizer</b>: Adam state sharded across DP; params replicated across DP.</p>
                  <p><b>torch_fsdp2</b>: params + grads + Adam state all sharded across DP. Incompatible with PP &gt; 1.</p>
                  <p><b>megatron_fsdp</b>: AMD-flavoured FSDP. Incompatible with PP &gt; 1 in v1.</p>
                </>
              }
            />
          }
        />

        {parallel.optimizer_kind === "distributed_optimizer" && (
          <div className="precision-aware">
            <CheckboxField
              label="use_precision_aware_optimizer"
              checked={!!parallel.use_precision_aware_optimizer}
              onChange={(v) => dispatch({ type: "SET_PARALLEL", patch: { use_precision_aware_optimizer: v } })}
              hint="Independently set master / grad / Adam dtypes (affects optimizer state memory)."
            />
            {parallel.use_precision_aware_optimizer && (
              <div className="grid">
                <DtypeField
                  label="main_grad_dtype"
                  value={parallel.optimizer_main_grad_dtype ?? "fp32"}
                  onChange={(v) => dispatch({ type: "SET_PARALLEL", patch: { optimizer_main_grad_dtype: v } })}
                />
                <DtypeField
                  label="main_param_dtype"
                  value={parallel.optimizer_main_param_dtype ?? "fp32"}
                  onChange={(v) => dispatch({ type: "SET_PARALLEL", patch: { optimizer_main_param_dtype: v } })}
                />
                <DtypeField
                  label="exp_avg_dtype (momentum)"
                  value={parallel.optimizer_exp_avg_dtype ?? "fp32"}
                  onChange={(v) => dispatch({ type: "SET_PARALLEL", patch: { optimizer_exp_avg_dtype: v } })}
                />
                <DtypeField
                  label="exp_avg_sq_dtype (variance)"
                  value={parallel.optimizer_exp_avg_sq_dtype ?? "fp32"}
                  onChange={(v) => dispatch({ type: "SET_PARALLEL", patch: { optimizer_exp_avg_sq_dtype: v } })}
                />
              </div>
            )}
          </div>
        )}
      </fieldset>

      <fieldset>
        <legend>Workload</legend>
        <div className="grid">
          <NumberField
            label="seq_length"
            value={workload.seq_length}
            min={1}
            onChange={(v) => dispatch({ type: "SET_WORKLOAD", patch: { seq_length: Math.max(1, v) } })}
          />
          <NumberField
            label="micro_batch_size"
            value={workload.micro_batch_size}
            min={1}
            onChange={(v) => dispatch({ type: "SET_WORKLOAD", patch: { micro_batch_size: Math.max(1, v) } })}
          />
          <NumberField
            label="global_batch_size"
            value={workload.global_batch_size}
            min={1}
            onChange={(v) => dispatch({ type: "SET_WORKLOAD", patch: { global_batch_size: Math.max(1, v) } })}
          />
          <SelectField
            label="recompute_granularity"
            value={workload.recompute_granularity}
            options={[
              { value: "none", label: "none" },
              { value: "selective", label: "selective" },
              { value: "full", label: "full" },
            ]}
            onChange={(v) => {
              const granularity = v as Workload["recompute_granularity"];
              const patch: Partial<Workload> = { recompute_granularity: granularity };
              if (granularity === "full" && !workload.recompute_method) {
                patch.recompute_method = "block";
                if (!workload.recompute_num_layers) patch.recompute_num_layers = 1;
              }
              dispatch({ type: "SET_WORKLOAD", patch });
            }}
          />
          {workload.recompute_granularity === "full" && (
            <>
              <SelectField
                label="recompute_method"
                value={workload.recompute_method ?? "block"}
                options={[
                  { value: "block", label: "block" },
                  { value: "uniform", label: "uniform" },
                ]}
                onChange={(v) =>
                  dispatch({
                    type: "SET_WORKLOAD",
                    patch: { recompute_method: v as "uniform" | "block" },
                  })
                }
              />
              <NumberField
                label="recompute_num_layers (per model chunk)"
                value={workload.recompute_num_layers ?? 1}
                min={1}
                onChange={(v) =>
                  dispatch({ type: "SET_WORKLOAD", patch: { recompute_num_layers: Math.max(1, v) } })
                }
                hint={
                  workload.recompute_method === "uniform"
                    ? "uniform method: every layer is recomputed (recompute_num_layers is the unit size)."
                    : "block method: per chunk. Total per rank = recompute_num_layers × num_chunks_per_rank (= vpp; each PP rank owns vpp model chunks)."
                }
              />
            </>
          )}
        </div>
        {workload.recompute_granularity === "full" && <PerRankRecomputeView />}
      </fieldset>

      <fieldset>
        <legend>Hyperparameters</legend>
        <p className="field-hint">Defaults are fine — no memory or throughput impact.</p>
      </fieldset>
    </section>
  );
}

function Derived({ label, value, hint }: { label: string; value: number | string; hint: string }) {
  return (
    <div className="derived-cell">
      <div className="derived-label">{label}</div>
      <div className="derived-value">{value}</div>
      <div className="derived-hint">{hint}</div>
    </div>
  );
}

interface DtypeFieldProps {
  label: string;
  value: OptimizerDtype;
  onChange: (v: OptimizerDtype) => void;
}
function DtypeField({ label, value, onChange }: DtypeFieldProps) {
  return (
    <SelectField
      label={label}
      value={value}
      options={OPTIMIZER_DTYPES.map((d) => ({ value: d, label: d }))}
      onChange={(v) => onChange(v as OptimizerDtype)}
    />
  );
}

function PerRankRecomputeView() {
  const { state } = useProjection();
  const [ppRank, setPpRank] = useState(0);
  const [info, setInfo] = useState<{
    total_num_layers: number;
    num_chunks_per_rank: number;
    total_recompute_num_layers: number;
  } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const pp = state.parallel.pipeline_model_parallel_size;
  const ppRankClamped = Math.min(Math.max(0, ppRank), pp - 1);

  useEffect(() => {
    let cancelled = false;
    if (!state.modelConfig) return;
    (async () => {
      try {
        const bridge = await getBridge();
        const result = bridge.computePerRankLayers({
          model: state.selectedModel ?? (state.modelConfig as unknown as Record<string, unknown>),
          parallel: state.parallel as unknown as Record<string, unknown>,
          workload: state.workload as unknown as Record<string, unknown>,
          pp_rank: ppRankClamped,
        });
        if (!cancelled) {
          setInfo(result);
          setErr(null);
        }
      } catch (e) {
        if (!cancelled) {
          setErr(e instanceof Error ? e.message : String(e));
          setInfo(null);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [state.modelConfig, state.selectedModel, state.parallel, state.workload, ppRankClamped]);

  return (
    <div className="per-rank-recompute">
      <label className="field per-rank-pp-select">
        <span className="field-label">Inspect PP rank</span>
        <select value={ppRankClamped} onChange={(e) => setPpRank(Number(e.target.value))}>
          {Array.from({ length: Math.max(1, pp) }, (_, i) => (
            <option key={i} value={i}>
              rank {i}
            </option>
          ))}
        </select>
      </label>
      {err && <p className="error">{err}</p>}
      {info && (
        <div className="derived-grid compact">
          <div className="derived-cell">
            <div className="derived-label">total_num_layers (PP rank {ppRankClamped})</div>
            <div className="derived-value">{info.total_num_layers}</div>
            <div className="derived-hint">layers physically owned by this PP stage</div>
          </div>
          <div className="derived-cell">
            <div className="derived-label">num_chunks_per_rank</div>
            <div className="derived-value">{info.num_chunks_per_rank}</div>
            <div className="derived-hint">
              model chunks (= vpp) this PP rank owns; recompute_num_layers applies per chunk
            </div>
          </div>
          <div className="derived-cell">
            <div className="derived-label">total_recompute_num_layers (PP rank {ppRankClamped})</div>
            <div className="derived-value">{info.total_recompute_num_layers}</div>
            <div className="derived-hint">
              {state.workload.recompute_method === "uniform"
                ? "uniform: all layers on this rank"
                : "block: recompute_num_layers × num_chunks_per_rank, capped"}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
