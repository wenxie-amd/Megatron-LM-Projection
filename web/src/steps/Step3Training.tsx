import { useEffect } from "react";

import { InfoIcon } from "../components/ExplanationPanel";
import { CheckboxField, NumberField, SelectField, TextField } from "../components/Field";
import { useProjection } from "../state/context";
import {
  OPTIMIZER_DTYPES,
  OPTIMIZER_KINDS,
  PRECISIONS,
  clientValidate,
  deriveView,
  parseLayoutList,
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

  let layoutError: string | null = null;
  if (ppMode === "layout") {
    if (!layoutParsed) {
      layoutError = "Enter positive integers separated by commas (e.g. 9, 8, 8, 7)";
    } else if (layoutParsed.length !== parallel.pipeline_model_parallel_size) {
      layoutError = `layout must have ${parallel.pipeline_model_parallel_size} entries (got ${layoutParsed.length})`;
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
            label="DP (data_parallel_size)"
            value={derived.data_parallel_size || "—"}
            hint="world_size / (TP × PP × CP)"
          />
          {isMoE && (
            <Derived
              label="EDP (expert_data_parallel_size)"
              value={derived.expert_data_parallel_size || "—"}
              hint="DP / EP"
            />
          )}
          <Derived
            label="GA (gradient_accumulation_steps)"
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
                <p>DP is derived from world_size and the other parallel sizes, not user-set.</p>
              </>
            }
          />
        </legend>
        <SelectField
          label="Precision"
          value={parallel.precision}
          options={PRECISIONS.map((p) => ({ value: p, label: p.toUpperCase() }))}
          onChange={(v: Precision) => dispatch({ type: "SET_PARALLEL", patch: { precision: v } })}
        />
        <div className="grid">
          <NumberField
            label="TP (tensor_model_parallel_size)"
            value={parallel.tensor_model_parallel_size}
            min={1}
            onChange={(v) => dispatch({ type: "SET_PARALLEL", patch: { tensor_model_parallel_size: Math.max(1, v) } })}
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
            onChange={(v) => dispatch({ type: "SET_PARALLEL", patch: { context_parallel_size: Math.max(1, v) } })}
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
          <NumberField
            label="PP (pipeline_model_parallel_size)"
            value={parallel.pipeline_model_parallel_size}
            min={1}
            onChange={(v) => dispatch({ type: "SET_PARALLEL", patch: { pipeline_model_parallel_size: Math.max(1, v) } })}
          />
        </div>

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
        {ppMode === "direct" && (
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
          />
        )}
        {ppMode === "layout" && (
          <TextField
            label="Layout (comma-separated layer counts per stage)"
            value={state.layoutText}
            onChange={(v) => dispatch({ type: "SET_LAYOUT_TEXT", value: v })}
            placeholder={
              numLayers !== null
                ? `e.g. ${[...Array(parallel.pipeline_model_parallel_size)].map((_, i) =>
                    Math.floor(numLayers / parallel.pipeline_model_parallel_size) + (i === 0 ? numLayers % parallel.pipeline_model_parallel_size : 0),
                  ).join(", ")}`
                : "e.g. 9, 8, 8, 7"
            }
            hint={layoutError ?? undefined}
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
            onChange={(v) =>
              dispatch({
                type: "SET_WORKLOAD",
                patch: { recompute_granularity: v as Workload["recompute_granularity"] },
              })
            }
          />
          {workload.recompute_granularity === "full" && (
            <>
              <SelectField
                label="recompute_method"
                value={workload.recompute_method ?? "uniform"}
                options={[
                  { value: "uniform", label: "uniform" },
                  { value: "block", label: "block" },
                ]}
                onChange={(v) =>
                  dispatch({
                    type: "SET_WORKLOAD",
                    patch: { recompute_method: v as "uniform" | "block" },
                  })
                }
              />
              <NumberField
                label="recompute_num_layers"
                value={workload.recompute_num_layers ?? 1}
                min={1}
                max={numLayers ?? undefined}
                onChange={(v) =>
                  dispatch({ type: "SET_WORKLOAD", patch: { recompute_num_layers: Math.max(1, v) } })
                }
              />
            </>
          )}
        </div>
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
