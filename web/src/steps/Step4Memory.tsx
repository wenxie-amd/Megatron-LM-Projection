import { Fragment, useEffect, useMemo, useState } from "react";

import { RankBars } from "../components/RankBars";
import { useProjection } from "../state/context";
import { clientValidate, deriveView, formatEdp, parseRankList } from "../state/store";
import type { RankReport } from "../pyodide/types";

const MAX_RANKS = 8;

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let value = bytes / 1024;
  let unitIdx = 0;
  while (value >= 1024 && unitIdx < units.length - 1) {
    value /= 1024;
    unitIdx++;
  }
  return `${value.toFixed(2)} ${units[unitIdx]}`;
}

export function Step4Memory() {
  const { state, dispatch, runProjection } = useProjection();
  const [rankText, setRankText] = useState(() => state.ranks.join(", "));
  const [rankError, setRankError] = useState<string | null>(null);

  const derived = deriveView(state);
  const upstreamErrors = clientValidate(state);

  const handleRankChange = (text: string) => {
    setRankText(text);
    const { ranks, error } = parseRankList(text, MAX_RANKS);
    if (error) {
      setRankError(error);
      return;
    }
    if (derived.world_size > 0) {
      const oor = ranks.filter((r) => r >= derived.world_size);
      if (oor.length > 0) {
        setRankError(`rank(s) ${oor.join(", ")} are >= world_size=${derived.world_size}`);
        return;
      }
    }
    setRankError(null);
    if (ranks.length > 0) {
      dispatch({ type: "SET_RANKS", ranks });
    }
  };

  const canRun =
    state.modelConfig !== null &&
    state.bridgeStatus === "ready" &&
    rankError === null &&
    upstreamErrors.length === 0 &&
    derived.data_parallel_size > 0;

  // Auto-run projection on Step 4 mount and whenever inputs change while we're
  // on this step. Skip if running, if config is invalid, or if we just kicked
  // off the same inputs (signature comparison).
  const signature = useMemo(
    () =>
      JSON.stringify({
        m: state.selectedModel,
        n: state.numLayersOverride,
        p: state.parallel,
        w: state.workload,
        r: state.ranks,
      }),
    [state.selectedModel, state.numLayersOverride, state.parallel, state.workload, state.ranks],
  );
  useEffect(() => {
    if (canRun && !state.projectionRunning) {
      runProjection();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signature, canRun]);

  return (
    <section className="step">
      <h2>Step 4 · Memory Analysis</h2>

      {!state.modelConfig && <p className="error">Please pick a model in Step 1 first.</p>}

      {upstreamErrors.length > 0 && (
        <div className="validation-banner" role="alert">
          <strong>Fix these in Step 3 before running the projection:</strong>
          <ul>
            {upstreamErrors.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </div>
      )}

      <DerivedStrip />

      <label className="field">
        <span className="field-label">
          Ranks to compare (max {MAX_RANKS}; comma- or space-separated, e.g. <code>0, 4, 10, 63</code>)
        </span>
        <input
          type="text"
          value={rankText}
          placeholder="0, 1, 2, 3"
          onChange={(e) => handleRankChange(e.target.value)}
        />
        {rankError && <span className="field-hint error-text">{rankError}</span>}
      </label>

      <div className="actions">
        <button type="button" onClick={runProjection} disabled={!canRun || state.projectionRunning}>
          {state.projectionRunning ? "Computing…" : "Re-run projection"}
        </button>
      </div>

      {state.projectionError && <pre className="error">{state.projectionError}</pre>}

      {state.projection && state.projection.rank_reports.length > 0 && (
        <ProjectionResults reports={state.projection.rank_reports} />
      )}
    </section>
  );
}

function DerivedStrip() {
  const { state } = useProjection();
  const derived = deriveView(state);
  const isMoE = !!state.modelConfig?.moe?.enabled;
  const folding = !!state.parallel.moe_folding;
  const adpLabel = isMoE ? "ADP" : "DP";
  const etpLabel = folding ? "ETP (folded)" : "ETP (=TP)";
  const entries: { label: string; value: string | number }[] = [
    { label: "World size", value: derived.world_size },
    { label: "TP", value: state.parallel.tensor_model_parallel_size },
    { label: "PP", value: state.parallel.pipeline_model_parallel_size },
    { label: "CP", value: state.parallel.context_parallel_size },
    { label: adpLabel, value: derived.data_parallel_size || "—" },
  ];
  if (isMoE) {
    entries.push({ label: etpLabel, value: derived.expert_tensor_parallel_size });
    entries.push({ label: "EP", value: state.parallel.expert_model_parallel_size });
    entries.push({ label: "EDP", value: formatEdp(derived.expert_data_parallel_size) });
  }
  entries.push({ label: "MBS", value: state.workload.micro_batch_size });
  entries.push({ label: "GBS", value: state.workload.global_batch_size });
  entries.push({ label: "GA", value: derived.gradient_accumulation_steps || "—" });

  return (
    <div className="derived-strip" style={{ gridTemplateColumns: `repeat(${entries.length}, minmax(64px, 1fr))` }}>
      {entries.map((e) => (
        <div key={`l-${e.label}`} className="derived-strip-label">
          {e.label}
        </div>
      ))}
      {entries.map((e) => (
        <div key={`v-${e.label}`} className="derived-strip-value">
          {e.value}
        </div>
      ))}
    </div>
  );
}

function ProjectionResults({ reports }: { reports: RankReport[] }) {
  const { state } = useProjection();
  // GPU "GB" in the spec is conventionally GiB (nvidia-smi reports HBM in GiB).
  const memoryRooflineGiB = state.primaryGpu?.memory_gb;
  const usePrecAware =
    state.parallel.optimizer_kind === "distributed_optimizer" && !!state.parallel.use_precision_aware_optimizer;
  const mainParamDtype = (usePrecAware && state.parallel.optimizer_main_param_dtype) || "fp32";
  const expAvgDtype = (usePrecAware && state.parallel.optimizer_exp_avg_dtype) || "fp32";
  const expAvgSqDtype = (usePrecAware && state.parallel.optimizer_exp_avg_sq_dtype) || "fp32";
  const mainGradDtype = (usePrecAware && state.parallel.optimizer_main_grad_dtype) || "fp32";
  const paramDtype = state.parallel.precision;
  const stateLabel = expAvgDtype === expAvgSqDtype ? expAvgDtype : `${expAvgDtype}/${expAvgSqDtype}`;

  return (
    <>
      <h3>Per-rank memory</h3>
      <RankBars
        reports={reports}
        roofLineGiB={memoryRooflineGiB}
        roofLineLabel={state.primaryGpu?.name}
        mainParamDtype={mainParamDtype}
        expAvgDtype={expAvgDtype}
        expAvgSqDtype={expAvgSqDtype}
        mainGradDtype={mainGradDtype}
      />
      <BreakdownTable
        reports={reports}
        paramDtype={paramDtype}
        mainGradDtype={mainGradDtype}
        mainParamDtype={mainParamDtype}
        stateDtypeLabel={stateLabel}
      />
      <CalculationDetails reports={reports} />
    </>
  );
}

interface BreakdownRow {
  label: string;
  values: number[];
}

interface BreakdownTableProps {
  reports: RankReport[];
  paramDtype: string;
  mainGradDtype: string;
  mainParamDtype: string;
  stateDtypeLabel: string;
}

function BreakdownTable({
  reports,
  paramDtype,
  mainGradDtype,
  mainParamDtype,
  stateDtypeLabel,
}: BreakdownTableProps) {
  const rows: BreakdownRow[] = [
    { label: `params (${paramDtype})`, values: reports.map((r) => r.memory.param_bytes) },
    { label: "activations", values: reports.map((r) => r.memory.activation_bytes) },
    { label: `main grad buffer (${mainGradDtype})`, values: reports.map((r) => r.memory.optimizer.grad_buffer_bytes) },
    { label: `optimizer main param (${mainParamDtype})`, values: reports.map((r) => r.memory.optimizer.main_param_bytes) },
    { label: `optimizer state m+v (${stateDtypeLabel})`, values: reports.map((r) => r.memory.optimizer.state_bytes) },
  ];
  const totals = reports.map((r) => r.memory.total_bytes || 1);
  return (
    <div>
      <h3>Breakdown</h3>
      <table className="combined-breakdown">
        <thead>
          <tr>
            <th rowSpan={2}>component</th>
            {reports.map((r) => {
              const rc = r.rank_coord;
              return (
                <th key={r.global_rank} colSpan={2}>
                  <div className="rank-head-name">rank {r.global_rank}</div>
                  <div className="rank-head-coord">
                    tp={rc.tp} cp={rc.cp} dp={rc.dp} pp={rc.pp}
                    {rc.ep > 0 || rc.expert_dp > 0 ? ` ep=${rc.ep} edp=${rc.expert_dp}` : ""}
                  </div>
                  <div className="rank-head-coord">
                    layers={r.partition.num_layers_on_rank}
                    {r.partition.has_embedding ? " · emb" : ""}
                    {r.partition.has_final_norm ? " · norm" : ""}
                    {r.partition.has_output_projection ? " · out" : ""}
                  </div>
                </th>
              );
            })}
          </tr>
          <tr className="sub-head">
            {reports.map((r) => (
              <Fragment key={r.global_rank}>
                <th>bytes</th>
                <th>%</th>
              </Fragment>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label}>
              <th>{row.label}</th>
              {row.values.map((v, i) => (
                <Fragment key={i}>
                  <td className="bytes-col">{formatBytes(v)}</td>
                  <td className="pct-col">{((v / totals[i]) * 100).toFixed(1)}%</td>
                </Fragment>
              ))}
            </tr>
          ))}
          <tr className="total">
            <th>total</th>
            {reports.map((r) => (
              <Fragment key={r.global_rank}>
                <td className="bytes-col">{formatBytes(r.memory.total_bytes)}</td>
                <td className="pct-col">100.0%</td>
              </Fragment>
            ))}
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function CalculationDetails({ reports }: { reports: RankReport[] }) {
  const { state } = useProjection();
  const [selectedRank, setSelectedRank] = useState(0);
  const r = reports[Math.min(selectedRank, reports.length - 1)];
  if (!r || !state.modelConfig) return null;

  const m = state.modelConfig;
  const p = state.parallel;
  const w = state.workload;
  const derived = deriveView(state);

  const precisionBytes = p.precision === "fp8" ? 1 : 2;
  const dp = derived.data_parallel_size || 1;
  const tp = p.tensor_model_parallel_size;
  const pp = p.pipeline_model_parallel_size;
  const cp = p.context_parallel_size;
  const vpp = p.virtual_pipeline_model_parallel_size ?? 1;
  const h = m.architecture.hidden_size;
  const s = w.seq_length;
  const mbs = w.micro_batch_size;
  const numLayers = m.architecture.num_layers;

  const totalParams = reports.reduce((a, b) => Math.max(a, b.param_count), 0);
  const paramsOnRank = totalParams / tp;
  const isFsdp = p.optimizer_kind === "torch_fsdp2" || p.optimizer_kind === "megatron_fsdp";
  const paramShardDivisor = isFsdp ? `TP × DP = ${tp} × ${dp}` : `TP = ${tp}`;

  const interleavedPenalty = vpp > 1 ? 1.0 + (pp - 1) / (pp * vpp) : 1.0;
  let inFlight: number;
  if (pp === 1) inFlight = 1;
  else if (vpp > 1) inFlight = Math.ceil(interleavedPenalty * pp);
  else inFlight = Math.max(1, Math.min(derived.gradient_accumulation_steps || pp, pp - r.rank_coord.pp));

  const sp = p.sequence_parallel;
  const selective = w.recompute_granularity === "selective";
  const ffn = m.architecture.ffn_hidden_size;
  const cpDiv = Math.max(1, cp);
  let perLayer: number;
  let perLayerFormula: string;
  if (sp && selective) {
    perLayer = Math.floor((18 * s * mbs * h + 4 * s * mbs * ffn) / (tp * cpDiv));
    perLayerFormula = "(SP + selective)  sbh·18 + 4sb·ffn  ÷ TP";
  } else if (sp) {
    perLayer = Math.floor((34 * s * mbs * h) / (tp * cpDiv));
    perLayerFormula = "(SP, no recompute)  sbh · 34 / TP  (Korthikanti)";
  } else if (selective) {
    perLayer = Math.floor(((10 * tp + 13) * s * mbs * h) / (tp * cpDiv));
    perLayerFormula = "(no SP, selective)  sbh × (10 + 13/TP)  (drops 11sbh/TP attention block)";
  } else {
    perLayer = Math.floor(((10 * tp + 24) * s * mbs * h) / (tp * cpDiv));
    perLayerFormula = "(no SP, no recompute)  sbh × (10 + 24/TP)";
  }

  const layersOnRank = r.partition.num_layers_on_rank;
  const numChunksPerRank = p.pipeline_model_parallel_layout
    ? p.pipeline_model_parallel_layout.length
    : pp * vpp;
  let recomputedOnRank = 0;
  if (w.recompute_granularity === "full") {
    if (w.recompute_method === "uniform") {
      recomputedOnRank = layersOnRank;
    } else {
      recomputedOnRank = Math.min(layersOnRank, (w.recompute_num_layers || 0) * numChunksPerRank);
    }
  }
  const fullPerLayer = Math.floor((2 * s * mbs * h) / Math.max(1, cp));

  const optimizerKindLabel = {
    distributed_optimizer: "Distributed optimizer (Adam shards across DP)",
    torch_fsdp2: "Torch FSDP2 (params + grads + Adam all shard across DP)",
    megatron_fsdp: "Megatron FSDP (similar to FSDP2)",
  }[p.optimizer_kind];

  const gradBufferDtype = p.use_precision_aware_optimizer ? p.optimizer_main_grad_dtype : "fp32";
  const mainParamDtype = p.use_precision_aware_optimizer ? p.optimizer_main_param_dtype : "fp32";
  const expAvgDtype = p.use_precision_aware_optimizer ? p.optimizer_exp_avg_dtype : "fp32";
  const expAvgSqDtype = p.use_precision_aware_optimizer ? p.optimizer_exp_avg_sq_dtype : "fp32";
  const stateDtypeLabel = expAvgDtype === expAvgSqDtype ? expAvgDtype : `${expAvgDtype} / ${expAvgSqDtype}`;

  return (
    <details className="calc-details" open>
      <summary>Calculation walk-through</summary>
      <label className="field calc-rank-pick">
        <span className="field-label">Walk through rank</span>
        <select value={selectedRank} onChange={(e) => setSelectedRank(Number(e.target.value))}>
          {reports.map((rr, i) => (
            <option key={rr.global_rank} value={i}>
              rank {rr.global_rank}
            </option>
          ))}
        </select>
      </label>

      <div className="calc-section">
        <h4>1. Parameters ({p.precision})</h4>
        <ol>
          <li>
            Full-model param count (from Step 1): <code>{totalParams.toLocaleString()}</code>
          </li>
          <li>
            Live params on this rank (after PP + {paramShardDivisor}): <code>{paramsOnRank.toLocaleString()}</code>
          </li>
          <li>
            Stored in <b>{p.precision}</b> ({precisionBytes} bytes / element) → <code>{formatBytes(r.memory.param_bytes)}</code>
          </li>
        </ol>
      </div>

      <div className="calc-section">
        <h4>2. Main grad buffer ({gradBufferDtype})</h4>
        <ol>
          <li>
            Megatron's DDP allocates a <b>single contiguous</b> grad buffer (per
            <code>third_party/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py</code>).
            Each <code>param.main_grad</code> is a view into it.
          </li>
          <li>
            Dtype:{" "}
            {p.use_precision_aware_optimizer
              ? `${p.optimizer_main_grad_dtype} (precision-aware optimizer)`
              : `fp32 (default, --accumulate-allreduce-grads-in-fp32)`}
          </li>
          <li>
            Size = paramsOnRank × dtype_bytes = <code>{formatBytes(r.memory.optimizer.grad_buffer_bytes)}</code>
          </li>
          <li>
            <b>Not</b> sharded by DP (buffer is full on every rank during backward, before reduce-scatter).
          </li>
        </ol>
      </div>

      <div className="calc-section">
        <h4>3. Activations ({p.precision} typical)</h4>
        <ol>
          <li>Per-layer formula: {perLayerFormula} = <code>{formatBytes(perLayer)}</code></li>
          <li>
            This rank holds <b>{layersOnRank}</b> layers
            {w.recompute_granularity === "full" && w.recompute_method === "uniform" && (
              <>
                {" "}; <b>{recomputedOnRank}</b> are fully recomputed (uniform: every layer);
                each stores only the layer input (<code>2sbh = {formatBytes(fullPerLayer)}</code>)
              </>
            )}
            {w.recompute_granularity === "full" && w.recompute_method === "block" && (
              <>
                {" "}; <b>{recomputedOnRank}</b> are fully recomputed
                (<code>recompute_num_layers={w.recompute_num_layers} × num_chunks_per_rank={numChunksPerRank}</code>,
                capped at this rank's layers); each stores only the layer input
                (<code>2sbh = {formatBytes(fullPerLayer)}</code>)
              </>
            )}
            {w.recompute_granularity === "selective" && <> with selective attention recompute reducing the per-layer cost</>}
          </li>
          <li>
            In-flight microbatches at this PP rank ({pp > 1 ? (vpp > 1 ? "VPP-interleaved" : "1F1B") : "no PP"}): <code>{inFlight}</code>
            {pp > 1 && vpp > 1 && (
              <> &nbsp;(penalty <code>1 + (pp-1)/(pp×vpp) = {interleavedPenalty.toFixed(3)}</code>, applied uniformly to all ranks)</>
            )}
            {pp > 1 && vpp === 1 && (
              <> &nbsp;(<code>min(num_microbatches, pp_size - pp_rank) = {inFlight}</code>; rank 0 holds the most)</>
            )}
          </li>
          {r.partition.has_embedding && (
            <li>First PP rank: + <b>embedding + dropout</b> activations × in_flight ({inFlight} copies)</li>
          )}
          {r.partition.has_final_norm && (
            <li>Last PP rank: + <b>final norm + logits + CE</b> activations (1 microbatch, no in-flight scaling)</li>
          )}
          <li>
            Total activations = <code>{formatBytes(r.memory.activation_bytes)}</code>
          </li>
        </ol>
      </div>

      <div className="calc-section">
        <h4>4. Optimizer main param ({mainParamDtype})</h4>
        <ol>
          <li>{optimizerKindLabel}</li>
          <li>
            Master copy dtype:{" "}
            {p.use_precision_aware_optimizer ? p.optimizer_main_param_dtype : "fp32"}
          </li>
          <li>
            Size = paramsOnRank × dtype_bytes / DP ({dp}) = <code>{formatBytes(r.memory.optimizer.main_param_bytes)}</code>
            {isFsdp && <> (paramsOnRank already sharded by DP for FSDP)</>}
          </li>
        </ol>
      </div>

      <div className="calc-section">
        <h4>5. Optimizer state m+v ({stateDtypeLabel})</h4>
        <ol>
          <li>
            Adam keeps two state tensors: <b>m</b> (momentum) and <b>v</b> (variance).
          </li>
          <li>
            Dtype: {p.use_precision_aware_optimizer
              ? `${p.optimizer_exp_avg_dtype} / ${p.optimizer_exp_avg_sq_dtype}`
              : "fp32 / fp32 (default)"}
          </li>
          <li>
            Sharded across DP ({dp}). Combined = <code>{formatBytes(r.memory.optimizer.state_bytes)}</code>
            {numLayers > 0 && m.moe?.enabled && p.expert_model_parallel_size > 1 && (
              <> ; Note: routed-expert optimizer state is sharded across EDP instead of DP (not yet modeled in v1).</>
            )}
          </li>
        </ol>
      </div>

      <div className="calc-section calc-total">
        <h4>Total: {formatBytes(r.memory.total_bytes)}</h4>
        {state.primaryGpu && (
          <p>
            Primary GPU ({state.primaryGpu.name}) memory: {state.primaryGpu.memory_gb} GB.{" "}
            {r.memory.total_bytes / 1024 ** 3 > state.primaryGpu.memory_gb ? (
              <b className="error-text">Exceeds GPU memory — OOM expected.</b>
            ) : (
              <span>Headroom: {(state.primaryGpu.memory_gb - r.memory.total_bytes / 1024 ** 3).toFixed(2)} GB.</span>
            )}
          </p>
        )}
      </div>
    </details>
  );
}
