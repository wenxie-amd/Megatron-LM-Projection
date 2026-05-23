import { useState } from "react";

import { RankBars } from "../components/RankBars";
import { useProjection } from "../state/context";
import { clientValidate, deriveView, parseRankList } from "../state/store";

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

      <p className="hint">
        World size <code>{derived.world_size}</code> · DP <code>{derived.data_parallel_size || "—"}</code>
        {state.modelConfig?.moe.enabled && (
          <>
            {" · EDP "}
            <code>{derived.expert_data_parallel_size || "—"}</code>
          </>
        )}{" "}
        · GA <code>{derived.gradient_accumulation_steps || "—"}</code>
      </p>

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
          {state.projectionRunning ? "Computing…" : "Run projection"}
        </button>
      </div>

      {state.projectionError && <pre className="error">{state.projectionError}</pre>}

      {state.projection && state.projection.rank_reports.length > 0 && (
        <>
          <h3>Per-rank memory</h3>
          <RankBars reports={state.projection.rank_reports} />
          <div className="rank-tables">
            {state.projection.rank_reports.map((r) => (
              <RankCard key={r.global_rank} report={r} />
            ))}
          </div>
        </>
      )}
    </section>
  );
}

function RankCard({ report }: { report: import("../pyodide/types").RankReport }) {
  const total = report.memory.total_bytes || 1;
  const pct = (n: number) => `${((n / total) * 100).toFixed(1)}%`;
  const row = (label: string, bytes: number) => (
    <tr>
      <th>{label}</th>
      <td>{formatBytes(bytes)}</td>
      <td className="pct-col">{pct(bytes)}</td>
    </tr>
  );
  const rc = report.rank_coord;
  return (
    <div className="rank-card">
      <h4>
        Rank {report.global_rank}{" "}
        <span className="rank-coord">
          (tp={rc.tp}, cp={rc.cp}, dp={rc.dp}, pp={rc.pp}, ep={rc.ep}, edp={rc.expert_dp})
        </span>
      </h4>
      <div className="rank-partition">
        layers={report.partition.num_layers_on_rank}
        {report.partition.has_embedding && " · has_embedding"}
        {report.partition.has_final_norm && " · has_final_norm"}
        {report.partition.has_output_projection && " · has_output_projection"}
      </div>
      <table className="memory-table">
        <tbody>
          {row(`params (${report.memory.precision})`, report.memory.param_bytes)}
          {row("main grad buffer", report.memory.optimizer.grad_buffer_bytes)}
          {row("activations", report.memory.activation_bytes)}
          {row("optimizer main param (/ DP)", report.memory.optimizer.main_param_bytes)}
          {row("optimizer state m+v (/ DP)", report.memory.optimizer.state_bytes)}
          <tr className="total">
            <th>total</th>
            <td>{formatBytes(report.memory.total_bytes)}</td>
            <td className="pct-col">100.0%</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
