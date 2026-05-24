import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { OptimizerDtype, Precision, RankReport } from "../pyodide/types";

interface Props {
  reports: RankReport[];
  /** GPU memory in GiB drawn as a horizontal roofline. */
  roofLineGiB?: number;
  roofLineLabel?: string;
  /** Actual optimizer dtypes (from ParallelConfig). Default fp32 / fp32 / fp32. */
  mainParamDtype?: OptimizerDtype;
  expAvgDtype?: OptimizerDtype;
  expAvgSqDtype?: OptimizerDtype;
  mainGradDtype?: OptimizerDtype;
}

const GIB = 1024 ** 3;

function toGiB(bytes: number): number {
  return Number((bytes / GIB).toFixed(2));
}

function precisionLabel(p: Precision): string {
  return p.toUpperCase();
}

export function RankBars({
  reports,
  roofLineGiB,
  roofLineLabel,
  mainParamDtype = "fp32",
  expAvgDtype = "fp32",
  expAvgSqDtype = "fp32",
  mainGradDtype = "fp32",
}: Props) {
  const data = reports.map((r) => ({
    rank: `rank ${r.global_rank}`,
    params: toGiB(r.memory.param_bytes),
    activations: toGiB(r.memory.activation_bytes),
    grad_buffer: toGiB(r.memory.optimizer.grad_buffer_bytes),
    optimizer_main_param: toGiB(r.memory.optimizer.main_param_bytes),
    optimizer_state: toGiB(r.memory.optimizer.state_bytes),
  }));

  const probe = reports[0];
  const paramLabel = probe ? `params (${precisionLabel(probe.memory.precision)})` : "params";
  const activationLabel = "activations";
  const gradLabel = `main grad buffer (${mainGradDtype})`;
  const mainParamLabel = `optimizer main param (${mainParamDtype})`;
  const stateLabel =
    expAvgDtype === expAvgSqDtype
      ? `optimizer state m+v (${expAvgDtype})`
      : `optimizer state m+v (${expAvgDtype}/${expAvgSqDtype})`;

  // Recharts stacks bars in JSX order from bottom up, so the LAST <Bar> goes on
  // top. We want, top → bottom: params, activations, grad buffer, main param,
  // state — i.e. reverse-order JSX. Both Legend and Tooltip use itemSorter to
  // show items in user-requested order regardless of bar render order.
  const orderKey = (dataKey: unknown): number => {
    const order: Record<string, number> = {
      params: 0,
      activations: 1,
      grad_buffer: 2,
      optimizer_main_param: 3,
      optimizer_state: 4,
    };
    return order[String(dataKey)] ?? 99;
  };

  return (
    <ResponsiveContainer width="100%" height={360}>
      <BarChart data={data} stackOffset="none">
        <CartesianGrid strokeDasharray="3 3" stroke="#ddd" />
        <XAxis dataKey="rank" />
        <YAxis label={{ value: "GiB", angle: -90, position: "insideLeft" }} />
        <Tooltip formatter={(v) => `${Number(v)} GiB`} itemSorter={(item) => orderKey(item.dataKey)} />
        <Legend itemSorter={(item) => orderKey(item.dataKey)} />
        <Bar stackId="m" dataKey="optimizer_state" fill="#ec4899" name={stateLabel} />
        <Bar stackId="m" dataKey="optimizer_main_param" fill="#8b5cf6" name={mainParamLabel} />
        <Bar stackId="m" dataKey="grad_buffer" fill="#10b981" name={gradLabel} />
        <Bar stackId="m" dataKey="activations" fill="#f59e0b" name={activationLabel} />
        <Bar stackId="m" dataKey="params" fill="#3b82f6" name={paramLabel} />
        {roofLineGiB !== undefined && (
          <ReferenceLine
            y={Number(roofLineGiB.toFixed(2))}
            stroke="#dc2626"
            strokeDasharray="6 4"
            label={{
              value: `${roofLineLabel ?? "GPU"} memory ${roofLineGiB.toFixed(1)} GiB`,
              position: "insideTopRight",
              fill: "#dc2626",
              fontSize: 12,
            }}
          />
        )}
      </BarChart>
    </ResponsiveContainer>
  );
}
