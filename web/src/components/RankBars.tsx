import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { RankReport } from "../pyodide/types";

interface Props {
  reports: RankReport[];
}

const GIB = 1024 ** 3;

function toGiB(bytes: number): number {
  return Number((bytes / GIB).toFixed(2));
}

export function RankBars({ reports }: Props) {
  const data = reports.map((r) => ({
    rank: `rank ${r.global_rank}`,
    params: toGiB(r.memory.param_bytes),
    grad_buffer: toGiB(r.memory.optimizer.grad_buffer_bytes),
    activations: toGiB(r.memory.activation_bytes),
    optimizer_main_param: toGiB(r.memory.optimizer.main_param_bytes),
    optimizer_state: toGiB(r.memory.optimizer.state_bytes),
  }));
  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart data={data} stackOffset="none">
        <CartesianGrid strokeDasharray="3 3" stroke="#ddd" />
        <XAxis dataKey="rank" />
        <YAxis label={{ value: "GiB", angle: -90, position: "insideLeft" }} />
        <Tooltip formatter={(v) => `${Number(v)} GiB`} />
        <Legend />
        <Bar stackId="m" dataKey="params" fill="#3b82f6" name="params (bf16)" />
        <Bar stackId="m" dataKey="grad_buffer" fill="#10b981" name="main grad buffer" />
        <Bar stackId="m" dataKey="activations" fill="#f59e0b" name="activations" />
        <Bar stackId="m" dataKey="optimizer_main_param" fill="#8b5cf6" name="optimizer main param" />
        <Bar stackId="m" dataKey="optimizer_state" fill="#ec4899" name="optimizer state (m+v)" />
      </BarChart>
    </ResponsiveContainer>
  );
}
