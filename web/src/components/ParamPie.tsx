import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

interface Datum {
  name: string;
  count: number;
}

interface Props {
  data: Datum[];
}

const COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#84cc16", "#f43f5e"];

function formatNumber(n: number): string {
  return n.toLocaleString();
}

export function ParamPie({ data }: Props) {
  const total = data.reduce((acc, d) => acc + d.count, 0);
  return (
    <div className="param-pie">
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={data}
            dataKey="count"
            nameKey="name"
            innerRadius={50}
            outerRadius={110}
            paddingAngle={1}
          >
            {data.map((_, idx) => (
              <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip
            formatter={(value, name) => {
              const v = Number(value);
              return [`${formatNumber(v)} (${((v / total) * 100).toFixed(2)}%)`, String(name)];
            }}
          />
        </PieChart>
      </ResponsiveContainer>
      <ul className="legend">
        {data.map((d, idx) => (
          <li key={d.name}>
            <span className="dot" style={{ background: COLORS[idx % COLORS.length] }} />
            <span className="name">{d.name}</span>
            <span className="count">{formatNumber(d.count)}</span>
            <span className="pct">{((d.count / total) * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
