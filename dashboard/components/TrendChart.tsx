"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { TrendPoint } from "@/lib/api";

export type TrendChartProps = { points: TrendPoint[] };

export function TrendChart({ points }: TrendChartProps) {
  // group points by task/lang into a series
  const byKey = new Map<string, { name: string; data: { run: number; rate: number }[] }>();
  for (const p of points) {
    const key = `${p.task}/${p.language}`;
    if (!byKey.has(key)) byKey.set(key, { name: key, data: [] });
    byKey.get(key)!.data.push({ run: p.run_id, rate: Math.round(p.pass_rate * 1000) / 10 });
  }

  // Pivot for recharts: X = run, lines = key.
  const xs = Array.from(new Set(points.map((p) => p.run_id))).sort((a, b) => a - b);
  const rows = xs.map((run) => {
    const row: Record<string, number> = { run };
    for (const [key, series] of byKey) {
      const hit = series.data.find((d) => d.run === run);
      if (hit) row[key] = hit.rate;
    }
    return row;
  });

  const colors = ["#34d399", "#60a5fa", "#fbbf24", "#f472b6", "#a78bfa", "#fb7185"];

  return (
    <div className="h-72 w-full" data-testid="trend-chart">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis dataKey="run" stroke="#71717a" />
          <YAxis stroke="#71717a" domain={[0, 100]} />
          <Tooltip contentStyle={{ background: "#18181b", border: "1px solid #3f3f46" }} />
          <Legend wrapperStyle={{ color: "#a1a1aa" }} />
          {Array.from(byKey.keys()).map((key, i) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={colors[i % colors.length]}
              dot={false}
              strokeWidth={2}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
