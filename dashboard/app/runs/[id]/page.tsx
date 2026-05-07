import { notFound } from "next/navigation";

import { PassRateGrid } from "@/components/PassRateGrid";
import { getRun, getTrends, listRunItems } from "@/lib/api";
import { TrendChart } from "@/components/TrendChart";

export const dynamic = "force-dynamic";

export default async function RunDetailPage({ params }: { params: { id: string } }) {
  const id = Number(params.id);
  if (!Number.isFinite(id)) notFound();

  let detail;
  try {
    detail = await getRun(id);
  } catch {
    notFound();
  }
  if (!detail) notFound();

  const cells = detail.summary.cells || [];
  const items = await listRunItems(id).catch(() => []);
  const trends = await getTrends({ model: detail.model }).catch(() => ({ points: [], count: 0 }));

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-lg font-semibold">
          Run #{detail.id} · {detail.model}{" "}
          <span className="text-zinc-500">({detail.provider})</span>
        </h2>
        <div className="mt-1 text-sm text-zinc-400">
          Started {new Date(detail.started_at).toLocaleString()} · status {detail.status}
        </div>
      </section>

      <section>
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-zinc-400">
          Pass rates
        </h3>
        <PassRateGrid cells={cells} />
      </section>

      <section>
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-zinc-400">
          Trend across runs (model = {detail.model})
        </h3>
        {trends.points.length > 0 ? (
          <TrendChart points={trends.points} />
        ) : (
          <div className="text-sm text-zinc-500">Not enough history yet.</div>
        )}
      </section>

      <section>
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-zinc-400">
          Items ({items.length})
        </h3>
        <div className="overflow-x-auto rounded border border-zinc-800">
          <table className="w-full text-sm">
            <thead className="bg-zinc-900/60 text-left text-xs uppercase tracking-wide text-zinc-400">
              <tr>
                <th className="px-3 py-2">id</th>
                <th className="px-3 py-2">task</th>
                <th className="px-3 py-2">lang</th>
                <th className="px-3 py-2">example</th>
                <th className="px-3 py-2">pass</th>
                <th className="px-3 py-2">latency</th>
                <th className="px-3 py-2">output (truncated)</th>
              </tr>
            </thead>
            <tbody>
              {items.map((it) => (
                <tr key={it.id} className="border-t border-zinc-900">
                  <td className="px-3 py-2 text-zinc-500">{it.id}</td>
                  <td className="px-3 py-2">{it.task_type}</td>
                  <td className="px-3 py-2">{it.language}</td>
                  <td className="px-3 py-2 text-zinc-400">{it.example_id}</td>
                  <td
                    className={
                      "px-3 py-2 " +
                      ((it.scores?.pass ?? 0) >= 1 ? "text-emerald-400" : "text-rose-400")
                    }
                  >
                    {(it.scores?.pass ?? 0) >= 1 ? "pass" : "fail"}
                  </td>
                  <td className="px-3 py-2 tabular-nums text-zinc-400">
                    {it.latency_ms.toFixed(1)}ms
                  </td>
                  <td className="max-w-md truncate px-3 py-2 text-zinc-300">
                    {it.output_text.replace(/\s+/g, " ").slice(0, 120)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
