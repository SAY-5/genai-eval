import type { RunSummary as Run } from "@/lib/api";
import Link from "next/link";

export function RunSummary({ run }: { run: Run }) {
  return (
    <Link
      href={`/runs/${run.id}`}
      className="block rounded border border-zinc-800 bg-zinc-900/40 px-4 py-3 hover:border-zinc-600"
    >
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm text-zinc-400">
            #{run.id} · <span className="text-zinc-200">{run.model}</span>{" "}
            <span className="text-zinc-500">({run.provider})</span>
          </div>
          <div className="text-xs text-zinc-500">{new Date(run.started_at).toLocaleString()}</div>
        </div>
        <div className="text-right">
          <div className="text-2xl font-semibold tabular-nums">
            {(run.overall_pass_rate * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-zinc-500">
            n={run.n_total} · errors={run.n_errors} · status={run.status}
          </div>
        </div>
      </div>
    </Link>
  );
}
