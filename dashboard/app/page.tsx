import Link from "next/link";

import { RunSummary } from "@/components/RunSummary";
import { listRuns } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function Home() {
  let runs: Awaited<ReturnType<typeof listRuns>> = [];
  let error: string | null = null;
  try {
    runs = await listRuns(50);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }

  return (
    <div className="space-y-6">
      <section>
        <h2 className="mb-3 text-lg font-semibold">Recent runs</h2>
        {error ? (
          <div className="rounded border border-rose-700 bg-rose-900/30 px-4 py-3 text-sm text-rose-200">
            Could not reach the API at <code>{process.env.GENAI_EVAL_API_URL}</code>: {error}
          </div>
        ) : runs.length === 0 ? (
          <div className="rounded border border-zinc-800 bg-zinc-900/30 px-4 py-3 text-sm text-zinc-400">
            No runs yet. Kick one off with{" "}
            <code className="rounded bg-zinc-800 px-1">make eval</code> or{" "}
            <Link href="/api/runs" className="underline">
              POST /v1/runs
            </Link>
            .
          </div>
        ) : (
          <div className="grid gap-2">
            {runs.map((r) => (
              <RunSummary key={r.id} run={r} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
