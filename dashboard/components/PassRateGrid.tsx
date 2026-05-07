import type { RunCell } from "@/lib/api";

const TASKS = ["summarization", "translation", "qa", "classification", "code_repair"];
const LANGS = ["en", "es", "ja", "en-es", "en-ja", "es-en", "py"];

function colorFor(rate: number): string {
  if (rate >= 0.85) return "bg-emerald-700 text-emerald-50";
  if (rate >= 0.6) return "bg-amber-700 text-amber-50";
  if (rate > 0) return "bg-rose-800 text-rose-50";
  return "bg-zinc-800 text-zinc-500";
}

export function PassRateGrid({ cells }: { cells: RunCell[] }) {
  const byKey = new Map<string, RunCell>();
  for (const c of cells) byKey.set(`${c.task}|${c.language}`, c);
  const langs = LANGS.filter((l) => cells.some((c) => c.language === l));

  return (
    <div className="overflow-x-auto">
      <table
        className="w-full border-separate border-spacing-1 text-sm"
        data-testid="pass-rate-grid"
      >
        <thead>
          <tr>
            <th className="text-left text-zinc-400 px-2 py-1">task ╲ lang</th>
            {langs.map((l) => (
              <th key={l} className="px-2 py-1 text-zinc-400">
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {TASKS.map((task) => (
            <tr key={task}>
              <td className="px-2 py-1 font-medium text-zinc-200">{task}</td>
              {langs.map((lang) => {
                const cell = byKey.get(`${task}|${lang}`);
                if (!cell) {
                  return (
                    <td key={lang} className="px-2 py-1">
                      <div className="h-10 rounded bg-zinc-900/40" />
                    </td>
                  );
                }
                return (
                  <td key={lang} className="px-2 py-1">
                    <div
                      className={`flex h-10 flex-col items-center justify-center rounded px-2 ${colorFor(
                        cell.pass_rate,
                      )}`}
                      title={`n=${cell.n}, errors=${cell.errors}`}
                      data-pass-rate={cell.pass_rate}
                    >
                      <span className="font-semibold">{(cell.pass_rate * 100).toFixed(0)}%</span>
                      <span className="text-[10px] opacity-80">n={cell.n}</span>
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
