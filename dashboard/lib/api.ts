// Thin client for the genai-eval Python API.

const BASE = process.env.GENAI_EVAL_API_URL || "http://localhost:8000";

export type RunSummary = {
  id: number;
  provider: string;
  model: string;
  started_at: string;
  finished_at: string | null;
  status: string;
  overall_pass_rate: number;
  n_total: number;
  n_errors: number;
};

export type RunCell = {
  task: string;
  language: string;
  n: number;
  pass_rate: number;
  mean_cost_usd: number;
  p95_latency_ms: number;
  errors: number;
};

export type RunDetail = {
  id: number;
  provider: string;
  model: string;
  started_at: string;
  finished_at: string | null;
  status: string;
  suite_filter: Record<string, unknown>;
  summary: { cells?: RunCell[]; overall_pass_rate?: number; n_total?: number; n_errors?: number };
};

export type RunItem = {
  id: number;
  task_type: string;
  language: string;
  example_id: string;
  output_text: string;
  scores: Record<string, number>;
  latency_ms: number;
  cost_usd: number;
  status: string;
  error_text: string | null;
};

export type TrendPoint = {
  run_id: number;
  model: string;
  started_at: string;
  task: string;
  language: string;
  pass_rate: number;
};

async function fetchJson<T>(path: string): Promise<T> {
  const url = `${BASE}${path}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`${url} -> ${res.status}`);
  }
  return (await res.json()) as T;
}

export async function listRuns(limit = 50): Promise<RunSummary[]> {
  return fetchJson<RunSummary[]>(`/v1/runs?limit=${limit}`);
}

export async function getRun(id: number): Promise<RunDetail> {
  return fetchJson<RunDetail>(`/v1/runs/${id}`);
}

export async function listRunItems(id: number): Promise<RunItem[]> {
  return fetchJson<RunItem[]>(`/v1/runs/${id}/items`);
}

export async function getTrends(params: {
  model?: string;
  task?: string;
  language?: string;
}): Promise<{ points: TrendPoint[]; count: number }> {
  const qs = new URLSearchParams();
  if (params.model) qs.set("model", params.model);
  if (params.task) qs.set("task", params.task);
  if (params.language) qs.set("language", params.language);
  return fetchJson(`/v1/trends?${qs.toString()}`);
}
