import { NextResponse } from "next/server";

import { listRuns } from "@/lib/api";

export async function GET() {
  try {
    const runs = await listRuns();
    return NextResponse.json({ runs, count: runs.length });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return NextResponse.json({ error: msg, runs: [] }, { status: 502 });
  }
}
