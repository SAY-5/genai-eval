"""Integration: full eval run + DB persistence + API listing."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from genai_eval.orchestrator import RunConfig, run_suite

SUITES_DIR = Path(__file__).resolve().parent.parent.parent / "eval" / "suites"

_RUN_INTEGRATION = os.environ.get("RUN_INTEGRATION") == "1"


pytestmark = pytest.mark.skipif(not _RUN_INTEGRATION, reason="set RUN_INTEGRATION=1 to enable")


@pytest.mark.asyncio
async def test_full_run_persists_and_lists(fresh_db: None) -> None:  # noqa: ARG001
    cfg = RunConfig(
        provider_name="fake",
        model="fake-large",
        suite_filter={},
        suites_dir=SUITES_DIR,
    )
    result = await run_suite(cfg)
    run_id = result["run_id"]
    summary = result["summary"]
    assert summary["n_total"] > 0
    assert "cells" in summary

    from genai_eval.api import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # /healthz
        r = await client.get("/healthz")
        assert r.status_code == 200

        # list runs
        r = await client.get("/v1/runs")
        assert r.status_code == 200
        runs = r.json()
        assert any(run["id"] == run_id for run in runs)

        # detail
        r = await client.get(f"/v1/runs/{run_id}")
        assert r.status_code == 200
        detail = r.json()
        assert detail["status"] == "complete"
        assert detail["summary"]["n_total"] == summary["n_total"]

        # items
        r = await client.get(f"/v1/runs/{run_id}/items")
        assert r.status_code == 200
        items = r.json()
        assert len(items) == summary["n_total"]

        # trends
        r = await client.get("/v1/trends?model=fake-large")
        assert r.status_code == 200
        trends = r.json()
        assert trends["count"] >= len(summary["cells"])
