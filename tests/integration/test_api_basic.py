"""API smoke tests gated on RUN_INTEGRATION=1."""

from __future__ import annotations

import os

import pytest
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1", reason="set RUN_INTEGRATION=1 to enable"
)


@pytest.mark.asyncio
async def test_healthz(fresh_db: None) -> None:  # noqa: ARG001
    from genai_eval.api import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_metrics_exposes_prometheus_format(fresh_db: None) -> None:  # noqa: ARG001
    from genai_eval.api import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/metrics")
        assert r.status_code == 200
        # Prometheus text format starts with "# HELP" or "# TYPE" lines.
        body = r.text
        assert "genai_eval_" in body or "# HELP" in body


@pytest.mark.asyncio
async def test_run_not_found_returns_404(fresh_db: None) -> None:  # noqa: ARG001
    from genai_eval.api import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/runs/999999")
        assert r.status_code == 404
