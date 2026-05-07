"""FastAPI app: runs CRUD + trends + healthz."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel
from sqlalchemy import desc, select
from starlette.responses import Response

from genai_eval.models import (
    HumanScore,
    ModelVersion,
    Run,
    RunItem,
    get_engine,
    get_session_factory,
    init_db,
)
from genai_eval.orchestrator import RunConfig, run_suite
from genai_eval.settings import get_settings

LOG = structlog.get_logger(__name__)

REQS = Counter("genai_eval_requests_total", "Total HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("genai_eval_request_seconds", "Request latency", ["path"])


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    await init_db()
    yield


app = FastAPI(
    title="genai-eval",
    version="0.1.0",
    description="Multilingual GenAI evaluation service",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _telemetry(request: Any, call_next: Any) -> Any:
    path = request.url.path
    with LATENCY.labels(path=path).time():
        response = await call_next(request)
    REQS.labels(method=request.method, path=path, status=str(response.status_code)).inc()
    return response


# ---- schemas ----


class CreateRunRequest(BaseModel):
    provider: str
    model: str
    suite_filter: dict[str, Any] = {}


class CreateRunResponse(BaseModel):
    run_id: int
    status_url: str


class RunSummary(BaseModel):
    id: int
    provider: str
    model: str
    started_at: datetime
    finished_at: datetime | None
    status: str
    overall_pass_rate: float
    n_total: int
    n_errors: int


class RunDetail(BaseModel):
    id: int
    provider: str
    model: str
    started_at: datetime
    finished_at: datetime | None
    status: str
    suite_filter: dict[str, Any]
    summary: dict[str, Any]


class RunItemDTO(BaseModel):
    id: int
    task_type: str
    language: str
    example_id: str
    output_text: str
    scores: dict[str, float]
    latency_ms: float
    cost_usd: float
    status: str
    error_text: str | None


# ---- routes ----


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    """Deep health check: confirms DB engine is reachable."""
    try:
        async with get_engine().connect() as conn:
            await conn.execute(select(1))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"db_unreachable: {exc}") from exc
    return {"status": "ok", "ts": datetime.now(UTC).isoformat()}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/runs", response_model=CreateRunResponse, status_code=202)
async def create_run(payload: CreateRunRequest, background: BackgroundTasks) -> CreateRunResponse:
    """Kick off a run in the background and return its id."""
    settings = get_settings()
    cfg = RunConfig(
        provider_name=payload.provider,
        model=payload.model,
        suite_filter=payload.suite_filter,
        suites_dir=Path(settings.suites_dir),
    )

    Session = get_session_factory()
    async with Session() as session:
        mv = await _ensure_model_version_in_session(session, payload.provider, payload.model)
        row = Run(
            model_version_id=mv.id,
            started_at=datetime.now(UTC),
            status="queued",
        )
        row.suite_filter = payload.suite_filter
        session.add(row)
        await session.commit()
        await session.refresh(row)
        run_id = row.id

    async def _kick() -> None:
        # Fresh orchestrator call; it will write its own Run row, so we wire up the
        # queued row by updating it.
        try:
            result = await run_suite(cfg)
            new_id = result["run_id"]
            async with get_session_factory()() as s:
                # Replace the placeholder with the real one's data.
                placeholder = await s.get(Run, run_id)
                actual = await s.get(Run, new_id)
                if placeholder is not None and actual is not None:
                    placeholder.summary = actual.summary
                    placeholder.status = actual.status
                    placeholder.finished_at = actual.finished_at
                    # Move items from actual to placeholder by repointing.
                    item_stmt = select(RunItem).where(RunItem.run_id == new_id)
                    items = (await s.execute(item_stmt)).scalars().all()
                    for it in items:
                        it.run_id = run_id
                    await s.delete(actual)
                    await s.commit()
        except Exception as exc:  # noqa: BLE001
            LOG.exception("background_run_failed", run_id=run_id, err=str(exc))
            async with get_session_factory()() as s:
                row = await s.get(Run, run_id)
                if row is not None:
                    row.status = "error"
                    row.summary = {"error": str(exc)}
                    await s.commit()

    background.add_task(_kick)

    return CreateRunResponse(run_id=run_id, status_url=f"/v1/runs/{run_id}")


@app.get("/v1/runs", response_model=list[RunSummary])
async def list_runs(
    cursor: int | None = Query(default=None, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[RunSummary]:
    Session = get_session_factory()
    async with Session() as session:
        stmt = (
            select(Run, ModelVersion)
            .join(ModelVersion, ModelVersion.id == Run.model_version_id)
            .order_by(desc(Run.id))
            .limit(limit)
        )
        if cursor is not None:
            stmt = stmt.where(Run.id < cursor)
        rows = (await session.execute(stmt)).all()
        out: list[RunSummary] = []
        for run, mv in rows:
            summary = run.summary
            out.append(
                RunSummary(
                    id=run.id,
                    provider=mv.provider,
                    model=mv.model_name,
                    started_at=run.started_at,
                    finished_at=run.finished_at,
                    status=run.status,
                    overall_pass_rate=float(summary.get("overall_pass_rate", 0.0)),
                    n_total=int(summary.get("n_total", 0)),
                    n_errors=int(summary.get("n_errors", 0)),
                )
            )
        return out


@app.get("/v1/runs/{run_id}", response_model=RunDetail)
async def get_run(run_id: int) -> RunDetail:
    Session = get_session_factory()
    async with Session() as session:
        run = await session.get(Run, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        mv = await session.get(ModelVersion, run.model_version_id)
        assert mv is not None
        return RunDetail(
            id=run.id,
            provider=mv.provider,
            model=mv.model_name,
            started_at=run.started_at,
            finished_at=run.finished_at,
            status=run.status,
            suite_filter=run.suite_filter,
            summary=run.summary,
        )


@app.get("/v1/runs/{run_id}/items", response_model=list[RunItemDTO])
async def list_run_items(
    run_id: int,
    cursor: int | None = Query(default=None, ge=0),
    limit: int = Query(default=200, ge=1, le=1000),
) -> list[RunItemDTO]:
    Session = get_session_factory()
    async with Session() as session:
        stmt = select(RunItem).where(RunItem.run_id == run_id).order_by(RunItem.id).limit(limit)
        if cursor is not None:
            stmt = stmt.where(RunItem.id > cursor)
        rows = (await session.execute(stmt)).scalars().all()
        return [
            RunItemDTO(
                id=r.id,
                task_type=r.task_type,
                language=r.language,
                example_id=r.example_id,
                output_text=r.output_text,
                scores=r.scores,
                latency_ms=r.latency_ms,
                cost_usd=r.cost_usd,
                status=r.status,
                error_text=r.error_text,
            )
            for r in rows
        ]


@app.get("/v1/trends")
async def get_trends(
    model: str | None = Query(default=None),
    task: str | None = Query(default=None),
    language: str | None = Query(default=None),
    since: datetime | None = Query(default=None),
) -> dict[str, Any]:
    """Per-cell time series. Used by the dashboard regression chart."""
    Session = get_session_factory()
    async with Session() as session:
        stmt = (
            select(Run, ModelVersion)
            .join(ModelVersion, ModelVersion.id == Run.model_version_id)
            .where(Run.status == "complete")
            .order_by(Run.id)
        )
        if model:
            stmt = stmt.where(ModelVersion.model_name == model)
        if since:
            stmt = stmt.where(Run.started_at >= since)
        points: list[dict[str, Any]] = []
        for run, mv in (await session.execute(stmt)).all():
            summary = run.summary
            cells = summary.get("cells", [])
            for cell in cells:
                if task and cell["task"] != task:
                    continue
                if language and cell["language"] != language:
                    continue
                points.append(
                    {
                        "run_id": run.id,
                        "model": mv.model_name,
                        "started_at": run.started_at.isoformat(),
                        "task": cell["task"],
                        "language": cell["language"],
                        "pass_rate": cell["pass_rate"],
                    }
                )
        return {"points": points, "count": len(points)}


# ---- human scoring ----


class HumanScoreRequest(BaseModel):
    """Request body for ``POST /v1/runs/{run_id}/items/{idx}/human-score``."""

    score: float
    category: str = "pass"
    rater: str = "anonymous"
    notes: str | None = None


class HumanScoreDTO(BaseModel):
    id: int
    run_id: int
    run_item_id: int
    category: str
    score: float
    rater: str
    notes: str | None
    created_at: datetime


class DisagreementDTO(BaseModel):
    run_item_id: int
    task_type: str
    language: str
    example_id: str
    judge_pass: float
    human_mean: float
    abs_delta: float
    n_human_scores: int


async def _resolve_run_item(session: Any, run_id: int, idx: int) -> RunItem:
    """Resolve the ``idx``-th item in ``run_id`` (0-based, ordered by id)."""
    stmt = select(RunItem).where(RunItem.run_id == run_id).order_by(RunItem.id).offset(idx).limit(1)
    row = (await session.execute(stmt)).scalars().first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"item idx={idx} not found in run {run_id}")
    return row  # type: ignore[no-any-return]


@app.post(
    "/v1/runs/{run_id}/items/{idx}/human-score",
    response_model=HumanScoreDTO,
    status_code=201,
)
async def post_human_score(run_id: int, idx: int, payload: HumanScoreRequest) -> HumanScoreDTO:
    """Record a human score for the ``idx``-th item of ``run_id``."""
    if not (0.0 <= payload.score <= 1.0):
        raise HTTPException(status_code=422, detail="score must be in [0, 1]")
    Session = get_session_factory()
    async with Session() as session:
        item = await _resolve_run_item(session, run_id, idx)
        hs = HumanScore(
            run_id=run_id,
            run_item_id=item.id,
            category=payload.category,
            score=payload.score,
            rater=payload.rater,
            notes=payload.notes,
        )
        session.add(hs)
        await session.commit()
        await session.refresh(hs)
        return HumanScoreDTO(
            id=hs.id,
            run_id=hs.run_id,
            run_item_id=hs.run_item_id,
            category=hs.category,
            score=hs.score,
            rater=hs.rater,
            notes=hs.notes,
            created_at=hs.created_at,
        )


@app.get(
    "/v1/runs/{run_id}/disagreements",
    response_model=list[DisagreementDTO],
)
async def get_disagreements(
    run_id: int,
    category: str = Query(default="pass"),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0),
) -> list[DisagreementDTO]:
    """Items where the judge's pass-score disagrees with the human mean by more than ``threshold``."""
    Session = get_session_factory()
    async with Session() as session:
        items_stmt = select(RunItem).where(RunItem.run_id == run_id).order_by(RunItem.id)
        items = list((await session.execute(items_stmt)).scalars().all())
        scores_stmt = select(HumanScore).where(
            HumanScore.run_id == run_id,
            HumanScore.category == category,
        )
        scores = list((await session.execute(scores_stmt)).scalars().all())
        by_item: dict[int, list[float]] = {}
        for s in scores:
            by_item.setdefault(s.run_item_id, []).append(s.score)
        out: list[DisagreementDTO] = []
        for it in items:
            human = by_item.get(it.id)
            if not human:
                continue
            mean = sum(human) / len(human)
            judge = float(it.scores.get("pass", 0.0))
            delta = abs(judge - mean)
            if delta > threshold:
                out.append(
                    DisagreementDTO(
                        run_item_id=it.id,
                        task_type=it.task_type,
                        language=it.language,
                        example_id=it.example_id,
                        judge_pass=judge,
                        human_mean=mean,
                        abs_delta=delta,
                        n_human_scores=len(human),
                    )
                )
        return out


# ---- helpers ----


async def _ensure_model_version_in_session(session: Any, provider: str, model: str) -> ModelVersion:
    stmt = select(ModelVersion).where(
        ModelVersion.provider == provider,
        ModelVersion.model_name == model,
    )
    existing = (await session.execute(stmt)).scalars().first()
    if existing:
        return existing  # type: ignore[no-any-return]
    mv = ModelVersion(
        provider=provider,
        model_name=model,
        version_string=f"{model}@{provider}",
    )
    session.add(mv)
    await session.flush()
    return mv


# Avoid unused-import warning when running mypy/ruff on this file alone.
_ = asyncio
