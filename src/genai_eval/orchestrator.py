"""Run the task × language matrix end-to-end and persist results."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import select

from genai_eval.models import ModelVersion, Run, RunItem, get_session_factory, init_db
from genai_eval.providers import ChatProvider, build_provider
from genai_eval.tasks import Example, classification, code_repair, qa, summarization, translation

LOG = structlog.get_logger(__name__)


# Task module dispatch table.
TASK_MODULES = {
    "summarization": summarization,
    "translation": translation,
    "qa": qa,
    "classification": classification,
    "code_repair": code_repair,
}

# Per-task language enumeration. translation enumerates pairs in its YAML;
# code_repair is language-agnostic (Python).
TASK_LANGUAGES: dict[str, tuple[str, ...]] = {
    "summarization": ("en", "es", "ja"),
    "translation": ("all",),
    "qa": ("en", "es", "ja"),
    "classification": ("en", "es", "ja"),
    "code_repair": ("py",),
}


@dataclass
class RunConfig:
    provider_name: str
    model: str
    suite_filter: dict[str, Any]
    suites_dir: Path
    smoke: bool = False  # if True, take only first 2 examples per (task, lang)


@dataclass
class ItemResult:
    task_type: str
    language: str
    example_id: str
    output_text: str
    scores: dict[str, float]
    latency_ms: float
    cost_usd: float
    status: str
    error_text: str | None


def collect_examples(cfg: RunConfig) -> list[Example]:
    """Apply suite_filter, return all examples to run."""
    selected_tasks = cfg.suite_filter.get("tasks") or list(TASK_MODULES.keys())
    selected_langs = cfg.suite_filter.get("languages")  # None = all
    examples: list[Example] = []
    for task in selected_tasks:
        if task not in TASK_MODULES:
            continue
        module = TASK_MODULES[task]
        for lang in TASK_LANGUAGES[task]:
            batch = module.load_suite(lang, cfg.suites_dir)
            if selected_langs and task != "code_repair":
                # filter by language tag (works for plain en/es/ja and pair codes)
                batch = [e for e in batch if any(code in e.language for code in selected_langs)]
            if cfg.smoke:
                batch = batch[:2]
            examples.extend(batch)
    return examples


async def _run_one(
    ex: Example,
    provider: ChatProvider,
    model: str,
    sem: asyncio.Semaphore,
) -> ItemResult:
    """Score a single example, swallowing exceptions into status='error'."""
    module = TASK_MODULES[ex.task_type]
    messages = module.build_messages(ex)
    started = time.perf_counter()
    try:
        async with sem:
            result = await provider.chat(messages, model=model)
        scores = module.score(ex, result.text)
        return ItemResult(
            task_type=ex.task_type,
            language=ex.language,
            example_id=ex.id,
            output_text=result.text,
            scores=scores,
            latency_ms=result.latency_ms,
            cost_usd=result.cost_usd,
            status="ok",
            error_text=None,
        )
    except Exception as exc:  # noqa: BLE001 — by design, single example failure is non-fatal
        elapsed = (time.perf_counter() - started) * 1000
        LOG.warning("example_failed", task=ex.task_type, lang=ex.language, id=ex.id, err=str(exc))
        return ItemResult(
            task_type=ex.task_type,
            language=ex.language,
            example_id=ex.id,
            output_text="",
            scores={"pass": 0.0},
            latency_ms=elapsed,
            cost_usd=0.0,
            status="error",
            error_text=str(exc),
        )


def summarise(items: list[ItemResult]) -> dict[str, Any]:
    """Aggregate per-(task, language) pass rate, mean cost, P95 latency."""
    by_cell: dict[tuple[str, str], list[ItemResult]] = {}
    for it in items:
        by_cell.setdefault((it.task_type, it.language), []).append(it)

    cells: list[dict[str, Any]] = []
    for (task, lang), bucket in sorted(by_cell.items()):
        passes = [i.scores.get("pass", 0.0) for i in bucket]
        latencies = sorted(i.latency_ms for i in bucket)
        p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
        cells.append(
            {
                "task": task,
                "language": lang,
                "n": len(bucket),
                "pass_rate": sum(passes) / len(passes) if passes else 0.0,
                "mean_cost_usd": sum(i.cost_usd for i in bucket) / len(bucket) if bucket else 0.0,
                "p95_latency_ms": p95,
                "errors": sum(1 for i in bucket if i.status == "error"),
            }
        )

    overall_passes = [i.scores.get("pass", 0.0) for i in items]
    return {
        "cells": cells,
        "overall_pass_rate": sum(overall_passes) / len(overall_passes) if overall_passes else 0.0,
        "n_total": len(items),
        "n_errors": sum(1 for i in items if i.status == "error"),
    }


async def run_suite(cfg: RunConfig) -> dict[str, Any]:
    """Execute the matrix, persist Run + RunItems, return the summary dict."""
    await init_db()
    provider = build_provider(cfg.provider_name)
    examples = collect_examples(cfg)
    sem = asyncio.Semaphore(8)

    # Persist a Run row up front so the API can stream status if needed.
    Session = get_session_factory()
    async with Session() as session:
        mv = await _ensure_model_version(session, cfg.provider_name, cfg.model)
        run_row = Run(
            model_version_id=mv.id,
            started_at=datetime.now(UTC),
            status="running",
        )
        run_row.suite_filter = cfg.suite_filter
        session.add(run_row)
        await session.commit()
        await session.refresh(run_row)
        run_id = run_row.id

    LOG.info(
        "run_started",
        run_id=run_id,
        n_examples=len(examples),
        provider=cfg.provider_name,
        model=cfg.model,
    )

    coros = [_run_one(ex, provider, cfg.model, sem) for ex in examples]
    results: list[ItemResult] = await asyncio.gather(*coros)
    summary = summarise(results)

    async with Session() as session:
        for it in results:
            row = RunItem(
                run_id=run_id,
                task_type=it.task_type,
                language=it.language,
                example_id=it.example_id,
                output_text=it.output_text,
                latency_ms=it.latency_ms,
                cost_usd=it.cost_usd,
                status=it.status,
                error_text=it.error_text,
            )
            row.scores = it.scores
            session.add(row)

        finishing = await session.get(Run, run_id)
        assert finishing is not None
        finishing.summary = summary
        finishing.finished_at = datetime.now(UTC)
        finishing.status = "complete"
        await session.commit()

    LOG.info("run_complete", run_id=run_id, overall_pass_rate=summary["overall_pass_rate"])

    return {"run_id": run_id, "summary": summary}


async def _ensure_model_version(session: Any, provider: str, model: str) -> ModelVersion:
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
