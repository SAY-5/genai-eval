"""Click CLI for batch eval runs."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import click

from genai_eval.orchestrator import RunConfig, run_suite
from genai_eval.settings import get_settings


@click.group()
def main() -> None:
    """genai-eval CLI."""


@main.command("run")
@click.option("--suite", default="all", show_default=True, help="Comma-separated tasks or 'all'.")
@click.option("--language", default=None, help="Optional comma-separated language codes.")
@click.option("--provider", default="fake", show_default=True)
@click.option("--model", default="fake-large", show_default=True)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the summary as JSON to this path.",
)
@click.option(
    "--smoke", is_flag=True, default=False, help="Take only first 2 examples per (task, lang) cell."
)
def run_cmd(
    suite: str,
    language: str | None,
    provider: str,
    model: str,
    output: Path | None,
    smoke: bool,
) -> None:
    """Run the eval matrix and optionally write the summary to disk."""
    settings = get_settings()

    suite_filter: dict[str, object] = {}
    if suite != "all":
        suite_filter["tasks"] = [t.strip() for t in suite.split(",") if t.strip()]
    if language:
        suite_filter["languages"] = [s.strip() for s in language.split(",") if s.strip()]

    cfg = RunConfig(
        provider_name=provider,
        model=model,
        suite_filter=suite_filter,
        suites_dir=Path(settings.suites_dir),
        smoke=smoke,
    )

    result = asyncio.run(run_suite(cfg))
    payload = {
        "run_id": result["run_id"],
        "provider": provider,
        "model": model,
        "produced_at": datetime.now(UTC).isoformat(),
        "suite_filter": suite_filter,
        "summary": result["summary"],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    click.echo(text)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        click.echo(f"\nWrote {output}")


@main.command("compare")
@click.option("--model-a", required=True, help="Model name for arm A (e.g. fake-small).")
@click.option("--model-b", required=True, help="Model name for arm B (e.g. fake-large).")
@click.option("--provider", default="fake", show_default=True, help="Provider for both arms.")
@click.option(
    "--suite",
    default="all",
    show_default=True,
    help="Comma-separated tasks or 'all'.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Write the comparison Markdown report to this path.",
)
@click.option(
    "--persist/--no-persist",
    default=True,
    show_default=True,
    help="Persist the Comparison row in the database.",
)
def compare_cmd(
    model_a: str,
    model_b: str,
    provider: str,
    suite: str,
    output: Path,
    persist: bool,
) -> None:
    """Run two evals back-to-back and emit a Markdown A/B report."""
    from typing import Any as _Any

    from genai_eval.comparisons import Comparison, compare_cells, render_markdown
    from genai_eval.models import get_session_factory, init_db

    settings = get_settings()

    suite_filter: dict[str, object] = {}
    if suite != "all":
        suite_filter["tasks"] = [t.strip() for t in suite.split(",") if t.strip()]

    async def _exec() -> tuple[dict[str, _Any], dict[str, _Any]]:
        cfg_a = RunConfig(
            provider_name=provider,
            model=model_a,
            suite_filter=suite_filter,
            suites_dir=Path(settings.suites_dir),
        )
        cfg_b = RunConfig(
            provider_name=provider,
            model=model_b,
            suite_filter=suite_filter,
            suites_dir=Path(settings.suites_dir),
        )
        a = await run_suite(cfg_a)
        b = await run_suite(cfg_b)
        return a, b

    res_a, res_b = asyncio.run(_exec())

    cells = compare_cells(res_a["summary"], res_b["summary"])
    overall_a = float(res_a["summary"].get("overall_pass_rate", 0.0))
    overall_b = float(res_b["summary"].get("overall_pass_rate", 0.0))

    md = render_markdown(
        model_a=model_a,
        model_b=model_b,
        suite=suite,
        cells=cells,
        overall_a=overall_a,
        overall_b=overall_b,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(md, encoding="utf-8")
    click.echo(md)

    if persist:

        async def _save() -> None:
            await init_db()
            Session = get_session_factory()
            async with Session() as session:
                comp = Comparison(
                    run_a_id=int(res_a["run_id"]),
                    run_b_id=int(res_b["run_id"]),
                    suite=suite,
                )
                comp.summary = {
                    "model_a": model_a,
                    "model_b": model_b,
                    "overall_a": overall_a,
                    "overall_b": overall_b,
                    "cells": [c.to_dict() for c in cells],
                }
                session.add(comp)
                await session.commit()
                click.echo(f"\nWrote comparison row id={comp.id}")

        asyncio.run(_save())

    click.echo(f"\nWrote {output}")


@main.command("seed")
def seed_cmd() -> None:
    """Initialise DB and create a fake-large ModelVersion row."""
    from genai_eval.models import ModelVersion, get_session_factory, init_db

    async def _seed() -> None:
        await init_db()
        Session = get_session_factory()
        async with Session() as session:
            mv = ModelVersion(
                provider="fake",
                model_name="fake-large",
                version_string="fake-large@fake",
            )
            session.add(mv)
            await session.commit()
            click.echo(f"Seeded model_version id={mv.id}")

    asyncio.run(_seed())


if __name__ == "__main__":
    main()
