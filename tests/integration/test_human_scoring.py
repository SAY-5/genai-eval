"""Integration: human-in-the-loop scoring API + calibration logic."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from genai_eval.orchestrator import RunConfig, run_suite

SUITES_DIR = Path(__file__).resolve().parent.parent.parent / "eval" / "suites"

_RUN_INTEGRATION = os.environ.get("RUN_INTEGRATION") == "1"
pytestmark = pytest.mark.skipif(not _RUN_INTEGRATION, reason="set RUN_INTEGRATION=1 to enable")


@pytest.mark.asyncio
async def test_post_ten_human_scores_and_disagreement_rate(fresh_db: None) -> None:  # noqa: ARG001
    """Post 10 human scores and verify the disagreements + calibration math.

    Strategy:
      * Run the qa suite at small scale.
      * For 10 items, post a human score equal to the judge score (perfect
        agreement); for the next item, post a contradicting score.
      * Assert the /disagreements endpoint flags exactly the contradicting
        ones and the calibration agreement rate matches expectation.
    """
    cfg = RunConfig(
        provider_name="fake",
        model="fake-large",
        suite_filter={"tasks": ["qa"]},
        suites_dir=SUITES_DIR,
    )
    result = await run_suite(cfg)
    run_id = result["run_id"]

    from genai_eval.api import app
    from genai_eval.models import HumanScore, RunItem, get_session_factory

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Pull the first 11 items so we can score them.
        r = await client.get(f"/v1/runs/{run_id}/items?limit=20")
        assert r.status_code == 200
        items = r.json()
        assert len(items) >= 11

        # Idx 0..9: post a human score that agrees with the judge "pass" score.
        for idx in range(10):
            judge_pass = float(items[idx]["scores"].get("pass", 0.0))
            r = await client.post(
                f"/v1/runs/{run_id}/items/{idx}/human-score",
                json={
                    "score": judge_pass,
                    "category": "pass",
                    "rater": "alice",
                    "notes": "agree",
                },
            )
            assert r.status_code == 201, r.text
            body = r.json()
            assert body["score"] == judge_pass
            assert body["category"] == "pass"

        # Idx 10: a sharp disagreement so we have one entry in /disagreements.
        judge_pass_10 = float(items[10]["scores"].get("pass", 0.0))
        contradicting = 1.0 if judge_pass_10 < 0.5 else 0.0
        r = await client.post(
            f"/v1/runs/{run_id}/items/10/human-score",
            json={"score": contradicting, "category": "pass", "rater": "alice"},
        )
        assert r.status_code == 201

        # Verify all 11 rows landed.
        Session = get_session_factory()
        async with Session() as session:
            count_stmt = select(HumanScore).where(HumanScore.run_id == run_id)
            saved = list((await session.execute(count_stmt)).scalars().all())
            assert len(saved) == 11

        # Disagreements endpoint: only the last item should be flagged.
        r = await client.get(f"/v1/runs/{run_id}/disagreements?category=pass&threshold=0.5")
        assert r.status_code == 200
        disagreements = r.json()
        assert len(disagreements) == 1
        d = disagreements[0]
        assert d["abs_delta"] == pytest.approx(1.0, abs=1e-9)
        assert d["n_human_scores"] == 1

        # Calibration math: 10 of 11 agree at threshold=0.5.
        async with Session() as session:
            scores = list(
                (await session.execute(select(HumanScore).where(HumanScore.run_id == run_id)))
                .scalars()
                .all()
            )
            items_db = {
                it.id: it
                for it in (await session.execute(select(RunItem).where(RunItem.run_id == run_id)))
                .scalars()
                .all()
            }
            agreements = 0
            for s in scores:
                judge = float(items_db[s.run_item_id].scores.get("pass", 0.0))
                if abs(judge - s.score) <= 0.5:
                    agreements += 1
            assert agreements == 10
            assert agreements / len(scores) == pytest.approx(10 / 11, abs=1e-9)


@pytest.mark.asyncio
async def test_post_human_score_validates_score_range(fresh_db: None) -> None:  # noqa: ARG001
    cfg = RunConfig(
        provider_name="fake",
        model="fake-large",
        suite_filter={"tasks": ["qa"]},
        suites_dir=SUITES_DIR,
    )
    await run_suite(cfg)

    from genai_eval.api import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/v1/runs/1/items/0/human-score",
            json={"score": 1.5, "category": "pass"},
        )
        assert r.status_code == 422


@pytest.mark.asyncio
async def test_disagreements_returns_empty_for_unscored_run(fresh_db: None) -> None:  # noqa: ARG001
    cfg = RunConfig(
        provider_name="fake",
        model="fake-large",
        suite_filter={"tasks": ["qa"]},
        suites_dir=SUITES_DIR,
    )
    result = await run_suite(cfg)

    from genai_eval.api import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get(f"/v1/runs/{result['run_id']}/disagreements")
        assert r.status_code == 200
        assert r.json() == []
