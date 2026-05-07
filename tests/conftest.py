"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio


@pytest.fixture(scope="session")
def suites_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "eval" / "suites"


@pytest_asyncio.fixture
async def fresh_db(tmp_path: Path) -> AsyncIterator[None]:
    """Point the app at an isolated SQLite file for the duration of one test."""
    db_path = tmp_path / "test.db"
    os.environ["GENAI_EVAL_DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"
    os.environ["GENAI_EVAL_DATABASE_URL_SYNC"] = f"sqlite:///{db_path}"
    # Clear cached settings + engine so they pick up the new URL.
    from genai_eval import models, settings

    settings._settings = None
    await models.reset_engine()
    await models.init_db()
    try:
        yield
    finally:
        await models.reset_engine()
        settings._settings = None
        os.environ.pop("GENAI_EVAL_DATABASE_URL", None)
        os.environ.pop("GENAI_EVAL_DATABASE_URL_SYNC", None)
