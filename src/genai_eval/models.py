"""SQLAlchemy models. SQLite via aiosqlite, async sessions."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from genai_eval.settings import get_settings


class Base(AsyncAttrs, DeclarativeBase):
    pass


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(64))
    model_name: Mapped[str] = mapped_column(String(128))
    version_string: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    runs: Mapped[list[Run]] = relationship("Run", back_populates="model_version")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    suite_filter_json: Mapped[str] = mapped_column(Text, default="{}")
    summary_json: Mapped[str] = mapped_column(Text, default="{}")
    status: Mapped[str] = mapped_column(String(32), default="running")

    model_version: Mapped[ModelVersion] = relationship("ModelVersion", back_populates="runs")
    items: Mapped[list[RunItem]] = relationship(
        "RunItem", back_populates="run", cascade="all, delete-orphan"
    )

    @property
    def suite_filter(self) -> dict[str, Any]:
        return _safe_loads(self.suite_filter_json)

    @suite_filter.setter
    def suite_filter(self, value: dict[str, Any]) -> None:
        self.suite_filter_json = json.dumps(value, ensure_ascii=False)

    @property
    def summary(self) -> dict[str, Any]:
        return _safe_loads(self.summary_json)

    @summary.setter
    def summary(self, value: dict[str, Any]) -> None:
        self.summary_json = json.dumps(value, ensure_ascii=False)


class RunItem(Base):
    __tablename__ = "run_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    task_type: Mapped[str] = mapped_column(String(32))
    language: Mapped[str] = mapped_column(String(16))
    example_id: Mapped[str] = mapped_column(String(64))
    output_text: Mapped[str] = mapped_column(Text, default="")
    scores_json: Mapped[str] = mapped_column(Text, default="{}")
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(16), default="ok")  # ok | error
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped[Run] = relationship("Run", back_populates="items")

    @property
    def scores(self) -> dict[str, float]:
        return _safe_loads(self.scores_json)

    @scores.setter
    def scores(self, value: dict[str, float]) -> None:
        self.scores_json = json.dumps(value, ensure_ascii=False)


def _safe_loads(s: str) -> dict[str, Any]:
    try:
        v = json.loads(s) if s else {}
    except json.JSONDecodeError:
        return {}
    return v if isinstance(v, dict) else {}


# ---- engine / session helpers ----

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> Any:
    global _engine
    if _engine is None:
        _engine = create_async_engine(get_settings().database_url, future=True)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


async def init_db() -> None:
    """Create tables (used in tests + first-time bootstrap)."""
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def reset_engine() -> None:
    """Dispose engine and clear session factory (for tests)."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
