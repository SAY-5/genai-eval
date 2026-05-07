"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-05-02 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("version_string", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_table(
        "runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("model_version_id", sa.Integer(), sa.ForeignKey("model_versions.id"), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("suite_filter_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("summary_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="running"),
    )
    op.create_table(
        "run_items",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("runs.id"), nullable=False),
        sa.Column("task_type", sa.String(length=32), nullable=False),
        sa.Column("language", sa.String(length=16), nullable=False),
        sa.Column("example_id", sa.String(length=64), nullable=False),
        sa.Column("output_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("scores_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("latency_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="ok"),
        sa.Column("error_text", sa.Text(), nullable=True),
    )
    op.create_index("ix_run_items_run_id", "run_items", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_run_items_run_id", table_name="run_items")
    op.drop_table("run_items")
    op.drop_table("runs")
    op.drop_table("model_versions")
