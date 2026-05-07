"""add human_scores table

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-07 23:30:00.000000

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "human_scores",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("runs.id"), nullable=False),
        sa.Column(
            "run_item_id", sa.Integer(), sa.ForeignKey("run_items.id"), nullable=False
        ),
        sa.Column("category", sa.String(length=64), nullable=False, server_default="pass"),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("rater", sa.String(length=64), nullable=False, server_default="anonymous"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_human_scores_run_id", "human_scores", ["run_id"])
    op.create_index("ix_human_scores_run_item_id", "human_scores", ["run_item_id"])


def downgrade() -> None:
    op.drop_index("ix_human_scores_run_item_id", table_name="human_scores")
    op.drop_index("ix_human_scores_run_id", table_name="human_scores")
    op.drop_table("human_scores")
