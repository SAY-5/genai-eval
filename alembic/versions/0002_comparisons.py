"""add comparisons table

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-07 23:00:00.000000

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "comparisons",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_a_id", sa.Integer(), sa.ForeignKey("runs.id"), nullable=False),
        sa.Column("run_b_id", sa.Integer(), sa.ForeignKey("runs.id"), nullable=False),
        sa.Column("suite", sa.String(length=64), nullable=False, server_default="all"),
        sa.Column("summary_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column(
            "produced_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_comparisons_run_a_id", "comparisons", ["run_a_id"])
    op.create_index("ix_comparisons_run_b_id", "comparisons", ["run_b_id"])


def downgrade() -> None:
    op.drop_index("ix_comparisons_run_b_id", table_name="comparisons")
    op.drop_index("ix_comparisons_run_a_id", table_name="comparisons")
    op.drop_table("comparisons")
