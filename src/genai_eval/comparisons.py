"""A/B comparison harness for two eval runs.

Computes per-cell deltas in pass-rate between two runs (model A vs model B),
plus a Welch-style two-proportion z-test p-value so a reviewer can tell
significant gaps from noise on small per-cell sample sizes.

This module is consumed by:
  * the ``genai-eval compare`` CLI (writes Markdown + persists to DB);
  * unit tests that pin the deterministic FakeProvider output.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from genai_eval.models import Base, Run


class Comparison(Base):
    """Persisted A/B comparison summary between two runs."""

    __tablename__ = "comparisons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_a_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), nullable=False)
    run_b_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), nullable=False)
    suite: Mapped[str] = mapped_column(String(64), nullable=False, default="all")
    summary_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    produced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    run_a: Mapped[Run] = relationship("Run", foreign_keys=[run_a_id])
    run_b: Mapped[Run] = relationship("Run", foreign_keys=[run_b_id])

    @property
    def summary(self) -> dict[str, Any]:
        try:
            v = json.loads(self.summary_json) if self.summary_json else {}
        except json.JSONDecodeError:
            return {}
        return v if isinstance(v, dict) else {}

    @summary.setter
    def summary(self, value: dict[str, Any]) -> None:
        self.summary_json = json.dumps(value, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Statistical helpers.
# ---------------------------------------------------------------------------


def _normal_cdf(z: float) -> float:
    """Standard-normal cumulative density. Pure math, no scipy dependency."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_proportion_pvalue(successes_a: int, n_a: int, successes_b: int, n_b: int) -> float:
    """Two-sided p-value for H0: p_a == p_b (Welch-style for proportions).

    Uses the unpooled-variance test (sometimes called Welch's adaptation for
    proportions): treats each sample's success rate as a Bernoulli mean and
    pools variance as ``var_a/n_a + var_b/n_b``. This matches the prompt's
    "Welch's two-proportion test" requirement and avoids scipy.

    Edge cases:
      * If either sample is empty, returns 1.0 (no signal).
      * If both rates are 0 or both are 1, returns 1.0.
    """
    if n_a <= 0 or n_b <= 0:
        return 1.0
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    if p_a == p_b:
        return 1.0
    var_a = p_a * (1.0 - p_a) / n_a
    var_b = p_b * (1.0 - p_b) / n_b
    se = math.sqrt(var_a + var_b)
    if se == 0.0:
        return 1.0
    z = (p_a - p_b) / se
    # Two-sided.
    return 2.0 * (1.0 - _normal_cdf(abs(z)))


# ---------------------------------------------------------------------------
# Per-cell comparison plumbing.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellCompare:
    task: str
    language: str
    n_a: int
    n_b: int
    a_rate: float
    b_rate: float
    delta: float
    p_value: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "language": self.language,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "a_rate": self.a_rate,
            "b_rate": self.b_rate,
            "delta": self.delta,
            "p_value": self.p_value,
        }


def compare_cells(summary_a: dict[str, Any], summary_b: dict[str, Any]) -> list[CellCompare]:
    """Compute per-(task, language) delta + p-value for two run summaries."""
    cells_a = {(c["task"], c["language"]): c for c in summary_a.get("cells", [])}
    cells_b = {(c["task"], c["language"]): c for c in summary_b.get("cells", [])}
    keys = sorted(set(cells_a) | set(cells_b))
    out: list[CellCompare] = []
    for key in keys:
        ca = cells_a.get(key)
        cb = cells_b.get(key)
        n_a = int(ca["n"]) if ca else 0
        n_b = int(cb["n"]) if cb else 0
        a_rate = float(ca["pass_rate"]) if ca else 0.0
        b_rate = float(cb["pass_rate"]) if cb else 0.0
        # Recover integer success counts from rate * n; round to nearest int.
        succ_a = int(round(a_rate * n_a)) if ca else 0
        succ_b = int(round(b_rate * n_b)) if cb else 0
        p = two_proportion_pvalue(succ_a, n_a, succ_b, n_b)
        out.append(
            CellCompare(
                task=key[0],
                language=key[1],
                n_a=n_a,
                n_b=n_b,
                a_rate=a_rate,
                b_rate=b_rate,
                delta=b_rate - a_rate,
                p_value=p,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Markdown rendering.
# ---------------------------------------------------------------------------


def _fmt_pct(x: float) -> str:
    return f"{x * 100:6.2f}%"


def _fmt_delta(x: float) -> str:
    return f"{x * 100:+6.2f}%"


def _fmt_p(x: float) -> str:
    if x < 1e-4:
        return "<1e-4"
    return f"{x:6.4f}"


def render_markdown(
    *,
    model_a: str,
    model_b: str,
    suite: str,
    cells: list[CellCompare],
    overall_a: float,
    overall_b: float,
) -> str:
    """Render a deterministic Markdown report for two runs.

    Determinism: cells are sorted by (task, language); all numeric formatting
    uses fixed widths and no timestamps appear in the body, so a re-run of the
    same FakeProvider against the same suite produces byte-identical output.
    """
    lines: list[str] = []
    lines.append(f"# A/B comparison: `{model_a}` vs `{model_b}`")
    lines.append("")
    lines.append(f"- suite: `{suite}`")
    lines.append(f"- model A: `{model_a}`")
    lines.append(f"- model B: `{model_b}`")
    lines.append(f"- overall A: {_fmt_pct(overall_a)}")
    lines.append(f"- overall B: {_fmt_pct(overall_b)}")
    lines.append(f"- overall delta (B - A): {_fmt_delta(overall_b - overall_a)}")
    lines.append("")
    lines.append("## Per-cell results")
    lines.append("")
    lines.append("| task | language | n_a | n_b | A | B | delta | p-value |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for c in cells:
        lines.append(
            f"| {c.task} | {c.language} | {c.n_a} | {c.n_b} | {_fmt_pct(c.a_rate)} | {_fmt_pct(c.b_rate)} | {_fmt_delta(c.delta)} | {_fmt_p(c.p_value)} |"
        )
    lines.append("")
    lines.append("Significance threshold conventionally p < 0.05.")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "CellCompare",
    "Comparison",
    "compare_cells",
    "render_markdown",
    "two_proportion_pvalue",
]
