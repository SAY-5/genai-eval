"""Bench-regress gate: compare a fresh eval result against a committed baseline.

Fails (exit 1) if any cell's pass-rate drifts from the baseline by more than
``--max-drift`` (absolute, e.g. 0.30 = 30 percentage points). Also fails if a
baseline cell disappears or a new cell appears, since those almost always
indicate a suite change that should be intentional.

Usage:
    python scripts/bench_regress.py --fresh fresh.json --baseline base.json --max-drift 0.30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_cells(path: Path) -> dict[tuple[str, str], float]:
    blob: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    cells = blob["summary"]["cells"]
    return {(c["task"], c["language"]): float(c["pass_rate"]) for c in cells}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fresh", type=Path, required=True, help="newly produced eval JSON")
    p.add_argument("--baseline", type=Path, required=True, help="committed baseline eval JSON")
    p.add_argument(
        "--max-drift",
        type=float,
        default=0.30,
        help="maximum allowed |fresh - baseline| per cell (default: 0.30)",
    )
    args = p.parse_args()

    fresh = _load_cells(args.fresh)
    base = _load_cells(args.baseline)

    fresh_keys = set(fresh)
    base_keys = set(base)
    failures: list[str] = []

    missing = base_keys - fresh_keys
    extra = fresh_keys - base_keys
    if missing:
        failures.append(f"missing cells in fresh: {sorted(missing)}")
    if extra:
        failures.append(f"unexpected cells in fresh: {sorted(extra)}")

    rows: list[tuple[str, str, float, float, float]] = []
    for key in sorted(base_keys & fresh_keys):
        f = fresh[key]
        b = base[key]
        delta = f - b
        rows.append((key[0], key[1], f, b, delta))
        if abs(delta) > args.max_drift:
            failures.append(
                f"{key[0]}/{key[1]}: |{f:.4f} - {b:.4f}| = {abs(delta):.4f} > {args.max_drift}"
            )

    print(f"{'task':<15} {'language':<8} {'fresh':>8} {'baseline':>10} {'delta':>9}")
    for task, lang, f, b, delta in rows:
        marker = "  OK" if abs(delta) <= args.max_drift else " FAIL"
        print(f"{task:<15} {lang:<8} {f:>8.4f} {b:>10.4f} {delta:>+9.4f}{marker}")

    if failures:
        print("\nbench-regress: FAIL", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        return 1
    print("\nbench-regress: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
