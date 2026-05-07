"""Code repair task — fix a buggy Python snippet so a pytest case passes.

Execution is sandboxed in a subprocess so a bad snippet can't poison the parent
process. We pass `-I` (isolated) and a short timeout.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from genai_eval.providers import ChatMessage
from genai_eval.tasks import Example


def load_suite(language: str, suites_dir: Path) -> list[Example]:
    """language is unused for code_repair; we always load python.yaml."""
    del language
    path = suites_dir / "code_repair" / "python.yaml"
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        Example(
            id=item["id"],
            task_type="code_repair",
            language="py",
            input={"buggy_code": item["buggy_code"], "test_code": item["test_code"]},
            gold={"reference_fix": item.get("reference_fix", "")},
            metadata=item.get("metadata", {}),
        )
        for item in raw["examples"]
    ]


def build_messages(ex: Example) -> list[ChatMessage]:
    tag = f"<<TAG kind=task|task_type=code_repair|example_id={ex.id}|lang_key=py>>"
    sys = (
        "You are a Python repair tool. Output only the corrected code (a complete "
        "module). Do not include explanations or markdown fences."
    )
    user = (
        f"{tag}\n"
        f"Buggy code:\n{ex.input['buggy_code']}\n\n"
        f"Failing test:\n{ex.input['test_code']}\n"
    )
    return [ChatMessage("system", sys), ChatMessage("user", user)]


def score(ex: Example, output: str) -> dict[str, float]:
    """Run the candidate fix against the failing test in a subprocess."""
    passed = _run_test(output, ex.input["test_code"])
    return {"pass": 1.0 if passed else 0.0}


def _run_test(candidate_module: str, test_code: str) -> bool:
    """Execute the candidate module and run the test; True iff test passes."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        (tmpdir / "candidate.py").write_text(candidate_module, encoding="utf-8")
        # Build the runner without leading indentation so test_code lines (which
        # come from YAML, no indent) line up with module-level code.
        runner_lines = [
            "import sys",
            f"sys.path.insert(0, {str(tmpdir)!r})",
            "from candidate import *  # noqa: F401,F403",
            test_code.rstrip("\n"),
            'print("OK")',
            "",
        ]
        runner_path = tmpdir / "run.py"
        runner_path.write_text("\n".join(runner_lines), encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(runner_path)],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except subprocess.TimeoutExpired:
            return False
        return proc.returncode == 0 and "OK" in proc.stdout


__all__ = ["load_suite", "build_messages", "score"]
