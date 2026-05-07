# Architecture

## Provider Protocol — why hermetic FakeProvider matters

Every LLM-touching code path goes through `ChatProvider`, a structural
`Protocol` with a single async `chat(...) -> ChatResult` method. Three
implementations ship in this repo:

| Provider          | Network | Used in CI? | Cost     |
|-------------------|---------|-------------|----------|
| `FakeProvider`    | No      | Yes         | $0       |
| `OpenAIProvider`  | Yes     | No          | BYOK     |
| `AnthropicProvider` | Yes   | No          | BYOK     |

For an evaluation system, hermetic execution is not a nicety — it is the
property that makes the eval itself testable. If CI ran against a live model,
the eval pipeline would never know whether a regression was caused by the
*evaluator* (a metric bug, a scoring threshold bug, a parsing bug) or the
*evaluated* (drift in the live model). So:

- `FakeProvider` is **deterministic**, keyed by SHA-256 of the prompt + model
  for unknown inputs and by a small scripted lookup table for tagged ones.
  Same prompt → same output, forever.
- The orchestrator embeds a tag of the form `<<TAG kind=task|task_type=…
  |example_id=…|lang_key=…>>` inside the user message. `FakeProvider` parses
  this tag and serves the corresponding canned response.
- Some entries in the scripted table are **deliberately wrong** so that
  per-cell pass rates land between 0 and 1. If the FakeProvider scored 100% on
  every example, the failure path of the scoring code would never get
  exercised — and a metric bug like "always returns 1.0" would pass CI silently.

Live-provider tests use `respx` to stub the HTTP layer at the wire level. We
never mock `ChatProvider` directly, so the real serializer / parser code is
exercised end-to-end.

## Task × language matrix design

Every task module exports the same three-function shape:

```python
def load_suite(language: str, suites_dir: Path) -> list[Example]: ...
def build_messages(ex: Example) -> list[ChatMessage]: ...
def score(ex: Example, output: str) -> dict[str, float]: ...
```

This contract is enforced structurally rather than via a base class. The
orchestrator dispatches via `TASK_MODULES[task_type]` and `TASK_LANGUAGES[task_type]`.

Languages are tracked as plain strings:

- `en`, `es`, `ja` — direct languages.
- `en-es`, `en-ja`, `es-en` — translation pair codes; the language for a
  translation example is the pair, not one side.
- `py` — code_repair is language-agnostic for the user-facing language axis,
  but the test runner is Python-specific, so we keep `py` as the language code
  for that one cell.

Per-task scoring threshold choices are documented next to each module:

| Task           | Metric           | Pass threshold |
|----------------|------------------|----------------|
| summarization  | ROUGE-L F1       | 0.30           |
| translation    | chrF (β=2)       | 0.40           |
| qa             | token-F1         | 0.50           |
| classification | exact-match      | 1.0 (binary)   |
| code_repair    | pytest pass/fail | 1.0 (binary)   |

## Localization-quality stack

Three independent checks per non-English output, because each catches a
different failure mode:

1. **Script** (`localization/script_check.py`). Pure Unicode-block math;
   deterministic, fast, no LLM call. Catches gross failures like "Japanese
   request got romaji response". Returns a ratio of in-script alphabetic
   characters; the binary `pass` is `ratio >= 0.9`.
2. **Honorific appropriateness** (`localization/honorific_judge.py`). LLM-as-judge,
   `ja` only. Rubric committed at `eval/rubrics/honorific_ja.md`. Catches
   register failures (excessive 敬語 in casual contexts, plain form where
   polite expected) that no script check ever could.
3. **Calque-free** (`localization/calque_judge.py`). LLM-as-judge, `es` only.
   Rubric committed at `eval/rubrics/calque_es.md`. Catches "translated from
   English word-for-word" failures: technically correct vocabulary,
   structurally English.

These run on top of the raw task metric, not in place of it. A Spanish
translation can score 0.95 chrF and still get a low calque score (the gold
itself was natural; the candidate is a calque that happens to share characters).

## Metric choices

- **ROUGE-L** for summarization. F1 over the longest common subsequence of
  whitespace-tokenised lowercased outputs. Picked because it tolerates
  paraphrase better than n-gram BLEU. Caveat: whitespace tokenisation breaks
  for CJK; `summarization/ja` is therefore a 0/1 metric (full-string match or
  nothing). Documented as a known limitation rather than papered over.
- **chrF** for translation. Character n-gram F-score over n=1..6, β=2 (the
  paper's default). Language-agnostic by construction — no tokeniser, no word
  segmentation. Strictly preferred over BLEU at sentence scale and for CJK.
- **Token-F1 + exact-match** for QA. SQuAD lineage; F1 picks up partial
  credit, exact-match for headline reporting.
- **pytest pass/fail** for code repair. Binary. The candidate fix is dropped
  into a temp file, the failing test is appended, and `python -I run.py` is
  spawned with a 5-second timeout. Pure outcome metric — no need for surface
  similarity to a reference fix.

## Regression-flag heuristic

For each (model, task, language) cell, the dashboard computes a rolling
7-run mean of `pass_rate`. Any run whose `pass_rate` is more than 5 percentage
points below that mean is flagged.

```
flag(run_n) = pass_rate(run_n) < (rolling_mean(run_n-7..run_n-1) - 0.05)
```

This is intentionally not a statistical significance test. With 3-9 examples
per cell, a confidence interval would be too wide to fire useful alarms. The
flag is a coarse trip-wire: "something changed; look at this run". The
expected workflow is: see the flag → click into the run → look at the items
table → identify the example(s) that flipped from pass to fail.

## Persistence

SQLite via aiosqlite. Three tables: `model_versions`, `runs`, `run_items`.
SQLite is the right call here because the workload is study-scale (an order of
magnitude under 100k items per day) and the dashboard is the only consumer.
Postgres adds operational cost without buying anything for this scope. If a
real team adopted this, they'd swap the URL via `GENAI_EVAL_DATABASE_URL` and
the schema would carry over with one alembic migration.

JSONB-style fields (`suite_filter`, `summary`, `scores`) are stored as TEXT
with property-side JSON encoding — kept in TEXT so that downgrading the
storage to a different backend (or running the schema on Postgres without
JSONB) doesn't fork the migration story. The cost is loss of server-side JSON
querying for these columns; for the access patterns we have, that doesn't
matter.

## Concurrent execution + failure handling

The orchestrator caps concurrent provider calls at 8 (`asyncio.Semaphore(8)`).
Per-example failures are caught individually and persisted as
`status='error'` with the exception text — the run continues. This is the
right default for an eval system: a single flaky API call shouldn't invalidate
the other 38 results. The summary reports `n_errors`, so reviewers know how
many cells they're looking at conditional on success.

## What's deliberately not here

- **No fine-tuning loop.** This service evaluates models; it does not train
  them. Adding training would dilute the contract.
- **No inference-cost optimization.** We report `cost_usd` per item from the
  provider's reported usage, but we don't try to minimize it. A separate tool
  belongs upstream of this one.
- **No human-in-the-loop UI.** The dashboard is read-only. Adding annotation
  workflows would expand scope into labeling-tool territory.
- **No public leaderboard.** The dashboard is local; the runs table is yours.
  A leaderboard would invite gaming and require de-duplication infrastructure
  that has nothing to do with evaluation.
- **No model-serving gateway.** Providers are a thin shim; we never proxy.
