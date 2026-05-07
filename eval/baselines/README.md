# Baselines

Each `*.json` in this directory is the committed summary of a full eval run
against the FakeProvider. The CI `eval-smoke` job exercises the same code path
against the same suites; if its summary diverges materially from the committed
baseline, the build fails.

## Conventions

- Filename matches the model id (e.g. `fake-large.json`).
- `produced_at` is the wall-clock time at which the baseline was generated.
- The committed FakeProvider deliberately produces some wrong outputs so the
  numbers below are not 100% — that is by design, so the failure path of the
  scoring code is exercised on every CI run.

## Live (BYOK) baselines

Live baselines from `openai` or `anthropic` providers are not committed here.
To produce one locally:

```bash
export OPENAI_API_KEY=...   # or ANTHROPIC_API_KEY
poetry run genai-eval run --provider openai --model gpt-4o-mini \
  --output eval/baselines/gpt-4o-mini.json
```

The README pass-rate table is sourced from `fake-large.json` only.
