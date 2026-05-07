.PHONY: help install dev migrate seed test test-int lint typecheck eval eval-smoke dashboard-dev dashboard-build up down clean

help:
	@echo "Targets:"
	@echo "  install         - poetry install"
	@echo "  dev             - run API on :8000 with reload"
	@echo "  migrate         - apply alembic migrations"
	@echo "  seed            - seed a fake-large model version row"
	@echo "  test            - pytest unit suite"
	@echo "  test-int        - pytest with RUN_INTEGRATION=1"
	@echo "  lint            - ruff + black --check"
	@echo "  typecheck       - mypy strict"
	@echo "  eval            - run full eval against FakeProvider, write baseline"
	@echo "  eval-smoke      - quick smoke run (subset)"
	@echo "  dashboard-dev   - next dev"
	@echo "  dashboard-build - next build"
	@echo "  up              - docker compose up"

install:
	poetry install

dev:
	poetry run uvicorn genai_eval.api:app --reload --host 0.0.0.0 --port 8000

migrate:
	poetry run alembic upgrade head

seed:
	poetry run python -m genai_eval.cli seed

test:
	poetry run pytest tests/unit -q

test-int:
	RUN_INTEGRATION=1 poetry run pytest tests/integration -q

lint:
	poetry run ruff check src tests
	poetry run black --check src tests

format:
	poetry run ruff check --fix src tests
	poetry run black src tests

typecheck:
	poetry run mypy

eval:
	poetry run genai-eval run --suite all --provider fake --model fake-large --output eval/baselines/fake-large.json

eval-smoke:
	poetry run genai-eval run --suite all --provider fake --model fake-large --output /tmp/eval-smoke.json --smoke

dashboard-dev:
	cd dashboard && pnpm dev

dashboard-build:
	cd dashboard && pnpm build

up:
	docker compose up --build

down:
	docker compose down

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build
	find . -type d -name __pycache__ -exec rm -rf {} +
