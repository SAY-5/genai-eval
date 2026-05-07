FROM python:3.12-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN pip install --no-cache-dir "poetry==1.8.4"

COPY pyproject.toml poetry.lock* ./
RUN poetry install --only=main --no-root --no-interaction

COPY src ./src
COPY alembic.ini ./alembic.ini
COPY alembic ./alembic
COPY eval ./eval
RUN poetry install --only=main --no-interaction

EXPOSE 8000
CMD ["uvicorn", "genai_eval.api:app", "--host", "0.0.0.0", "--port", "8000"]
