"""Settings loaded from environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, all overridable via env vars."""

    model_config = SettingsConfigDict(env_prefix="GENAI_EVAL_", env_file=".env", extra="ignore")

    database_url: str = Field(default="sqlite+aiosqlite:///./genai_eval.db")
    database_url_sync: str = Field(default="sqlite:///./genai_eval.db")
    suites_dir: str = Field(default="eval/suites")
    rubrics_dir: str = Field(default="eval/rubrics")
    max_concurrent_calls: int = Field(default=8)
    judge_provider: str = Field(default="fake")
    judge_model: str = Field(default="fake-large")
    log_level: str = Field(default="INFO")


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
