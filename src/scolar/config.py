from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from dynaconf import Dynaconf
from pydantic import BaseModel, ConfigDict, Field, field_validator


_dynaconf_settings = Dynaconf(
    envvar_prefix="SCOLAR",
    settings_files=["settings.toml"],
    load_dotenv=True,
    environments=True,
)


class Settings(BaseModel):
    fetch_concurrency: int = Field(default=5, ge=1)
    request_timeout: float = Field(default=15.0, ge=1.0)
    request_retries: int = Field(default=2, ge=0)
    request_backoff: float = Field(default=0.5, ge=0.0)
    user_agent: str = Field(
        default="Mozilla/5.0 (compatible; Scolar/0.1; +https://example.com)",
        min_length=1,
    )
    output_dir: Path = Field(default=Path("artifacts"))
    max_markdown_chars: int = Field(default=20_000, ge=1_000)
    max_links_inspected: int = Field(default=100, ge=1)
    max_recommended_links: int = Field(default=5, ge=0)
    openai_model: str = Field(default="gpt-4.1-mini", min_length=1)
    openai_temperature: float = Field(default=0.2, ge=0.0)
    openai_timeout: float = Field(default=30.0, ge=1.0)
    llm_concurrency: int = Field(default=1, ge=1)

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("output_dir", mode="before")
    @classmethod
    def _coerce_output_dir(cls, value: Any) -> Path:
        if value is None or value == "":
            return Path("artifacts").expanduser()
        return Path(value).expanduser()


def load_settings() -> Settings:
    raw: Dict[str, Any] = {}
    for field_name in Settings.model_fields:
        value = _dynaconf_settings.get(field_name, default=None)
        if value is not None:
            raw[field_name] = value
    return Settings(**raw)


__all__ = ["Settings", "load_settings", "_dynaconf_settings"]
