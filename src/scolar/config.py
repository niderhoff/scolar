from __future__ import annotations

from pathlib import Path
from typing import cast

from dynaconf import Dynaconf
from pydantic import BaseModel, ConfigDict, Field, field_validator

_dynaconf_settings = Dynaconf(
    envvar_prefix="SCOLAR",
    settings_files=["settings.toml"],
    load_dotenv=True,
    environments=True,
)


SettingValue = int | float | str | Path


class Settings(BaseModel):
    """Runtime configuration for the Scolar orchestrator."""

    fetch_concurrency: int = Field(
        default=5,
        ge=1,
        description="Maximum number of HTTP fetch tasks executed concurrently.",
    )
    request_timeout: float = Field(
        default=15.0,
        ge=1.0,
        description="Seconds before an outbound HTTP request is aborted.",
    )
    request_retries: int = Field(
        default=2,
        ge=0,
        description="Number of retry attempts for failed HTTP requests.",
    )
    request_backoff: float = Field(
        default=0.5,
        ge=0.0,
        description="Base exponential backoff interval (seconds) between retries.",
    )
    user_agent: str = Field(
        default="Mozilla/5.0 (compatible; Scolar/0.1; +https://example.com)",
        min_length=1,
        description="User-Agent header applied to every outbound HTTP request.",
    )
    output_dir: Path = Field(
        default=Path("artifacts"),
        description="Filesystem directory where generated artifacts are saved.",
    )
    max_markdown_chars: int = Field(
        default=20_000,
        ge=1_000,
        description="Upper bound on characters retained from fetched Markdown content.",
    )
    max_links_inspected: int = Field(
        default=10,
        ge=1,
        description="Cap on candidate links evaluated during discovery.",
    )
    max_recommended_links: int = Field(
        default=5,
        ge=0,
        description="Maximum number of links surfaced in the final recommendations.",
    )
    openai_model: str = Field(
        default="gpt-4.1-mini",
        min_length=1,
        description="Identifier of the OpenAI chat completion model in use.",
    )
    openai_temperature: float = Field(
        default=0.2,
        ge=0.0,
        description="Sampling temperature applied to OpenAI completions.",
    )
    openai_timeout: float = Field(
        default=30.0,
        ge=1.0,
        description="Seconds allowed for each OpenAI API request before timing out.",
    )
    llm_concurrency: int = Field(
        default=1,
        ge=1,
        description="Maximum number of simultaneous OpenAI requests issued by the pipeline.",
    )
    final_answer_max_pages: int = Field(
        default=5,
        ge=1,
        description="Highest page count permitted in the compiled final answer PDF.",
    )
    final_answer_excerpt_chars: int = Field(
        default=1_500,
        ge=200,
        description="Character budget allocated for each excerpt embedded in the final answer.",
    )
    cache_ttl_hours: int = Field(
        default=72,
        ge=1,
        description="Duration in hours that cached fetch results remain valid.",
    )

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("output_dir", mode="before")
    @classmethod
    def _coerce_output_dir(cls, value: str | Path | None) -> Path:
        if value is None or value == "":
            return Path("artifacts").expanduser()
        return Path(value).expanduser()


def load_settings() -> Settings:
    raw: dict[str, SettingValue] = {}
    for field_name in Settings.model_fields:
        value = _dynaconf_settings.get(field_name, default=None)
        if value is not None:
            raw[field_name] = cast(SettingValue, value)
    return Settings.model_validate(raw)


__all__ = ["Settings", "load_settings", "_dynaconf_settings"]
