"""Tests for scolar.config ensuring Dynaconf + Pydantic configuration behaves correctly."""

from __future__ import annotations

import importlib
from pathlib import Path
from textwrap import dedent

import pytest
from dynaconf import Dynaconf
from pydantic import ValidationError

from scolar import config


def _patch_dynaconf(monkeypatch: pytest.MonkeyPatch, *files: Path) -> None:
    """Replace the module-level Dynaconf instance with one bound to the provided files."""

    settings_files = [str(path) for path in files]
    dynaconf_instance = Dynaconf(
        envvar_prefix="SCOLAR",
        settings_files=settings_files,
        environments=True,
        load_dotenv=False,
    )
    monkeypatch.setattr(config, "_dynaconf_settings", dynaconf_instance)


def test_load_settings_from_single_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Settings should deserialize cleanly from a TOML file without environment overrides."""

    settings_file = tmp_path / "settings.toml"
    settings_file.write_text(
        dedent(
            """
            [default]
            fetch_concurrency = 7
            request_timeout = 22.5
            request_retries = 3
            request_backoff = 0.75
            user_agent = "TestAgent/1.0"
            output_dir = "artifacts/output"
            max_markdown_chars = 12345
            max_links_inspected = 42
            max_recommended_links = 2
            openai_model = "gpt-test"
            openai_temperature = 0.55
            openai_timeout = 45.0
            llm_concurrency = 2
            """
        ).strip(),
        encoding="utf-8",
    )

    _patch_dynaconf(monkeypatch, settings_file)
    settings = config.load_settings()

    assert settings.fetch_concurrency == 7
    assert settings.request_timeout == pytest.approx(22.5)
    assert settings.request_retries == 3
    assert settings.request_backoff == pytest.approx(0.75)
    assert settings.user_agent == "TestAgent/1.0"
    assert settings.output_dir == Path("artifacts/output")
    assert settings.max_markdown_chars == 12345
    assert settings.max_links_inspected == 42
    assert settings.max_recommended_links == 2
    assert settings.openai_model == "gpt-test"
    assert settings.openai_temperature == pytest.approx(0.55)
    assert settings.openai_timeout == pytest.approx(45.0)
    assert settings.llm_concurrency == 2


def test_environment_overrides_take_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Environment variables with the SCOLAR_ prefix should override TOML values."""

    settings_file = tmp_path / "settings.toml"
    settings_file.write_text(
        """
        [default]
        fetch_concurrency = 3
        openai_temperature = 0.1
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("SCOLAR_FETCH_CONCURRENCY", "11")
    monkeypatch.setenv("SCOLAR_OPENAI_TEMPERATURE", "0.9")
    _patch_dynaconf(monkeypatch, settings_file)

    settings = config.load_settings()

    assert settings.fetch_concurrency == 11
    assert settings.openai_temperature == pytest.approx(0.9)


def test_invalid_values_raise_validation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid numeric ranges should trigger Pydantic validation errors."""

    bad_settings = tmp_path / "settings.toml"
    bad_settings.write_text(
        """
        [default]
        fetch_concurrency = 0
        """.strip(),
        encoding="utf-8",
    )

    _patch_dynaconf(monkeypatch, bad_settings)

    with pytest.raises(ValidationError):
        config.load_settings()


def test_multiple_settings_files_override_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Later settings files should override earlier ones while preserving untouched keys."""

    base = tmp_path / "base.toml"
    override = tmp_path / "override.toml"

    base.write_text(
        """
        [default]
        fetch_concurrency = 4
        user_agent = "BaseAgent"
        """.strip(),
        encoding="utf-8",
    )
    override.write_text(
        """
        [default]
        fetch_concurrency = 8
        """.strip(),
        encoding="utf-8",
    )

    _patch_dynaconf(monkeypatch, base, override)
    settings = config.load_settings()

    assert settings.fetch_concurrency == 8
    assert settings.user_agent == "BaseAgent"


def test_output_dir_expands_user_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The configured output directory should expand user home references."""

    file = tmp_path / "settings.toml"
    file.write_text(
        """
        [default]
        output_dir = "~/custom/artifacts"
        """.strip(),
        encoding="utf-8",
    )

    _patch_dynaconf(monkeypatch, file)
    settings = config.load_settings()

    assert settings.output_dir == Path("~/custom/artifacts").expanduser()


def test_non_prefixed_environment_variables_are_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Environment variables without the SCOLAR_ prefix must not override settings."""

    file = tmp_path / "settings.toml"
    file.write_text(
        """
        [default]
        fetch_concurrency = 5
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("FETCH_CONCURRENCY", "99")
    _patch_dynaconf(monkeypatch, file)

    settings = config.load_settings()
    assert settings.fetch_concurrency == 5


def test_merging_files_and_environment_priority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Priority order should be defaults < first file < second file < environment variables."""

    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"

    first.write_text(
        """
        [default]
        fetch_concurrency = 2
        user_agent = "FirstAgent"
        """.strip(),
        encoding="utf-8",
    )
    second.write_text(
        """
        [default]
        fetch_concurrency = 6
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("SCOLAR_FETCH_CONCURRENCY", "12")
    _patch_dynaconf(monkeypatch, first, second)

    settings = config.load_settings()

    assert settings.fetch_concurrency == 12
    assert settings.user_agent == "FirstAgent"


def test_custom_settings_file_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pointing Dynaconf at an alternate settings file should control the resulting configuration."""

    custom_dir = tmp_path / "alt_config"
    custom_dir.mkdir()
    custom_file = custom_dir / "custom.toml"
    custom_file.write_text(
        """
        [default]
        llm_concurrency = 4
        """.strip(),
        encoding="utf-8",
    )

    _patch_dynaconf(monkeypatch, custom_file)
    settings = config.load_settings()

    assert settings.llm_concurrency == 4


@pytest.fixture(autouse=True)
def _restore_config_module():
    """Reload the config module after each test to reset Dynaconf state."""

    yield
    importlib.reload(config)
