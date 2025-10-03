"""Tests for markdown storage utilities."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from scolar.config import Settings
from scolar.models import LinkInfo, PageContent
from scolar.storage import store_markdown


@pytest.mark.asyncio
async def test_store_markdown_generates_unique_slugs(tmp_path: Path) -> None:
    """Markdown files from duplicate titles should not overwrite one another."""

    settings = Settings(output_dir=tmp_path)

    page_one = PageContent(
        url="https://example.com/posts/first",
        title="Shared Title",
        markdown="First markdown",
        links=[LinkInfo(title="Next", url="https://example.com/next")],
        truncated=False,
    )
    page_two = PageContent(
        url="https://example.com/posts/second",
        title="Shared Title",
        markdown="Second markdown",
        links=[],
        truncated=False,
    )

    path_one = await store_markdown(page_one, settings)
    path_two = await store_markdown(page_two, settings)

    assert path_one != path_two
    assert path_one.exists()
    assert path_two.exists()
    assert path_one.read_text(encoding="utf-8") == "First markdown"
    assert path_two.read_text(encoding="utf-8") == "Second markdown"


@pytest.mark.asyncio
async def test_store_markdown_slug_includes_hash_suffix(tmp_path: Path) -> None:
    """Stored filenames should carry a deterministic hash suffix."""

    settings = Settings(output_dir=tmp_path)

    page = PageContent(
        url="https://sub.example.com/path/to/page",
        title="",
        markdown="Markdown body",
        links=[],
        truncated=False,
    )

    path = await store_markdown(page, settings)
    stem = path.stem

    assert re.fullmatch(r"[a-z0-9-]+-[0-9a-f]{8}", stem)


@pytest.mark.asyncio
async def test_store_markdown_slug_respects_length_limit(tmp_path: Path) -> None:
    """Generated slugs should not exceed the maximum configured length."""

    settings = Settings(output_dir=tmp_path)

    long_title = "Very Long Title " * 10
    page = PageContent(
        url="https://example.com/very/long/title",
        title=long_title,
        markdown="Body",
        links=[],
        truncated=False,
    )

    path = await store_markdown(page, settings)
    assert len(path.stem) <= 80


@pytest.mark.asyncio
async def test_store_markdown_idempotent_for_same_url(tmp_path: Path) -> None:
    """Calling store_markdown twice for the same URL should reuse the slug."""

    settings = Settings(output_dir=tmp_path)

    page = PageContent(
        url="https://repeat.example.com/item",
        title="Repeated",
        markdown="Snapshot",
        links=[],
        truncated=False,
    )

    first = await store_markdown(page, settings)
    page.markdown = "Updated"
    second = await store_markdown(page, settings)

    assert first == second
    assert second.read_text(encoding="utf-8") == "Updated"
