from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse

from .config import Settings
from .models import PageContent

_MAX_SLUG_LENGTH = 80
_HASH_LENGTH = 8
_slug_cleanup = re.compile(r"[^a-z0-9-]+")


def _slugify(text: str) -> str:
    base = text.lower().strip().replace(" ", "-")
    return _slug_cleanup.sub("", base)


def _base_slug(page: PageContent) -> str:
    title_slug = _slugify(page.title)
    if title_slug:
        return title_slug

    parsed_url = urlparse(page.url)
    url_parts = [parsed_url.netloc, parsed_url.path]
    for part in url_parts:
        candidate = _slugify(part)
        if candidate:
            return candidate

    return "page"


def _build_slug(page: PageContent) -> str:
    base = _base_slug(page)

    hash_suffix = hashlib.sha256(page.url.encode("utf-8")).hexdigest()[:_HASH_LENGTH]
    max_base_length = _MAX_SLUG_LENGTH - _HASH_LENGTH - 1

    trimmed_base = base[:max_base_length].rstrip("-")
    if not trimmed_base:
        trimmed_base = "page"

    return f"{trimmed_base}-{hash_suffix}"


def _write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


async def store_markdown(page: PageContent, settings: Settings) -> Path:
    slug = _build_slug(page)
    path = settings.output_dir / f"{slug}.md"
    await asyncio.to_thread(_write_markdown, path, page.markdown)
    return path


__all__ = ["store_markdown"]
