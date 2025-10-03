from __future__ import annotations

import asyncio
import re
from pathlib import Path

from .config import Settings
from .models import PageContent


_slug_cleanup = re.compile(r"[^a-z0-9-]+")


def _slugify(title: str) -> str:
    base = title.lower().strip().replace(" ", "-")
    base = _slug_cleanup.sub("", base)
    return base or "page"


def _write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


async def store_markdown(page: PageContent, settings: Settings) -> Path:
    slug = _slugify(page.title)[:80]
    path = settings.output_dir / f"{slug}.md"
    await asyncio.to_thread(_write_markdown, path, page.markdown)
    return path


__all__ = ["store_markdown"]
