from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import Settings
from .models import (
    PageAssessment,
    PageContent,
    assessment_to_dict,
    dict_to_assessment,
    dict_to_page,
    page_to_dict,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CachedPage:
    page: PageContent
    assessment: PageAssessment
    fetched_at: datetime


class PageCache:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache_dir = settings.output_dir / "_cache"
        self._ttl = timedelta(hours=settings.cache_ttl_hours)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self._cache_dir / f"{digest}.json"

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    async def load(self, url: str) -> CachedPage | None:
        path = self._cache_path(url)
        if not path.exists():
            return None

        try:
            raw_text = await asyncio.to_thread(path.read_text, encoding="utf-8")
            payload = json.loads(raw_text)
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            logger.warning("Discarding corrupt cache entry for %s", url)
            return None

        fetched_raw = payload.get("fetched_at")
        if not fetched_raw:
            return None

        fetched_at = datetime.fromisoformat(fetched_raw)
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)

        if self._now() - fetched_at > self._ttl:
            return None

        page_data = payload.get("page")
        assessment_data = payload.get("assessment")
        if not page_data or not assessment_data:
            return None

        page = dict_to_page(page_data)
        path_hint = page.markdown_path
        if path_hint and not path_hint.is_absolute():
            page.markdown_path = (self._settings.output_dir / path_hint).resolve()

        assessment = dict_to_assessment(assessment_data)
        return CachedPage(page=page, assessment=assessment, fetched_at=fetched_at)

    async def save(
        self, *, url: str, page: PageContent, assessment: PageAssessment
    ) -> None:
        page_dict = page_to_dict(page)
        markdown_path_raw = page_dict.get("markdown_path")
        path_obj: Path | None = None
        if isinstance(markdown_path_raw, Path):
            path_obj = markdown_path_raw
        elif isinstance(markdown_path_raw, str) and markdown_path_raw:
            path_obj = Path(markdown_path_raw)

        if path_obj is not None:
            try:
                relative = path_obj.relative_to(self._settings.output_dir)
                page_dict["markdown_path"] = str(relative)
            except ValueError:
                page_dict["markdown_path"] = str(path_obj)

        payload = {
            "url": url,
            "fetched_at": self._now().isoformat(),
            "page": page_dict,
            "assessment": assessment_to_dict(assessment),
        }

        path = self._cache_path(url)
        await asyncio.to_thread(
            path.write_text,
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


__all__ = ["CachedPage", "PageCache"]
