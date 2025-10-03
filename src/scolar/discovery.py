"""Candidate URL discovery utilities backed by subreddit search."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final, Protocol

import httpx

from .config import Settings

logger = logging.getLogger(__name__)

_REDDIT_SEARCH_ENDPOINT: Final[str] = "https://www.reddit.com/r/localllama/search.json"
_SEARCH_CACHE_DIRNAME: Final[str] = "search_hits"
_DEFAULT_SEARCH_TTL: Final[timedelta] = timedelta(days=3)
_DEFAULT_RESULT_LIMIT: Final[int] = 10


@dataclass(slots=True)
class SearchHit:
    prompt: str
    urls: list[str]
    fetched_at: datetime


class SupportsHttpResponse(Protocol):
    def raise_for_status(self) -> None: ...

    def json(self) -> object: ...


class SupportsAsyncGet(Protocol):
    async def get(
        self,
        url: str,
        *,
        params: object = None,
        timeout: object = None,
        **kwargs: object,
    ) -> SupportsHttpResponse: ...


class SearchHitCache:
    def __init__(self, settings: Settings, *, ttl: timedelta | None = None) -> None:
        self._settings = settings
        cache_root = settings.output_dir / "_cache" / _SEARCH_CACHE_DIRNAME
        cache_root.mkdir(parents=True, exist_ok=True)
        self._cache_dir = cache_root
        self._ttl = ttl if ttl is not None else _DEFAULT_SEARCH_TTL

    def _cache_path(self, prompt: str) -> Path:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return self._cache_dir / f"{digest}.json"

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    async def load(self, prompt: str) -> SearchHit | None:
        path = self._cache_path(prompt)
        if not path.exists():
            return None

        try:
            raw_text = await asyncio.to_thread(path.read_text, encoding="utf-8")
            payload = json.loads(raw_text)
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            logger.warning("Discarding corrupt search cache entry for %r", prompt)
            return None

        fetched_raw = payload.get("fetched_at")
        if not fetched_raw:
            return None

        fetched_at = datetime.fromisoformat(fetched_raw)
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)

        if self._now() - fetched_at > self._ttl:
            return None

        urls_raw = payload.get("urls")
        if not isinstance(urls_raw, list):
            return None

        urls: list[str] = []
        for value in urls_raw:
            if isinstance(value, str) and value.strip():
                urls.append(value.strip())

        if not urls:
            return None

        return SearchHit(prompt=prompt, urls=urls, fetched_at=fetched_at)

    async def save(self, *, prompt: str, urls: list[str]) -> None:
        payload = {
            "prompt": prompt,
            "urls": urls,
            "fetched_at": self._now().isoformat(),
        }
        path = self._cache_path(prompt)
        await asyncio.to_thread(
            path.write_text,
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _dedupe_urls(urls: list[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        normalized = url.strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(normalized)
        if len(result) >= limit:
            break
    return result


async def _search_reddit_localllama(
    prompt: str,
    *,
    http_client: httpx.AsyncClient,
    settings: Settings,
    limit: int,
) -> list[str]:
    if not prompt.strip():
        return []

    params = {
        "q": prompt,
        "restrict_sr": "1",
        "sort": "relevance",
        "limit": str(limit),
        "include_over_18": "on",
    }

    try:
        logger.info(
            "Searching r/localllama for prompt %r (limit=%d)",
            prompt,
            limit,
        )
        response = await http_client.get(
            _REDDIT_SEARCH_ENDPOINT,
            params=params,
            timeout=settings.request_timeout,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("Reddit search failed for %r: %s", prompt, exc)
        return []

    try:
        data = response.json()
    except json.JSONDecodeError:
        logger.error("Failed to decode Reddit search response for %r", prompt)
        return []

    children = data.get("data", {}).get("children") if isinstance(data, dict) else None
    if not isinstance(children, list):
        logger.warning("Unexpected Reddit search payload for %r", prompt)
        return []

    urls: list[str] = []
    for child in children:
        if not isinstance(child, dict):
            continue
        child_data = child.get("data")
        if not isinstance(child_data, dict):
            continue
        url_value = child_data.get("url")
        if isinstance(url_value, str):
            urls.append(url_value)
        permalink = child_data.get("permalink")
        if isinstance(permalink, str):
            urls.append(f"https://www.reddit.com{permalink}")

    deduped = _dedupe_urls(urls, limit=limit)
    logger.info("Reddit search yielded %d candidate urls", len(deduped))
    return deduped


async def discover_candidate_urls(
    prompt: str,
    *,
    http_client: httpx.AsyncClient,
    settings: Settings,
    refresh_cache: bool,
    cache: SearchHitCache | None = None,
) -> list[str]:
    limit = min(settings.max_links_inspected, _DEFAULT_RESULT_LIMIT)
    cache_obj = cache if cache is not None else SearchHitCache(settings)

    if not refresh_cache:
        cached = await cache_obj.load(prompt)
        if cached:
            logger.info(
                "Using cached Reddit search results for %r from %s",
                prompt,
                cached.fetched_at.isoformat(),
            )
            return _dedupe_urls(cached.urls, limit=limit)

    urls = await _search_reddit_localllama(
        prompt,
        http_client=http_client,
        settings=settings,
        limit=limit,
    )

    if urls:
        await cache_obj.save(prompt=prompt, urls=urls)

    return urls


__all__ = [
    "SearchHit",
    "SearchHitCache",
    "SupportsAsyncGet",
    "SupportsHttpResponse",
    "discover_candidate_urls",
]
