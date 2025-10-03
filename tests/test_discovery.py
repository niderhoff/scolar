from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from scolar.config import Settings
from scolar.discovery import SearchHitCache, discover_candidate_urls


class _FailingClient:
    async def get(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("HTTP client should not be invoked when cache is valid")


class _DummyResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _RecordingClient:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.calls = 0

    async def get(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        self.calls += 1
        return _DummyResponse(self._payload)


@pytest.mark.asyncio
async def test_discover_candidate_urls_uses_cache(tmp_path: Path) -> None:
    settings = Settings(output_dir=tmp_path)
    cache = SearchHitCache(settings, ttl=timedelta(days=3))
    await cache.save(
        prompt="llm agents",
        urls=["https://www.reddit.com/r/localllama/comments/abc/agent_thread/"],
    )

    result = await discover_candidate_urls(
        "llm agents",
        http_client=_FailingClient(),
        settings=settings,
        refresh_cache=False,
        cache=cache,
    )

    assert result == [
        "https://www.reddit.com/r/localllama/comments/abc/agent_thread/"
    ]


@pytest.mark.asyncio
async def test_discover_candidate_urls_fetches_and_caches(tmp_path: Path) -> None:
    settings = Settings(output_dir=tmp_path, max_links_inspected=2)
    payload = {
        "data": {
            "children": [
                {
                    "data": {
                        "url": "https://example.com/discussion",
                        "permalink": "/r/localllama/comments/xyz/discussion/",
                    }
                },
                {
                    "data": {
                        "url": "https://example.com/discussion",
                        "permalink": "/r/localllama/comments/xyz/discussion/",
                    }
                },
            ]
        }
    }
    client = _RecordingClient(payload)
    cache = SearchHitCache(settings, ttl=timedelta(days=3))

    first_result = await discover_candidate_urls(
        "prompt",
        http_client=client,
        settings=settings,
        refresh_cache=False,
        cache=cache,
    )

    assert client.calls == 1
    assert first_result == [
        "https://example.com/discussion",
        "https://www.reddit.com/r/localllama/comments/xyz/discussion/",
    ]

    cached_result = await discover_candidate_urls(
        "prompt",
        http_client=_FailingClient(),
        settings=settings,
        refresh_cache=False,
        cache=cache,
    )

    assert cached_result == first_result
