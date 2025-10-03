"""Integration tests for the async pipeline using mocked HTTP and LLM clients."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

import httpx
from openai import AsyncOpenAI

from scolar.cache import PageCache
from scolar.config import Settings
from scolar.fetcher import RedditComment, RedditThread
from scolar.models import (
    LinkInfo,
    PageAssessment,
    PageContent,
    RecommendedLink,
    Score,
)
from scolar.pipeline import gather_pages


@dataclass
class _FakeResponse:
    """Minimal stand-in for httpx.Response used in fetcher."""

    text: str
    headers: dict[str, str]

    def raise_for_status(self) -> None:
        return None


class _FakeHTTPClient:
    """Async HTTP client that returns pre-baked HTML payloads keyed by URL."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping
        self.requested: list[str] = []
        self.sent_headers: list[dict[str, str] | None] = []

    async def get(
        self,
        url: str,
        timeout: float,
        *,
        headers: dict[str, str] | None = None,
    ) -> _FakeResponse:
        self.requested.append(url)
        self.sent_headers.append(headers)
        html = self._mapping[url]
        return _FakeResponse(text=html, headers={"content-type": "text/html"})


@dataclass
class _FakeLLMResponse:
    output_text: str


class _FakeLLMResponses:
    def __init__(self, outputs: Iterable[str]) -> None:
        self._outputs = list(outputs)
        self.calls = 0

    async def create(self, **_kwargs) -> _FakeLLMResponse:
        output = self._outputs[self.calls]
        self.calls += 1
        return _FakeLLMResponse(output_text=output)


class _FakeLLMClient:
    """AsyncOpenAI replacement returning canned JSON payloads."""

    def __init__(self, outputs: Iterable[str]) -> None:
        self.responses = _FakeLLMResponses(outputs)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _make_settings(output_dir: Path) -> Settings:
    return Settings(
        fetch_concurrency=3,
        request_timeout=10.0,
        request_retries=0,
        request_backoff=0.0,
        user_agent="TestAgent",
        output_dir=output_dir,
        max_markdown_chars=10_000,
        max_links_inspected=10,
        max_recommended_links=3,
        openai_model="mock-model",
        openai_temperature=0.0,
        openai_timeout=10.0,
        llm_concurrency=2,
        final_answer_max_pages=5,
        final_answer_excerpt_chars=1_500,
        cache_ttl_hours=72,
    )


@pytest.mark.asyncio
async def test_gather_pages_happy_path(tmp_path: Path) -> None:
    """The pipeline should persist markdown and return a populated assessment."""

    url = "https://example.com/article"
    html = """
    <html><head><title>Example Title</title></head>
    <body><p>Sample content paragraph.</p><a href='https://example.com/next'>Next</a></body>
    </html>
    """
    fake_http_client = _FakeHTTPClient({url: html})

    llm_output = {
        "summary": "Concise summary of the page.",
        "technical_depth": {
            "rating": 4,
            "justification": "Covers implementation details.",
        },
        "prompt_fit": {"rating": 5, "justification": "Directly answers the prompt."},
        "recommended_links": [
            {
                "title": "Follow-up",
                "url": "https://example.com/follow-up",
                "reason": "Provides additional architecture guidance.",
            }
        ],
    }
    fake_llm_client = _FakeLLMClient([json_dumps(llm_output)])

    settings = _make_settings(tmp_path)

    results = await gather_pages(
        [url],
        research_prompt="Prompt",
        settings=settings,
        http_client=cast(httpx.AsyncClient, fake_http_client),
        llm_client=cast(AsyncOpenAI, fake_llm_client),
    )

    assert len(results) == 1
    processed = results[0]

    assert processed.page.url == url
    assert processed.assessment.summary == llm_output["summary"]
    assert processed.assessment.technical_depth.rating == 4
    assert processed.assessment.prompt_fit.rating == 5

    assert processed.page.markdown_path is not None
    assert processed.page.markdown_path.exists()
    stored = processed.page.markdown_path.read_text(encoding="utf-8")
    assert "Sample content paragraph" in stored

    assert fake_http_client.requested == [url]
    assert fake_llm_client.responses.calls == 1


@pytest.mark.asyncio
async def test_gather_pages_skips_when_llm_fails(tmp_path: Path) -> None:
    """If the LLM returns invalid JSON the page should be skipped after markdown persistence."""

    url = "https://example.com/bad"
    html = "<html><body><p>Content</p></body></html>"
    fake_http_client = _FakeHTTPClient({url: html})
    fake_llm_client = _FakeLLMClient(["not json"])

    settings = _make_settings(tmp_path)

    results = await gather_pages(
        [url],
        research_prompt="Prompt",
        settings=settings,
        http_client=cast(httpx.AsyncClient, fake_http_client),
        llm_client=cast(AsyncOpenAI, fake_llm_client),
    )

    assert results == []

    markdown_files = list(tmp_path.glob("*.md"))
    assert len(markdown_files) == 1
    assert markdown_files[0].read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_gather_pages_processes_reddit_thread(monkeypatch, tmp_path: Path) -> None:
    reddit_url = "https://www.reddit.com/r/test/comments/abc/thread/"
    thread = RedditThread(
        identifier="abc",
        url=reddit_url,
        title="Sample Thread",
        author="thread_op",
        body_html="<p>OP body</p>",
        score=12,
        comments=[
            RedditComment(
                identifier="c1",
                author="commenter",
                body_html="<p>First comment</p>",
                score=5,
                children=[],
            )
        ],
    )

    async def _fake_fetch_resource(
        url: str,
        client: httpx.AsyncClient,
        settings: Settings,
        *,
        semaphore: asyncio.Semaphore | None = None,
    ) -> RedditThread:
        assert url == reddit_url
        assert isinstance(client, _FakeHTTPClient)
        assert isinstance(settings, Settings)
        assert isinstance(semaphore, asyncio.Semaphore)
        assert semaphore._value == settings.fetch_concurrency
        return thread

    monkeypatch.setattr("scolar.pipeline.fetch_resource", _fake_fetch_resource)

    fake_http_client = _FakeHTTPClient({})

    llm_output = {
        "summary": "Summary",
        "technical_depth": {"rating": 3, "justification": "Explains details."},
        "prompt_fit": {"rating": 4, "justification": "Relevant."},
        "recommended_links": [],
    }
    fake_llm_client = _FakeLLMClient([json_dumps(llm_output)])

    settings = _make_settings(tmp_path)

    results = await gather_pages(
        [reddit_url],
        research_prompt="Prompt",
        settings=settings,
        http_client=cast(httpx.AsyncClient, fake_http_client),
        llm_client=cast(AsyncOpenAI, fake_llm_client),
    )

    assert len(results) == 1
    processed = results[0]

    assert processed.page.url == reddit_url
    assert processed.page.title == "Sample Thread"
    assert "[1] thread_op: Sample Thread - OP body" in processed.page.markdown
    assert "[1.1] commenter: First comment" in processed.page.markdown
    assert processed.page.links == []
    assert processed.page.truncated is False

    assert fake_http_client.requested == []


def json_dumps(obj: dict) -> str:
    import json

    return json.dumps(obj)


@pytest.mark.asyncio
async def test_gather_pages_uses_cache_within_ttl(tmp_path: Path) -> None:
    """Cached pages fetched within the TTL should be reused without new HTTP or LLM calls."""

    settings = _make_settings(tmp_path)
    url = "https://example.com/cached"

    markdown_path = tmp_path / "cached.md"
    markdown_text = "Cached markdown content"
    markdown_path.write_text(markdown_text, encoding="utf-8")

    cached_page = PageContent(
        url=url,
        title="Cached Title",
        markdown=markdown_text,
        links=[LinkInfo(title="More", url="https://example.com/more")],
        truncated=False,
        markdown_path=markdown_path,
    )
    cached_assessment = PageAssessment(
        summary="Cached summary",
        technical_depth=Score(rating=4, justification="Depth"),
        prompt_fit=Score(rating=5, justification="Fit"),
        recommended_links=[
            RecommendedLink(
                title="Follow",
                url="https://example.com/follow",
                reason="More",
            )
        ],
    )

    cache = PageCache(settings)
    await cache.save(url=url, page=cached_page, assessment=cached_assessment)

    fake_http_client = _FakeHTTPClient({})
    fake_llm_client = _FakeLLMClient([])

    results = await gather_pages(
        [url],
        research_prompt="Prompt",
        settings=settings,
        http_client=cast(httpx.AsyncClient, fake_http_client),
        llm_client=cast(AsyncOpenAI, fake_llm_client),
    )

    assert len(results) == 1
    processed = results[0]
    assert processed.page.markdown == markdown_text
    assert processed.assessment.summary == "Cached summary"
    assert fake_http_client.requested == []
    assert fake_llm_client.responses.calls == 0


@pytest.mark.asyncio
async def test_gather_pages_refresh_flag_bypasses_cache(tmp_path: Path) -> None:
    """The refresh flag should force new network and LLM work even when cache exists."""

    settings = _make_settings(tmp_path)
    url = "https://example.com/refresh"

    markdown_path = tmp_path / "refresh.md"
    markdown_path.write_text("Old", encoding="utf-8")

    cached_page = PageContent(
        url=url,
        title="Old Title",
        markdown="Old",
        links=[],
        truncated=False,
        markdown_path=markdown_path,
    )
    cached_assessment = PageAssessment(
        summary="Old summary",
        technical_depth=Score(rating=2, justification="Old depth"),
        prompt_fit=Score(rating=2, justification="Old fit"),
        recommended_links=[],
    )

    cache = PageCache(settings)
    await cache.save(url=url, page=cached_page, assessment=cached_assessment)

    html = """
    <html><head><title>New Title</title></head>
    <body><p>Fresh content</p></body>
    </html>
    """
    fake_http_client = _FakeHTTPClient({url: html})

    llm_output = {
        "summary": "New summary",
        "technical_depth": {
            "rating": 5,
            "justification": "Very deep",
        },
        "prompt_fit": {"rating": 4, "justification": "Quite relevant"},
        "recommended_links": [],
    }
    fake_llm_client = _FakeLLMClient([json_dumps(llm_output)])

    results = await gather_pages(
        [url],
        research_prompt="Prompt",
        settings=settings,
        http_client=cast(httpx.AsyncClient, fake_http_client),
        llm_client=cast(AsyncOpenAI, fake_llm_client),
        refresh_cache=True,
    )

    assert fake_http_client.requested == [url]
    assert fake_llm_client.responses.calls == 1
    assert len(results) == 1
    processed = results[0]
    assert processed.assessment.summary == "New summary"
    assert processed.page.title == "New Title"
