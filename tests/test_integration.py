"""Integration tests for the async pipeline using mocked HTTP and LLM clients."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import pytest

from scolar.config import Settings
from scolar.pipeline import gather_pages


@dataclass
class _FakeResponse:
    """Minimal stand-in for httpx.Response used in fetcher."""

    text: str
    headers: Dict[str, str]

    def raise_for_status(self) -> None:
        return None


class _FakeHTTPClient:
    """Async HTTP client that returns pre-baked HTML payloads keyed by URL."""

    def __init__(self, mapping: Dict[str, str]) -> None:
        self._mapping = mapping
        self.requested: list[str] = []

    async def get(self, url: str, timeout: float) -> _FakeResponse:  # noqa: ARG002
        self.requested.append(url)
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
    http_client: Any = _FakeHTTPClient({url: html})

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
    llm_client: Any = _FakeLLMClient([json_dumps(llm_output)])

    settings = _make_settings(tmp_path)

    results = await gather_pages(
        [url],
        research_prompt="Prompt",
        settings=settings,
        http_client=http_client,
        llm_client=llm_client,
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

    assert http_client.requested == [url]
    assert llm_client.responses.calls == 1


@pytest.mark.asyncio
async def test_gather_pages_skips_when_llm_fails(tmp_path: Path) -> None:
    """If the LLM returns invalid JSON the page should be skipped after markdown persistence."""

    url = "https://example.com/bad"
    html = "<html><body><p>Content</p></body></html>"
    http_client: Any = _FakeHTTPClient({url: html})
    llm_client: Any = _FakeLLMClient(["not json"])

    settings = _make_settings(tmp_path)

    results = await gather_pages(
        [url],
        research_prompt="Prompt",
        settings=settings,
        http_client=http_client,
        llm_client=llm_client,
    )

    assert results == []

    markdown_files = list(tmp_path.glob("*.md"))
    assert len(markdown_files) == 1
    assert markdown_files[0].read_text(encoding="utf-8")


def json_dumps(obj: dict) -> str:
    import json

    return json.dumps(obj)
