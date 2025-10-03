"""Tests for the final answer synthesis stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from openai import AsyncOpenAI

from scolar.answer import SynthesisResult, synthesize_answer
from scolar.config import Settings
from scolar.models import LinkInfo, PageAssessment, PageContent, RecommendedLink, Score
from scolar.pipeline import ProcessedPage


@dataclass
class _FakeLLMResponse:
    output_text: str


class _FakeResponses:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    async def create(self, **kwargs) -> _FakeLLMResponse:  # noqa: ANN003, ANN204
        self.calls.append(kwargs)
        return _FakeLLMResponse(output_text=self.payload)


class _FakeLLMClient:
    def __init__(self, payload: str) -> None:
        self.responses = _FakeResponses(payload)


def _settings(
    tmp_path: Path, *, max_pages: int = 2, excerpt_chars: int = 240
) -> Settings:
    return Settings(
        fetch_concurrency=2,
        request_timeout=10.0,
        request_retries=0,
        request_backoff=0.0,
        user_agent="TestAgent",
        output_dir=tmp_path,
        max_markdown_chars=1_000,
        max_links_inspected=5,
        max_recommended_links=2,
        openai_model="mock-model",
        openai_temperature=0.0,
        openai_timeout=10.0,
        llm_concurrency=1,
        final_answer_max_pages=max_pages,
        final_answer_excerpt_chars=excerpt_chars,
        cache_ttl_hours=72,
    )


def _page(
    *,
    url: str,
    title: str,
    markdown: str,
    prompt_fit: int,
    prompt_fit_reason: str,
    technical_depth: int,
    technical_reason: str,
) -> ProcessedPage:
    page = PageContent(
        url=url,
        title=title,
        markdown=markdown,
        links=[LinkInfo(title="Example", url=f"{url}/more")],
        truncated=False,
    )
    assessment = PageAssessment(
        summary=f"Summary for {title}",
        technical_depth=Score(rating=technical_depth, justification=technical_reason),
        prompt_fit=Score(rating=prompt_fit, justification=prompt_fit_reason),
        recommended_links=[
            RecommendedLink(title="Follow", url=f"{url}/follow", reason="Extra"),
        ],
    )
    return ProcessedPage(page=page, assessment=assessment)


@pytest.mark.asyncio
async def test_synthesize_answer_orders_and_limits_pages(tmp_path: Path) -> None:
    """Top pages should be ordered by prompt fit then technical depth and passed to the LLM."""

    page_low = _page(
        url="https://example.com/low",
        title="Low",
        markdown="Low content",
        prompt_fit=3,
        prompt_fit_reason="Somewhat relevant",
        technical_depth=2,
        technical_reason="Surface level",
    )
    page_mid = _page(
        url="https://example.com/mid",
        title="Mid",
        markdown=(
            "Mid content with extended details that should be trimmed at the excerpt limit. "
            * 6
        ),
        prompt_fit=5,
        prompt_fit_reason="Highly relevant",
        technical_depth=1,
        technical_reason="Shallow",
    )
    page_high = _page(
        url="https://example.com/high",
        title="High",
        markdown=(
            "High content with substantial depth and detailed explanations spanning multiple sentences. "
            * 8
        ),
        prompt_fit=5,
        prompt_fit_reason="Highly relevant",
        technical_depth=4,
        technical_reason="Deep dive",
    )

    client = _FakeLLMClient(
        "## Answer\nReady\n\n## Evidence\n- (Page 1)\n\n## Remaining Gaps\nNone"
    )
    settings = _settings(tmp_path, max_pages=2, excerpt_chars=240)

    result = await synthesize_answer(
        cast(AsyncOpenAI, client),
        settings,
        research_prompt="Prompt",
        pages=[page_low, page_mid, page_high],
    )

    assert isinstance(result, SynthesisResult)
    assert result.answer.startswith("## Answer")
    assert [item.page.title for item in result.ordered_pages] == ["High", "Mid"]

    call = client.responses.calls[0]
    user_message = call["input"][1]["content"]
    assert "Page 1: High" in user_message
    assert "Page 2: Mid" in user_message
    assert "...[truncated]..." in user_message
    assert "Low" not in user_message


@pytest.mark.asyncio
async def test_synthesize_answer_returns_none_on_empty_payload(tmp_path: Path) -> None:
    """If the LLM returns an empty payload the synthesis result should be None."""

    page = _page(
        url="https://example.com",
        title="Example",
        markdown="Content",
        prompt_fit=4,
        prompt_fit_reason="Relevant",
        technical_depth=3,
        technical_reason="Detailed",
    )

    class _EmptyClient(_FakeLLMClient):
        def __init__(self) -> None:  # noqa: D401
            super().__init__(payload=" ")

    client = _EmptyClient()
    settings = _settings(tmp_path)

    result = await synthesize_answer(
        cast(AsyncOpenAI, client),
        settings,
        research_prompt="Prompt",
        pages=[page],
    )

    assert result is None


@pytest.mark.asyncio
async def test_synthesize_answer_handles_no_pages(tmp_path: Path) -> None:
    """Requesting synthesis without pages should short-circuit gracefully."""

    client = _FakeLLMClient("irrelevant")
    settings = _settings(tmp_path)

    result = await synthesize_answer(
        cast(AsyncOpenAI, client),
        settings,
        research_prompt="Prompt",
        pages=[],
    )

    assert result is None
    assert client.responses.calls == []
