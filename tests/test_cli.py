"""CLI-level tests for scolar.main using patched dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scolar.answer import SynthesisResult
from scolar.config import Settings
from scolar.main import run_async
from scolar.models import (
    LinkInfo,
    PageAssessment,
    PageContent,
    RecommendedLink,
    Score,
)
from scolar.pipeline import ProcessedPage


class _DummyAsyncClient:
    def __init__(self, *args, **kwargs):  # noqa: D401, ANN002
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN202
        return False


class _DummyLLMClient:
    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_run_async_outputs_markdown_and_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI should print markdown and write JSON when requested."""

    output_dir = tmp_path / "artifacts"
    settings = Settings(
        fetch_concurrency=2,
        request_timeout=10.0,
        request_retries=0,
        request_backoff=0.0,
        user_agent="TestAgent",
        output_dir=output_dir,
        max_markdown_chars=1000,
        max_links_inspected=10,
        max_recommended_links=3,
        openai_model="mock-model",
        openai_temperature=0.0,
        openai_timeout=10.0,
        llm_concurrency=1,
        final_answer_max_pages=5,
        final_answer_excerpt_chars=1_500,
        cache_ttl_hours=72,
    )

    monkeypatch.setattr("scolar.main.load_settings", lambda: settings)
    monkeypatch.setattr("scolar.main.httpx.AsyncClient", _DummyAsyncClient)

    dummy_llm = _DummyLLMClient()
    monkeypatch.setattr("scolar.main.AsyncOpenAI", lambda timeout: dummy_llm)

    page = PageContent(
        url="https://example.com",
        title="Example Page",
        markdown="Content",
        links=[LinkInfo(title="More", url="https://example.com/more")],
        truncated=False,
    )
    markdown_path = output_dir / "example-page.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("Content", encoding="utf-8")
    page.markdown_path = markdown_path

    assessment = PageAssessment(
        summary="Summary text",
        technical_depth=Score(rating=4, justification="Technical"),
        prompt_fit=Score(rating=5, justification="Relevant"),
        recommended_links=[
            RecommendedLink(
                title="Follow", url="https://example.com/follow", reason="More info"
            ),
        ],
    )

    processed = ProcessedPage(page=page, assessment=assessment)

    async def fake_gather_pages(
        urls, prompt, *, settings, http_client, llm_client, refresh_cache
    ):  # noqa: ANN001, ANN202
        assert prompt == "Test prompt"
        assert llm_client is dummy_llm
        assert refresh_cache is False
        return [processed]

    monkeypatch.setattr("scolar.main.gather_pages", fake_gather_pages)

    async def fake_synthesize_answer(llm_client, settings, research_prompt, pages):  # noqa: ANN001, ANN202
        assert llm_client is dummy_llm
        assert research_prompt == "Test prompt"
        assert pages == [processed]
        return SynthesisResult(
            answer="Final synthesized answer", ordered_pages=[processed]
        )

    monkeypatch.setattr("scolar.main.synthesize_answer", fake_synthesize_answer)

    json_path = tmp_path / "report.json"
    args = argparse.Namespace(
        prompt="Test prompt",
        urls=["https://example.com"],
        urls_file=None,
        output_dir=None,
        json_output=json_path,
        verbose=False,
        refresh_cache=False,
    )

    exit_code = await run_async(args)
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "Final synthesized answer" in output
    assert "Example Page" in output
    assert "Summary text" in output

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["prompt"] == "Test prompt"
    assert data["pages"][0]["title"] == "Example Page"
    assert (
        data["pages"][0]["recommended_links"][0]["url"] == "https://example.com/follow"
    )
    assert data["final_answer"] == "Final synthesized answer"
    assert data["sources_consulted"][0]["title"] == "Example Page"
