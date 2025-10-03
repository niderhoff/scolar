from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import pytest

from openai import AsyncOpenAI

from scolar.config import Settings
from scolar.search import (
    SearchExpansion,
    generate_search_queries,
    render_search_expansion,
)


@dataclass
class _FakeLLMResponse:
    output_text: str


class _FakeLLMResponses:
    def __init__(self, outputs: Iterable[str]) -> None:
        self._outputs = list(outputs)
        self.calls = 0

    async def create(self, **_kwargs) -> _FakeLLMResponse:  # noqa: ANN003
        output = self._outputs[self.calls]
        self.calls += 1
        return _FakeLLMResponse(output_text=output)


class _FakeLLMClient:
    def __init__(self, outputs: Iterable[str]) -> None:
        self.responses = _FakeLLMResponses(outputs)


@pytest.mark.asyncio
async def test_generate_search_queries_parses_payload() -> None:
    payload = {
        "primary_query": "quantum computing research roadmap",
        "expanded_queries": [
            "quantum computing near term roadmap",
            "post-quantum cryptography adoption timeline",
        ],
        "focus_topics": ["quantum hardware roadmap", "fault tolerance status"],
        "site_filters": ["site:nature.com", "filetype:pdf"],
        "notes": "Mix roadmaps with funding outlook keywords.",
    }
    fake_client = _FakeLLMClient([json.dumps(payload)])
    settings = Settings()

    result = await generate_search_queries(
        cast(AsyncOpenAI, fake_client), settings, "Quantum research roadmap"
    )

    assert result is not None
    assert result.primary_query == payload["primary_query"]
    assert result.expanded_queries == payload["expanded_queries"]
    assert result.focus_topics == payload["focus_topics"]
    assert result.site_filters == payload["site_filters"]
    assert result.notes == payload["notes"]


@pytest.mark.asyncio
async def test_generate_search_queries_deduplicates_and_limits() -> None:
    payload = {
        "primary_query": "machine learning reproducibility case studies",
        "expanded_queries": [
            "ml reproducibility benchmarks",
            "ml reproducibility benchmarks",
            "ml replication checklists",
            "ml reproducibility governance",
            "ml replication dataset access",
            "ml experiment tracking open source",
        ],
        "focus_topics": [
            "experiment tracking",
            "experiment tracking",
            "data versioning",
            "governance",
            "model cards",
        ],
        "site_filters": [
            "site:arxiv.org",
            "site:arxiv.org",
            "site:nips.cc",
            "site:paperswithcode.com",
            "site:mlcommons.org",
            "site:mlcommons.org",
        ],
        "notes": "",
    }
    fake_client = _FakeLLMClient([json.dumps(payload)])
    settings = Settings(final_answer_max_pages=3)

    result = await generate_search_queries(
        cast(AsyncOpenAI, fake_client), settings, "ML reproducibility"
    )

    assert result is not None
    assert len(result.expanded_queries) == 5  # limit defaults to at least 5
    assert result.expanded_queries[0] == "ml reproducibility benchmarks"
    assert "ml reproducibility governance" in result.expanded_queries
    assert len(result.focus_topics) == 4
    assert result.focus_topics[0] == "experiment tracking"
    assert len(result.site_filters) == 4
    assert result.site_filters[-1] == "site:mlcommons.org"
    assert result.notes is None


def test_render_search_expansion_outputs_sections() -> None:
    expansion = SearchExpansion(
        primary_query="llm agent evaluation",
        expanded_queries=["llm agent benchmarking", "autonomous agent assessment"],
        focus_topics=["evaluation dataset", "hallucination metrics"],
        site_filters=["site:arxiv.org"],
        notes="Focus on 2024-2025 literature.",
    )

    text = render_search_expansion(expansion)

    assert "Suggested Search Queries" in text
    assert "Primary query: llm agent evaluation" in text
    assert "autonomous agent assessment" in text
    assert "hallucination metrics" in text
    assert "Focus on 2024-2025 literature." in text
