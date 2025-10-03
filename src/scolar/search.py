"""Search query expansion helpers driven by the LLM client."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from textwrap import dedent
from typing import Sequence

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from .config import Settings

logger = logging.getLogger(__name__)

SEARCH_SYSTEM_PROMPT = (
    "You are an adept web research strategist. Expand the user's research prompt "
    "into effective web search queries and supporting keywords. Respond strictly "
    "with JSON that follows the requested schema."
)


@dataclass(slots=True)
class SearchExpansion:
    """Structured representation of generated search queries and keywords."""

    primary_query: str
    expanded_queries: list[str]
    focus_topics: list[str]
    site_filters: list[str]
    notes: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "primary_query": self.primary_query,
            "expanded_queries": list(self.expanded_queries),
            "focus_topics": list(self.focus_topics),
            "site_filters": list(self.site_filters),
            "notes": self.notes,
        }


class SearchExpansionPayload(BaseModel):
    primary_query: str = Field(..., min_length=1)
    expanded_queries: list[str] = Field(default_factory=list)
    focus_topics: list[str] = Field(default_factory=list)
    site_filters: list[str] = Field(default_factory=list)
    notes: str | None = None

    @classmethod
    def parse_json(cls, data: str) -> "SearchExpansionPayload":
        return cls.model_validate_json(data)


def _clean_unique(values: Sequence[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = value.strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        result.append(text)
        if len(result) >= limit:
            break
    return result


def render_search_expansion(expansion: SearchExpansion) -> str:
    lines: list[str] = ["# Suggested Search Queries", ""]
    lines.append(f"Primary query: {expansion.primary_query}")

    if expansion.expanded_queries:
        lines.extend(["", "## Expanded Queries"])
        for query in expansion.expanded_queries:
            lines.append(f"- {query}")

    if expansion.focus_topics:
        lines.extend(["", "## Focus Topics"])
        for topic in expansion.focus_topics:
            lines.append(f"- {topic}")

    if expansion.site_filters:
        lines.extend(["", "## Suggested Site Filters"])
        for site in expansion.site_filters:
            lines.append(f"- {site}")

    if expansion.notes:
        lines.extend(["", "## Notes", expansion.notes])

    return "\n".join(lines).strip()


async def generate_search_queries(
    client: AsyncOpenAI,
    settings: Settings,
    research_prompt: str,
    *,
    semaphore: asyncio.Semaphore | None = None,
) -> SearchExpansion | None:
    """Ask the LLM for search queries relevant to the research prompt."""

    max_queries = max(settings.final_answer_max_pages, 5)
    payload_spec = json.dumps(
        {
            "primary_query": "string",
            "expanded_queries": ["string", "..."],
            "focus_topics": ["string", "..."],
            "site_filters": ["site:example.com", "..."],
            "notes": "string | null",
        },
        ensure_ascii=False,
    )

    user_prompt = dedent(
        f"""
        Research prompt:
        {research_prompt}

        Produce JSON that conforms to this schema:
        {payload_spec}

        Guidance:
        - primary_query should be the single best general-purpose query for search engines.
        - expanded_queries should list up to {max_queries} diverse variations covering complementary angles.
        - focus_topics should include 3-{max_queries} short keywords or phrases to mix and match.
        - site_filters should include domain or filetype qualifiers when appropriate; return an empty list if none.
        - notes is optional but may contain strategy tips. Use null when no additional guidance is needed.
        Ensure the response is valid JSON and obey the limits.
        """
    ).strip()

    try:
        if semaphore:
            async with semaphore:
                response = await client.responses.create(
                    model=settings.openai_model,
                    temperature=settings.openai_temperature,
                    input=[
                        {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
        else:
            response = await client.responses.create(
                model=settings.openai_model,
                temperature=settings.openai_temperature,
                input=[
                    {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("OpenAI search expansion request failed: %s", exc)
        return None

    raw_output = (
        response.output_text.strip() if hasattr(response, "output_text") else ""
    )
    if not raw_output:
        logger.error("Empty search expansion response from model")
        return None

    try:
        payload = SearchExpansionPayload.parse_json(raw_output)
    except ValidationError as exc:
        logger.error("Invalid JSON from search expansion model: %s", exc)
        return None

    expansion = SearchExpansion(
        primary_query=payload.primary_query.strip(),
        expanded_queries=_clean_unique(payload.expanded_queries, limit=max_queries),
        focus_topics=_clean_unique(payload.focus_topics, limit=max_queries),
        site_filters=_clean_unique(payload.site_filters, limit=5),
        notes=payload.notes.strip() if payload.notes else None,
    )

    return expansion


__all__ = [
    "SearchExpansion",
    "SearchExpansionPayload",
    "SEARCH_SYSTEM_PROMPT",
    "generate_search_queries",
    "render_search_expansion",
]
