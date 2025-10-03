from __future__ import annotations

import asyncio
import json
import logging
from textwrap import dedent

from openai import AsyncOpenAI

from .config import Settings
from .models import (
    AssessmentPayload,
    PageAssessment,
    PageContent,
    ValidationError,
    payload_to_assessment,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a careful research assistant. Summarize web pages and judge their usefulness "
    "for the provided research prompt. Respond in compact JSON only."
)


def _links_payload(page: PageContent, limit: int) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for link in page.links[:limit]:
        payload.append({"title": link.title, "url": link.url})
    return payload


async def assess_page(
    client: AsyncOpenAI,
    settings: Settings,
    research_prompt: str,
    page: PageContent,
    *,
    semaphore: asyncio.Semaphore | None = None,
) -> PageAssessment | None:
    links_payload = _links_payload(page, settings.max_links_inspected)
    prompt = dedent(
        f"""
        Research prompt:
        {research_prompt}

        Page title: {page.title}
        Page URL: {page.url}
        Page content (Markdown only){" [TRUNCATED]" if page.truncated else ""}:
        ---
        {page.markdown}
        ---

        Outbound links (first {len(links_payload)}):
        {json.dumps(links_payload, ensure_ascii=False)}

        Respond strictly as JSON with:
        {{
          "summary": <80-120 word neutral summary>,
          "technical_depth": {{"rating": 1-5, "justification": <text>}},
          "prompt_fit": {{"rating": 1-5, "justification": <text>}},
          "recommended_links": [
            {{"title": <text>, "url": <absolute url>, "reason": <text>}}
          ]
        }}
        Limit recommended_links to at most {settings.max_recommended_links} items that advance the research.
        """
    ).strip()

    try:
        if semaphore:
            async with semaphore:
                response = await client.responses.create(
                    model=settings.openai_model,
                    temperature=settings.openai_temperature,
                    input=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
        else:
            response = await client.responses.create(
                model=settings.openai_model,
                temperature=settings.openai_temperature,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("OpenAI request failed for %s: %s", page.url, exc)
        return None

    raw_output = (
        response.output_text.strip() if hasattr(response, "output_text") else ""
    )
    if not raw_output:
        logger.error("Empty response from model for %s", page.url)
        return None

    try:
        payload = AssessmentPayload.parse_json(raw_output)
    except ValidationError as exc:
        logger.error(
            "Invalid JSON from model for %s: %s\nPayload: %s", page.url, exc, raw_output
        )
        return None

    assessment = payload_to_assessment(payload)
    if settings.max_recommended_links >= 0:
        assessment.recommended_links = assessment.recommended_links[
            : settings.max_recommended_links
        ]
    return assessment


__all__ = ["assess_page", "SYSTEM_PROMPT"]
