from __future__ import annotations

import logging
from dataclasses import dataclass
from textwrap import dedent

from openai import AsyncOpenAI

from .config import Settings
from .pipeline import ProcessedPage

logger = logging.getLogger(__name__)


SYNTHESIS_SYSTEM_PROMPT = (
    "You are an expert research synthesizer. Combine evidence from provided pages "
    "to answer the research prompt. Be precise, neutral, and acknowledge "
    "uncertainties."
)


@dataclass(slots=True)
class SynthesisResult:
    answer: str
    ordered_pages: list[ProcessedPage]


def _ordered_pages(pages: list[ProcessedPage]) -> list[ProcessedPage]:
    return sorted(
        pages,
        key=lambda item: (
            item.assessment.prompt_fit.rating,
            item.assessment.technical_depth.rating,
        ),
        reverse=True,
    )


def _excerpt(markdown: str, limit: int) -> str:
    text = markdown.strip()
    if not text:
        return "[No content extracted]"
    if limit <= 0 or len(text) <= limit:
        return text

    clipped = text[:limit].rstrip()
    return f"{clipped}\n...[truncated]..."


def _build_context(pages: list[ProcessedPage], limit: int) -> str:
    sections: list[str] = []
    for index, item in enumerate(pages, start=1):
        block = dedent(
            f"""
            Page {index}: {item.page.title}
            URL: {item.page.url}
            Prompt fit: {item.assessment.prompt_fit.rating}/5 - {item.assessment.prompt_fit.justification}
            Technical depth: {item.assessment.technical_depth.rating}/5 - {item.assessment.technical_depth.justification}
            Summary: {item.assessment.summary}
            Content excerpt:
            ---
            {_excerpt(item.page.markdown, limit)}
            ---
            """
        ).strip()
        sections.append(block)
    return "\n\n".join(sections)


async def synthesize_answer(
    client: AsyncOpenAI,
    settings: Settings,
    research_prompt: str,
    pages: list[ProcessedPage],
) -> SynthesisResult | None:
    if not pages:
        logger.warning("Requested synthesis with no pages available")
        return None

    ordered = _ordered_pages(pages)
    selected = ordered[: settings.final_answer_max_pages]

    context = _build_context(selected, settings.final_answer_excerpt_chars)
    user_prompt = dedent(
        f"""
        Research prompt:
        {research_prompt}

        The following page digests are ordered from most relevant to least, based on the
        prompt fit and technical depth ratings. Use only this evidence to answer the
        research prompt. Cite supporting material inline using the notation (Page N).
        If the information is insufficient, state the gaps explicitly.

        Page digests:
        {context}

        Respond in markdown with the following structure:
        ## Answer
        <direct response>

        ## Evidence
        - <bullet points referencing Page N>

        ## Remaining Gaps
        <short explanation or "None">

        ## Suggest Follow-up Questions
        - <bullet points with suggested Questions>
        """
    ).strip()

    try:
        response = await client.responses.create(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            input=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("OpenAI synthesis request failed: %s", exc)
        return None

    raw_output = (
        response.output_text.strip() if hasattr(response, "output_text") else ""
    )
    if not raw_output:
        logger.error("Empty synthesis response from model")
        return None

    return SynthesisResult(answer=raw_output, ordered_pages=selected)


__all__ = ["SYNTHESIS_SYSTEM_PROMPT", "SynthesisResult", "synthesize_answer"]
