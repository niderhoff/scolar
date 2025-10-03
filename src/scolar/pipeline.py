from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import httpx
from openai import AsyncOpenAI

from .config import Settings
from .fetcher import fetch_html
from .models import PageAssessment, PageContent
from .parser import parse_html
from .storage import store_markdown
from .summarizer import assess_page

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProcessedPage:
    page: PageContent
    assessment: PageAssessment


async def process_url(
    url: str,
    research_prompt: str,
    *,
    http_client: httpx.AsyncClient,
    llm_client: AsyncOpenAI,
    settings: Settings,
    fetch_semaphore: Optional[asyncio.Semaphore] = None,
    llm_semaphore: Optional[asyncio.Semaphore] = None,
) -> Optional[ProcessedPage]:
    html = await fetch_html(url, http_client, settings, semaphore=fetch_semaphore)
    if not html:
        return None

    page = await asyncio.to_thread(parse_html, url, html, settings)

    markdown_path = await store_markdown(page, settings)
    page.markdown_path = markdown_path
    logger.info("Stored markdown for %s at %s", url, markdown_path)

    assessment = await assess_page(
        llm_client,
        settings,
        research_prompt,
        page=page,
        semaphore=llm_semaphore,
    )
    if not assessment:
        return None

    return ProcessedPage(page=page, assessment=assessment)


async def gather_pages(
    urls: list[str],
    research_prompt: str,
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
    llm_client: AsyncOpenAI,
) -> list[ProcessedPage]:
    fetch_semaphore = asyncio.Semaphore(settings.fetch_concurrency)
    llm_semaphore = asyncio.Semaphore(settings.llm_concurrency)

    tasks = [
        asyncio.create_task(
            process_url(
                url,
                research_prompt,
                http_client=http_client,
                llm_client=llm_client,
                settings=settings,
                fetch_semaphore=fetch_semaphore,
                llm_semaphore=llm_semaphore,
            )
        )
        for url in urls
    ]

    results: list[ProcessedPage] = []
    for task in asyncio.as_completed(tasks):
        try:
            result = await task
        except Exception:  # noqa: BLE001
            logger.exception("Unhandled error processing URL")
            continue
        if result:
            results.append(result)

    return results


__all__ = ["ProcessedPage", "process_url", "gather_pages"]
