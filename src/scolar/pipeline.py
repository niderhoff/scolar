from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI

from .cache import PageCache
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
    cache: PageCache | None = None,
    fetch_semaphore: asyncio.Semaphore | None = None,
    llm_semaphore: asyncio.Semaphore | None = None,
) -> ProcessedPage | None:
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

    processed = ProcessedPage(page=page, assessment=assessment)

    if cache:
        await cache.save(url=url, page=page, assessment=assessment)

    return processed


async def gather_pages(
    urls: list[str],
    research_prompt: str,
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
    llm_client: AsyncOpenAI,
    refresh_cache: bool = False,
) -> list[ProcessedPage]:
    fetch_semaphore = asyncio.Semaphore(settings.fetch_concurrency)
    llm_semaphore = asyncio.Semaphore(settings.llm_concurrency)

    cache = PageCache(settings)

    tasks: list[asyncio.Task[ProcessedPage | None]] = []
    task_urls: list[str] = []
    results_by_url: dict[str, ProcessedPage] = {}

    for url in urls:
        if not refresh_cache:
            cached = await cache.load(url)
            if cached:
                logger.info(
                    "Cache hit for %s (fetched at %s)",
                    url,
                    cached.fetched_at.isoformat(),
                )
                results_by_url[url] = ProcessedPage(
                    page=cached.page, assessment=cached.assessment
                )
                continue

            logger.info("Cache miss for %s; scheduling fetch", url)
        else:
            logger.info("Refresh requested; scheduling fetch for %s", url)

        task = asyncio.create_task(
            process_url(
                url,
                research_prompt,
                http_client=http_client,
                llm_client=llm_client,
                settings=settings,
                cache=cache,
                fetch_semaphore=fetch_semaphore,
                llm_semaphore=llm_semaphore,
            )
        )
        tasks.append(task)
        task_urls.append(url)

    if tasks:
        completed = await asyncio.gather(*tasks, return_exceptions=True)
        for url, outcome in zip(task_urls, completed, strict=False):
            if isinstance(outcome, Exception):
                logger.error("Unhandled error processing URL %s", url, exc_info=outcome)
                continue
            if outcome:
                results_by_url[url] = outcome

    ordered_results: list[ProcessedPage] = []
    for url in urls:
        result = results_by_url.get(url)
        if result:
            ordered_results.append(result)

    return ordered_results


__all__ = ["ProcessedPage", "process_url", "gather_pages"]
