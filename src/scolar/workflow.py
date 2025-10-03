from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from openai import AsyncOpenAI
from pydantic import Field

from .answer import SynthesisResult
from .config import Settings
from .pipeline import ProcessedPage
from .search import SearchExpansion

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResearchResult:
    prompt: str
    urls: list[str]
    search_plan: SearchExpansion | None
    processed_pages: list[ProcessedPage]
    synthesis: SynthesisResult | None
    exit_code: int
    errors: list[str] = field(default_factory=list)


@runtime_checkable
class GenerateSearchQueriesFn(Protocol):
    async def __call__(
        self,
        client: AsyncOpenAI,
        settings: Settings,
        research_prompt: str,
    ) -> SearchExpansion | None:
        ...


@runtime_checkable
class GatherPagesFn(Protocol):
    async def __call__(
        self,
        urls: list[str],
        research_prompt: str,
        *,
        settings: Settings,
        http_client: httpx.AsyncClient,
        llm_client: AsyncOpenAI,
        refresh_cache: bool,
    ) -> list[ProcessedPage]:
        ...


@runtime_checkable
class SynthesizeAnswerFn(Protocol):
    async def __call__(
        self,
        client: AsyncOpenAI,
        settings: Settings,
        research_prompt: str,
        pages: list[ProcessedPage],
    ) -> SynthesisResult | None:
        ...


class ResearchStartEvent(StartEvent):
    prompt: str
    urls: list[str] = Field(default_factory=list)
    suggest_queries: bool = False
    refresh_cache: bool = False


class ResearchPreparedEvent(Event):
    prompt: str
    urls: list[str] = Field(default_factory=list)
    search_plan: SearchExpansion | None = None
    refresh_cache: bool = False


class PagesReadyEvent(Event):
    prompt: str
    urls: list[str] = Field(default_factory=list)
    search_plan: SearchExpansion | None = None
    results: list[ProcessedPage] = Field(default_factory=list)


class ResearchWorkflow(Workflow):
    def __init__(
        self,
        *,
        settings: Settings,
        http_client: httpx.AsyncClient,
        llm_client: AsyncOpenAI,
        generate_search_queries_fn: GenerateSearchQueriesFn,
        gather_pages_fn: GatherPagesFn,
        synthesize_answer_fn: SynthesizeAnswerFn,
    ) -> None:
        super().__init__(timeout=None)
        self._settings = settings
        self._http_client = http_client
        self._llm_client = llm_client
        self._generate_search_queries = generate_search_queries_fn
        self._gather_pages = gather_pages_fn
        self._synthesize_answer = synthesize_answer_fn

    @step(num_workers=1)
    async def start(
        self, event: ResearchStartEvent
    ) -> ResearchPreparedEvent | StopEvent:
        urls = list(event.urls)
        logger.info(
            "Workflow[start]: prompt=%r urls=%d suggest_queries=%s refresh_cache=%s",
            event.prompt,
            len(urls),
            event.suggest_queries,
            event.refresh_cache,
        )

        if not urls and not event.suggest_queries:
            message = "At least one URL must be provided via --url or --urls-file"
            logger.error(message)
            result = ResearchResult(
                prompt=event.prompt,
                urls=urls,
                search_plan=None,
                processed_pages=[],
                synthesis=None,
                exit_code=2,
                errors=[message],
            )
            return StopEvent(result=result)

        search_plan: SearchExpansion | None = None
        if event.suggest_queries:
            search_plan = await self._generate_search_queries(
                self._llm_client,
                self._settings,
                event.prompt,
            )
            logger.info(
                "Workflow[start]: generated search plan=%s",
                bool(search_plan),
            )

        if urls:
            logger.info(
                "Workflow[start]: advancing with %d urls (search_plan=%s)",
                len(urls),
                bool(search_plan),
            )
            return ResearchPreparedEvent(
                prompt=event.prompt,
                urls=urls,
                search_plan=search_plan,
                refresh_cache=event.refresh_cache,
            )

        if search_plan:
            logger.info("Workflow[start]: emitting search plan only result")
            result = ResearchResult(
                prompt=event.prompt,
                urls=urls,
                search_plan=search_plan,
                processed_pages=[],
                synthesis=None,
                exit_code=0,
            )
            return StopEvent(result=result)

        message = "Failed to generate search queries for the provided prompt"
        logger.error(message)
        result = ResearchResult(
            prompt=event.prompt,
            urls=urls,
            search_plan=None,
            processed_pages=[],
            synthesis=None,
            exit_code=1,
            errors=[message],
        )
        return StopEvent(result=result)

    @step(num_workers=1)
    async def gather(self, event: ResearchPreparedEvent) -> PagesReadyEvent | StopEvent:
        logger.info(
            "Workflow[gather]: fetching %d urls refresh_cache=%s",
            len(event.urls),
            event.refresh_cache,
        )

        results = await self._gather_pages(
            event.urls,
            event.prompt,
            settings=self._settings,
            http_client=self._http_client,
            llm_client=self._llm_client,
            refresh_cache=event.refresh_cache,
        )
        if not results:
            message = "No pages processed successfully"
            logger.error(message)
            result = ResearchResult(
                prompt=event.prompt,
                urls=event.urls,
                search_plan=event.search_plan,
                processed_pages=[],
                synthesis=None,
                exit_code=1,
                errors=[message],
            )
            return StopEvent(result=result)

        logger.info(
            "Workflow[gather]: processed %d/%d urls",
            len(results),
            len(event.urls),
        )

        return PagesReadyEvent(
            prompt=event.prompt,
            urls=event.urls,
            search_plan=event.search_plan,
            results=results,
        )

    @step(num_workers=1)
    async def synthesize(self, event: PagesReadyEvent) -> StopEvent:
        logger.info(
            "Workflow[synthesize]: synthesizing prompt=%r from %d pages",
            event.prompt,
            len(event.results),
        )
        synthesis = await self._synthesize_answer(
            self._llm_client,
            self._settings,
            event.prompt,
            event.results,
        )
        logger.info(
            "Workflow[synthesize]: synthesis=%s",
            bool(synthesis),
        )
        result = ResearchResult(
            prompt=event.prompt,
            urls=event.urls,
            search_plan=event.search_plan,
            processed_pages=event.results,
            synthesis=synthesis,
            exit_code=0,
        )
        logger.info("Workflow[synthesize]: signalling completion")
        return StopEvent(result=result)


__all__ = [
    "PagesReadyEvent",
    "ResearchPreparedEvent",
    "ResearchResult",
    "ResearchStartEvent",
    "ResearchWorkflow",
]
