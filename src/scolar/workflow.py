from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.utils.workflow import draw_all_possible_flows
from openai import AsyncOpenAI
from pydantic import Field

from .answer import SynthesisResult
from .config import Settings
from .discovery import SearchHitCache
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
    ) -> SearchExpansion | None: ...


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
    ) -> list[ProcessedPage]: ...


@runtime_checkable
class DiscoverCandidateUrlsFn(Protocol):
    async def __call__(
        self,
        prompt: str,
        *,
        http_client: httpx.AsyncClient,
        settings: Settings,
        refresh_cache: bool,
        cache: SearchHitCache | None = None,
    ) -> list[str]: ...


@runtime_checkable
class SynthesizeAnswerFn(Protocol):
    async def __call__(
        self,
        client: AsyncOpenAI,
        settings: Settings,
        research_prompt: str,
        pages: list[ProcessedPage],
    ) -> SynthesisResult | None: ...


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


class CandidateUrlsEvent(Event):
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
        discover_candidate_urls_fn: DiscoverCandidateUrlsFn,
        gather_pages_fn: GatherPagesFn,
        synthesize_answer_fn: SynthesizeAnswerFn,
    ) -> None:
        super().__init__(timeout=None)
        self._settings = settings
        self._http_client = http_client
        self._llm_client = llm_client
        self._generate_search_queries = generate_search_queries_fn
        self._discover_candidate_urls = discover_candidate_urls_fn
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

    @step(num_workers=1)
    async def discover(
        self, event: ResearchPreparedEvent
    ) -> CandidateUrlsEvent | StopEvent:
        if event.urls:
            logger.info(
                "Workflow[discover]: received %d pre-supplied urls",
                len(event.urls),
            )
            return CandidateUrlsEvent(
                prompt=event.prompt,
                urls=event.urls,
                search_plan=event.search_plan,
                refresh_cache=event.refresh_cache,
            )

        urls = await self._discover_candidate_urls(
            prompt=event.prompt,
            http_client=self._http_client,
            settings=self._settings,
            refresh_cache=event.refresh_cache,
        )

        if not urls:
            message = "No candidate URLs discovered for the provided prompt"
            logger.error(message)
            result = ResearchResult(
                prompt=event.prompt,
                urls=[],
                search_plan=event.search_plan,
                processed_pages=[],
                synthesis=None,
                exit_code=3,
                errors=[message],
            )
            return StopEvent(result=result)

        logger.info("Workflow[discover]: discovered %d candidate urls", len(urls))

        return CandidateUrlsEvent(
            prompt=event.prompt,
            urls=urls,
            search_plan=event.search_plan,
            refresh_cache=event.refresh_cache,
        )

    @step(num_workers=1)
    async def gather(self, event: CandidateUrlsEvent) -> PagesReadyEvent | StopEvent:
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


def visualize_research_workflow(
    workflow: ResearchWorkflow,
    *,
    output_path: str = "workflow.html",
    notebook: bool = False,
    max_label_length: int | None = None,
) -> None:
    """Render the research workflow graph to an HTML file for inspection."""

    draw_all_possible_flows(
        workflow,
        filename=output_path,
        notebook=notebook,
        max_label_length=max_label_length,
    )


__all__ = [
    "CandidateUrlsEvent",
    "PagesReadyEvent",
    "ResearchPreparedEvent",
    "ResearchResult",
    "ResearchStartEvent",
    "ResearchWorkflow",
    "visualize_research_workflow",
]
