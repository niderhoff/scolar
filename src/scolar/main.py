from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import httpx
from openai import AsyncOpenAI

from .answer import synthesize_answer
from .config import load_settings
from .discovery import discover_candidate_urls
from .pipeline import ProcessedPage, gather_pages
from .report import build_json_record, render_report
from .search import generate_search_queries, render_search_expansion
from .workflow import (
    ResearchResult,
    ResearchWorkflow,
    visualize_research_workflow,
)

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s - %(message)s")


def _read_urls(urls: Iterable[str], urls_file: Path | None) -> list[str]:
    collection: list[str] = list(urls)
    if urls_file:
        content = urls_file.read_text(encoding="utf-8")
        file_urls = [line.strip() for line in content.splitlines() if line.strip()]
        collection.extend(file_urls)
    deduped: list[str] = []
    seen = set()
    for url in collection:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize URLs against a research prompt"
    )
    parser.add_argument(
        "--prompt", required=True, help="Research prompt to evaluate pages against"
    )
    parser.add_argument(
        "--url",
        dest="urls",
        action="append",
        default=[],
        help="URL to include (repeat for multiple)",
    )
    parser.add_argument(
        "--urls-file",
        type=Path,
        help="Path to a file containing newline-delimited URLs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where markdown artifacts are stored (overrides setting)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to write a JSON summary report",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached pages and force fresh fetches",
    )
    parser.add_argument(
        "--suggest-queries",
        action="store_true",
        help=(
            "Generate expanded search queries for the prompt before processing URLs. "
            "If no URLs are provided, only the suggested queries are output."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


async def run_async(args: argparse.Namespace) -> int:
    if getattr(args, "command", "research") != "research":  # pragma: no cover
        raise ValueError("run_async expects research command arguments")

    _configure_logging(args.verbose)

    settings = load_settings()
    if args.output_dir:
        settings.output_dir = args.output_dir.expanduser()

    urls = _read_urls(args.urls, args.urls_file)
    result: ResearchResult | None = None
    async with httpx.AsyncClient(
        headers={"User-Agent": settings.user_agent}
    ) as http_client:
        llm_client = AsyncOpenAI(timeout=settings.openai_timeout)
        try:
            workflow = ResearchWorkflow(
                settings=settings,
                http_client=http_client,
                llm_client=llm_client,
                generate_search_queries_fn=generate_search_queries,
                discover_candidate_urls_fn=discover_candidate_urls,
                gather_pages_fn=gather_pages,
                synthesize_answer_fn=synthesize_answer,
            )
            handler = workflow.run(
                prompt=args.prompt,
                urls=urls,
                suggest_queries=args.suggest_queries,
                refresh_cache=args.refresh_cache,
            )
            result = await handler
        finally:
            close = getattr(llm_client, "close", None)
            if callable(close):
                maybe_coro = close()
                if asyncio.iscoroutine(maybe_coro):
                    await maybe_coro

    if result is None:
        logger.error("Research workflow did not return a result")
        return 1

    logger.info(
        "Workflow completed: exit_code=%d pages=%d search_plan=%s",
        result.exit_code,
        len(result.processed_pages),
        bool(result.search_plan),
    )

    separator = "\n" + "=" * 80 + "\n"
    sections: list[str] = []
    if result.search_plan:
        sections.append(render_search_expansion(result.search_plan))

    ordered_results: list[ProcessedPage] = list(result.processed_pages)
    synthesis = result.synthesis

    if synthesis:
        prioritized = list(synthesis.ordered_pages)
        prioritized_ids = {id(item) for item in prioritized}
        remaining = [
            item for item in result.processed_pages if id(item) not in prioritized_ids
        ]
        ordered_results = prioritized + remaining

    if synthesis:
        lines = ["# Final Answer", "", synthesis.answer.strip()]
        if synthesis.ordered_pages:
            lines.extend(["", "## Sources Consulted"])
            for index, item in enumerate(synthesis.ordered_pages, start=1):
                lines.append(
                    (
                        f"- Page {index}: {item.page.title} ({item.page.url}) - "
                        f"prompt fit {item.assessment.prompt_fit.rating}/5, "
                        f"technical depth {item.assessment.technical_depth.rating}/5"
                    )
                )
        sections.append("\n".join(lines).strip())

    if ordered_results:
        sections.extend(
            render_report(item.page, item.assessment) for item in ordered_results
        )

    if sections:
        if len(sections) == 1:
            print(sections[0])
        else:
            print(separator.join(sections))

    if args.json_output and result.exit_code == 0:
        payload = {
            "prompt": args.prompt,
            "final_answer": synthesis.answer if synthesis else None,
            "sources_consulted": [
                {
                    "page_number": index,
                    "title": item.page.title,
                    "url": item.page.url,
                    "prompt_fit": item.assessment.prompt_fit.rating,
                    "technical_depth": item.assessment.technical_depth.rating,
                }
                for index, item in enumerate(synthesis.ordered_pages, start=1)
            ]
            if synthesis
            else [],
            "pages": [
                build_json_record(item.page, item.assessment)
                for item in ordered_results
            ],
            "search_queries": (
                result.search_plan.to_dict() if result.search_plan else None
            ),
        }
        json_path = args.json_output.expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Wrote JSON summary to %s", json_path)

    return result.exit_code


def run_visualize(args: argparse.Namespace) -> int:
    _configure_logging(args.verbose)

    settings = load_settings()
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    workflow = ResearchWorkflow(
        settings=settings,
        http_client=cast(httpx.AsyncClient, object()),
        llm_client=cast(AsyncOpenAI, object()),
        generate_search_queries_fn=generate_search_queries,
        discover_candidate_urls_fn=discover_candidate_urls,
        gather_pages_fn=gather_pages,
        synthesize_answer_fn=synthesize_answer,
    )

    visualize_research_workflow(
        workflow,
        output_path=str(output_path),
        notebook=args.notebook,
        max_label_length=args.max_label_length,
    )

    logger.info("Workflow visualization written to %s", output_path)
    return 0


def main() -> None:
    args = parse_args(sys.argv[1:])
    exit_code = asyncio.run(run_async(args))
    raise SystemExit(exit_code)


__all__ = ["main", "run_async", "run_visualize", "parse_args"]
