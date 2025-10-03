from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from .answer import SynthesisResult, synthesize_answer
from .config import load_settings
from .pipeline import ProcessedPage, gather_pages
from .report import build_json_record, render_report

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
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


async def run_async(args: argparse.Namespace) -> int:
    _configure_logging(args.verbose)

    settings = load_settings()
    if args.output_dir:
        settings.output_dir = args.output_dir.expanduser()

    urls = _read_urls(args.urls, args.urls_file)
    if not urls:
        logger.error("At least one URL must be provided via --url or --urls-file")
        return 2

    results: list[ProcessedPage] = []
    synthesis: SynthesisResult | None = None

    async with httpx.AsyncClient(
        headers={"User-Agent": settings.user_agent}
    ) as http_client:
        llm_client = AsyncOpenAI(timeout=settings.openai_timeout)
        try:
            results = await gather_pages(
                urls,
                args.prompt,
                settings=settings,
                http_client=http_client,
                llm_client=llm_client,
            )
            if results:
                synthesis = await synthesize_answer(
                    llm_client,
                    settings,
                    args.prompt,
                    results,
                )
        finally:
            close = getattr(llm_client, "close", None)
            if callable(close):
                maybe_coro = close()
                if asyncio.iscoroutine(maybe_coro):
                    await maybe_coro

    if not results:
        logger.error("No pages processed successfully")
        return 1

    separator = "\n" + "=" * 80 + "\n"
    sections: list[str] = []
    ordered_results: list[ProcessedPage] = list(results)

    if synthesis:
        prioritized = list(synthesis.ordered_pages)
        prioritized_ids = {id(item) for item in prioritized}
        remaining = [item for item in results if id(item) not in prioritized_ids]
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

    sections.extend(
        render_report(item.page, item.assessment) for item in ordered_results
    )

    print(separator.join(sections))

    if args.json_output:
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
        }
        json_path = args.json_output.expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Wrote JSON summary to %s", json_path)

    return 0


def main() -> None:
    args = parse_args(sys.argv[1:])
    exit_code = asyncio.run(run_async(args))
    raise SystemExit(exit_code)


__all__ = ["main", "run_async"]
