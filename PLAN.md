# Asynchronous Scolar Plan

## Architecture Overview

- [x] Orchestrate tasks with `asyncio`; main entry creates a queue of URLs, awaits per-page tasks, aggregates reports.
- [x] Use a shared `httpx.AsyncClient` with connection pooling and an `asyncio.Semaphore` to cap concurrency.
- [x] Offload synchronous HTML parsing and markdown conversion via `asyncio.to_thread` to avoid blocking the event loop.
- [x] Execute OpenAI requests with the async client, keeping them sequential by default and guarding with a semaphore if future parallelism is desired.
- [x] Persist markdown artifacts using `asyncio.to_thread`; reporting can remain synchronous once data is collected.

## Key Components

- [x] `config.py`: Dynaconf-backed configuration validated with Pydantic, exposing concurrency limits, timeouts, model settings, and output paths.
- [x] `models.py`: Dataclasses (`PageContent`, `PageAssessment`, `LinkInfo`) and helpers for JSON validation.
- [x] `fetcher.py`: `async fetch(url, session)` with retries, content-type checks, and size guardrails.
- [x] `parser.py`: Synchronous HTML→markdown + link extraction invoked through `to_thread`.
- [x] `storage.py`: Async helpers to write markdown files and return artifact paths.
- [x] `summarizer.py` (LLM assessment helpers): Async wrapper around the OpenAI client enforcing JSON schema checks.
- [x] `report.py`: Markdown and JSON rendering helpers for processed pages.
- [x] `main.py`: CLI handling for markdown output, optional JSON export, and wiring of async pipeline components.

## Data Flow

1. [x] `main` seeds URL jobs.
2. [x] `fetcher.fetch` concurrently retrieves HTML and records failures.
3. [x] `parser.parse_html` returns `PageContent` with markdown and outbound links.
4. [x] `storage.store_markdown` persists markdown artifacts and tracks truncation.
5. [x] `summarizer.assess_page` invokes the LLM for summaries and ratings.
6. [x] `report` emits markdown and JSON structures for downstream consumption.

## Dependencies & Settings

- [x] Libraries: `httpx[http2]`, `beautifulsoup4`, `html2text`, `openai`, `pydantic`, `dynaconf`, test deps `pytest`, `pytest-asyncio` (optional `tenacity` for retries).
- [x] Configuration: Dynaconf `settings.toml` + environment overrides validated through Pydantic and CLI flags for concurrency, model name, truncation limits, and output directory.
- [x] Logging: `logging.basicConfig` currently used; structured logging via `dictConfig` remains future work.

## Testing

- [x] Configuration module coverage via pytest (`tests/test_config.py`) ensuring type safety, precedence, and path handling.
- [x] Async pipeline integration tests with mocked HTTP and LLM clients (`tests/test_integration.py`).
- [x] Report serialization unit test (`tests/test_report.py`).
- [x] CLI-level test verifying markdown and JSON outputs with mocked dependencies (`tests/test_cli.py`).
- [x] End-to-end smoke equivalent via CLI run_async entry, covering the same code path as `uv run scolar`.

## Recent Updates

- [x] Eliminated `typing.Any` usages in configuration and integration tests to tighten static type coverage.
- [x] Migrated legacy generics (`Dict`, `List`, `Optional`) to builtin `dict`, `list`, and union syntax across runtime and tests.

## Next Steps

1. [ ] Add optional structured logging configuration if deeper observability is required.
2. [ ] Consider snapshot-based assertions for large markdown outputs once additional report sections are introduced.
3. [ ] Evaluate adding a static type checker (e.g., mypy or pyright) to enforce stricter annotations in CI.
4. [ ] Audit remaining modules for `collections.abc` usage to ensure consistency with modern type hints.
