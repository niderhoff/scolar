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
- [x] `discovery.py`: Async candidate URL discovery backed by cached subreddit search for `r/localllama`.

## Data Flow

1. [x] `main` seeds URL jobs.
2. [x] `fetcher.fetch` concurrently retrieves HTML and records failures.
3. [x] `parser.parse_html` returns `PageContent` with markdown and outbound links.
4. [x] `storage.store_markdown` persists markdown artifacts and tracks truncation.
5. [x] `summarizer.assess_page` invokes the LLM for summaries and ratings.
6. [x] `report` emits markdown and JSON structures for downstream consumption.

## Dependencies & Settings

- [x] Libraries: `httpx[http2]`, `beautifulsoup4`, `html2text`, `openai`, `pydantic`, `dynaconf`, `llama-index-core`, `llama-index-utils-workflow`, test deps `pytest`, `pytest-asyncio` (optional `tenacity` for retries).
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
- [x] Added GitHub Actions workflow that syncs dependencies with `uv` and runs `ruff check` plus `mypy` on pushes and pull requests.
- [x] Extended CI workflow to execute `uv run pytest` alongside lint and type checks.
- [x] Verified `_build_slug` hash suffix logic and ensured duplicate-title collision coverage via `tests/test_storage.py` on October 3, 2025.
- [x] Added a synthesis stage that orders assessed pages by prompt fit and technical depth, then prompts the LLM for a sourced final answer (October 3, 2025).
- [x] Ensured console and JSON outputs list pages in the same relevance order used during final answer synthesis (October 3, 2025).
- [x] Implemented a three-day page cache with an optional CLI refresh flag to skip cache hits when required (October 3, 2025).
- [x] Added an LLM-powered search query generator with a `--suggest-queries` CLI flag and JSON export (October 3, 2025).
- [x] Refactored the main CLI orchestration to run on LlamaIndex Workflows, keeping reporting behaviour identical while enabling future step fan-out (October 3, 2025).
- [x] Added `llama-index-core` (and workflow runtime dependencies) via `uv add` to support the new orchestrator (October 3, 2025).
- [x] Added `llama-index-utils-workflow` and helper to render the research workflow graph to `workflow.html` via `draw_all_possible_flows` (October 3, 2025).
- [x] Introduced a `visualize-workflow` CLI command that emits the HTML diagram using the new helper (October 3, 2025).
- [x] Instrumented each workflow step with logging to surface event order and execution details before final output (October 3, 2025).
- [x] Introduced a discovery workflow step that queries `r/localllama`, caches search hits for three days, and feeds discovered URLs into page gathering when none are provided by the user (October 3, 2025).

## Next Steps

1. [ ] Add optional structured logging configuration if deeper observability is required.
2. [ ] Consider snapshot-based assertions for large markdown outputs once additional report sections are introduced.
3. [x] Evaluate adding a static type checker (e.g., mypy or pyright) to enforce stricter annotations in CI (confirmed mypy already runs via CI workflow and pre-commit).
4. [x] Audit remaining modules for `collections.abc` usage to ensure consistency with modern type hints (confirmed only `main.py` and integration tests import `Iterable` from `collections.abc`).
5. [ ] Expose cache TTL and artifact directory overrides via CLI flags to support varied research horizons and storage layouts.
6. [ ] Add cache hit/miss metrics and periodic cleanup of expired entries for long-lived research runs.
7. [x] Wire the generated search query plan into an automated discovery step that populates candidate URLs before assessment (completed October 3, 2025).
8. [ ] Surface workflow progress/telemetry (e.g., streaming events or verbose step logging) to improve long-running run visibility.
9. [x] Remove unused `FetchError` exception in src/scolar/fetcher.py once fetcher either raises or logs its own errors more explicitly (completed October 3, 2025).
10. [ ] Expand discovery sources beyond Reddit (e.g., curated blogs or broader web search) while reusing caching semantics and respecting site-specific query parameters.
11. [x] Expose the workflow visualization helper via CLI command and add regression coverage that the HTML artifact is generated (October 3, 2025).
12. [ ] Document the visualization workflow command usage in README and consider adding sample output.

## New Features

- [x] add answer step that will actually answer the prompt given the files ordered by relevancy and technical depth.
- [ ] modify the answer synthesis with another additional step that will list potential followup research questions, depending on the context it has seen (derived from things that were mentioned in the webpages).
- [ ] between crawling links and answer step I want the agent to preprocess each link by summarizing and collecting all different opinions and facts in each page (if they relate to the question).
- [x] add a search query generator that will given the prompt create various search queries that help discover links to answer the question.
- [x] add a Reddit-backed URL discovery stage that automatically seeds the workflow when no URLs are supplied (October 3, 2025).
- [ ] create a crawler that will crawl outbound links and link them in a knowledge graph; the knowledge graph should then be able to be traversed for the summary step at the end to find out which pages relate well to question & follow up questions.
