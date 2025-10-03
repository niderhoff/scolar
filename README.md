# Scolar

Scolar is an asynchronous research helper that fetches a list of web pages, extracts their text content, saves Markdown snapshots, and asks an OpenAI model to summarize each page. For every URL the tool reports:

- an 80–120 word neutral summary
- technical-depth and prompt-fit ratings (1–5) plus justifications
- recommended follow-up links pulled from the page’s outbound references
- the local Markdown artifact path for offline review

You can optionally export the complete results as JSON to feed into downstream workflows.

## Prerequisites

- Python 3.13 (managed automatically via `uv`)
- An OpenAI API key with access to the configured model (`SCOLAR_OPENAI_MODEL`, default `gpt-4.1-mini`)
- Network access to the target URLs

## Installation

The project uses [uv](https://github.com/astral-sh/uv) for dependency management. To install the runtime and development dependencies:

```bash
uv sync
```

This creates a `.venv` virtual environment pinned to Python 3.13 and installs the package in editable mode plus test/lint tooling (`pytest`, `pytest-asyncio`, `pre-commit`).

## Configuration

Runtime settings are managed via [Dynaconf](https://www.dynaconf.com/). Default values live in `settings.toml`; any `SCOLAR_*` environment variable overrides a field (e.g., `SCOLAR_FETCH_CONCURRENCY=10`). Key options include:

- `fetch_concurrency`: parallel HTTP requests (default 5)
- `llm_concurrency`: simultaneous OpenAI calls (default 1)
- `max_markdown_chars`: truncation threshold for page text
- `output_dir`: relative/absolute path for Markdown artifacts
- `openai_model`: model name supplied to the OpenAI Responses API

## Usage

Provide at least one URL and a research prompt. Example:

```bash
uv run scolar \
  --prompt "Design considerations for production RAG systems" \
  --url https://example.com/article
```

Options:

- `--prompt`: research question/context (required)
- `--url`: page to include; repeat for multiple URLs
- `--urls-file`: newline-delimited file of URLs (merged with `--url` entries)
- `--output-dir`: override Markdown artifact directory (defaults to `artifacts/`)
- `--json-output`: path to write a structured JSON report of all processed pages
- `--verbose`: enable debug logging

Example with JSON export:

```bash
uv run scolar \
  --prompt "LLM guardrail techniques" \
  --url https://example.com/blog/guardrails \
  --json-output reports/guardrails.json
```

### Output

The CLI prints a Markdown overview for each page, separated by `====================`. JSON output (when requested) includes the prompt, the rendered assessments, outbound links, and Markdown artifact paths.

## Development

### Testing

```bash
uv run pytest
```

The test suite covers configuration precedence, async pipeline behavior with mocked HTTP/LLM clients, CLI JSON output, and serialization helpers.

### Pre-commit Hooks (Recommended for Linting & Formatting)

Install Git hooks so Ruff linting/formatting and mypy type checks run automatically before each commit:

```bash
uv run pre-commit install
```

Run the hooks manually when needed to ensure consistency:

```bash
uv run pre-commit run --all-files
```

Need a one-off type check without the hook runner?

```bash
uv run mypy src tests
```

By relying on pre-commit, everyone runs the same Ruff and mypy checks with consistent versions, avoiding local drift.

## Environment Variables

Set the OpenAI API key before running:

```bash
export OPENAI_API_KEY=sk-...
```

For local experiments, override any setting via environment variables or by providing alternate Dynaconf settings files (e.g., `SCOLAR_SETTINGS_FILES="prod.toml,local.toml"`).

## License

Copyright (c) 2025 Nicolas Iderhoff. All rights reserved. No license is granted for redistribution or derivative works without explicit permission.
