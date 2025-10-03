from __future__ import annotations

import asyncio
import logging

import httpx

from .config import Settings

logger = logging.getLogger(__name__)


async def fetch_html(
    url: str,
    client: httpx.AsyncClient,
    settings: Settings,
    *,
    semaphore: asyncio.Semaphore | None = None,
) -> str | None:
    """Fetch a URL as HTML text with retries and basic content validation."""

    attempt = 0
    max_attempts = settings.request_retries + 1
    backoff = settings.request_backoff

    while attempt < max_attempts:
        attempt += 1
        try:
            logger.info("Fetching %s (attempt %s/%s)", url, attempt, max_attempts)
            if semaphore:
                async with semaphore:
                    response = await client.get(url, timeout=settings.request_timeout)
            else:
                response = await client.get(url, timeout=settings.request_timeout)

            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if (
                "text/html" not in content_type
                and "application/xhtml+xml" not in content_type
            ):
                logger.warning(
                    "Skipping %s due to unsupported content-type: %s", url, content_type
                )
                return None
            return response.text
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "HTTP error fetching %s (attempt %s/%s): %s",
                url,
                attempt,
                max_attempts,
                exc,
            )
            if 400 <= exc.response.status_code < 500:
                # Do not retry client errors except 429
                if exc.response.status_code != 429:
                    return None
        except httpx.RequestError as exc:
            logger.warning(
                "Request error fetching %s (attempt %s/%s): %s",
                url,
                attempt,
                max_attempts,
                exc,
            )

        if attempt < max_attempts:
            await asyncio.sleep(backoff if backoff > 0 else 0)
            backoff *= 2

    logger.error("Failed to fetch %s after %s attempts", url, max_attempts)
    return None


__all__ = ["fetch_html"]
