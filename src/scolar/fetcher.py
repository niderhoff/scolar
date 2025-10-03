from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from json import JSONDecodeError
from urllib.parse import ParseResult, urlparse, urlunparse

import httpx

from .config import Settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HtmlDocument:
    url: str
    html: str


@dataclass(slots=True)
class RedditComment:
    identifier: str
    author: str | None
    body_html: str
    score: int | None
    children: list["RedditComment"]


@dataclass(slots=True)
class RedditThread:
    identifier: str
    url: str
    title: str
    author: str | None
    body_html: str
    score: int | None
    comments: list[RedditComment]


FetchResult = HtmlDocument | RedditThread


async def _get_with_retries(
    url: str,
    client: httpx.AsyncClient,
    settings: Settings,
    *,
    semaphore: asyncio.Semaphore | None = None,
) -> httpx.Response | None:
    attempt = 0
    max_attempts = settings.request_retries + 1
    backoff = settings.request_backoff

    while attempt < max_attempts:
        attempt += 1
        try:
            logger.info("Fetching %s (attempt %s/%s)", url, attempt, max_attempts)
            headers = {"User-Agent": settings.user_agent}
            if semaphore:
                async with semaphore:
                    response = await client.get(
                        url,
                        headers=headers,
                        timeout=settings.request_timeout,
                    )
            else:
                response = await client.get(
                    url,
                    headers=headers,
                    timeout=settings.request_timeout,
                )

            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "HTTP error fetching %s (attempt %s/%s): %s",
                url,
                attempt,
                max_attempts,
                exc,
            )
            status = exc.response.status_code
            if 400 <= status < 500 and status != 429:
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


async def fetch_html(
    url: str,
    client: httpx.AsyncClient,
    settings: Settings,
    *,
    semaphore: asyncio.Semaphore | None = None,
) -> str | None:
    """Fetch a URL as HTML text with retries and basic content validation."""

    response = await _get_with_retries(
        url,
        client,
        settings,
        semaphore=semaphore,
    )
    if response is None:
        return None

    content_type = response.headers.get("content-type", "").lower()
    if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
        logger.warning(
            "Skipping %s due to unsupported content-type: %s", url, content_type
        )
        return None

    return response.text


def _is_reddit_url(url: str) -> bool:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    return hostname.endswith("reddit.com")


def _normalize_reddit_json_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or "/"
    if not path.endswith("/"):
        path = f"{path}/"
    if not path.endswith(".json"):
        path = f"{path}.json"

    sanitized = ParseResult(
        scheme=parsed.scheme or "https",
        netloc=parsed.netloc,
        path=path,
        params="",
        query="",
        fragment="",
    )
    return urlunparse(sanitized)


def _parse_reddit_comment(payload: Mapping[str, object]) -> RedditComment:
    identifier = str(payload.get("id", ""))
    author_value = payload.get("author")
    author = (
        str(author_value) if isinstance(author_value, str) and author_value else None
    )

    score_value = payload.get("score")
    score = int(score_value) if isinstance(score_value, int) else None

    body_html_raw = payload.get("body_html")
    body_html = str(body_html_raw) if isinstance(body_html_raw, str) else ""

    children: list[RedditComment] = []
    replies = payload.get("replies")
    if isinstance(replies, Mapping):
        data = replies.get("data")
        if isinstance(data, Mapping):
            raw_children = data.get("children")
            if isinstance(raw_children, list):
                for entry in raw_children:
                    if not isinstance(entry, Mapping):
                        continue
                    kind = entry.get("kind")
                    if kind != "t1":
                        continue
                    comment_data = entry.get("data")
                    if isinstance(comment_data, Mapping):
                        children.append(_parse_reddit_comment(comment_data))

    return RedditComment(
        identifier=identifier,
        author=author,
        body_html=body_html,
        score=score,
        children=children,
    )


async def fetch_reddit(
    url: str,
    client: httpx.AsyncClient,
    settings: Settings,
    *,
    semaphore: asyncio.Semaphore | None = None,
) -> RedditThread | None:
    json_url = _normalize_reddit_json_url(url)
    response = await _get_with_retries(
        json_url,
        client,
        settings,
        semaphore=semaphore,
    )
    if response is None:
        return None

    try:
        payload = response.json()
    except (JSONDecodeError, ValueError) as exc:
        logger.error("Failed to decode Reddit JSON for %s: %s", url, exc)
        return None

    if not isinstance(payload, list) or len(payload) < 2:
        logger.error("Unexpected Reddit response structure for %s", url)
        return None

    post_listing = payload[0]
    comments_listing = payload[1]
    if not isinstance(post_listing, Mapping) or not isinstance(
        comments_listing, Mapping
    ):
        logger.error("Unexpected Reddit listing format for %s", url)
        return None

    post_children = post_listing.get("data")
    if not isinstance(post_children, Mapping):
        logger.error("Missing Reddit post data for %s", url)
        return None
    post_items = post_children.get("children")
    if not isinstance(post_items, list) or not post_items:
        logger.error("Empty Reddit post listing for %s", url)
        return None
    first_item = post_items[0]
    if not isinstance(first_item, Mapping):
        logger.error("Invalid Reddit post entry for %s", url)
        return None
    post_data = first_item.get("data")
    if not isinstance(post_data, Mapping):
        logger.error("Missing Reddit post details for %s", url)
        return None

    identifier = str(post_data.get("id", ""))
    author_value = post_data.get("author")
    author = (
        str(author_value) if isinstance(author_value, str) and author_value else None
    )
    title_value = post_data.get("title")
    title = str(title_value) if isinstance(title_value, str) else url

    body_html_raw = post_data.get("selftext_html")
    body_html = str(body_html_raw) if isinstance(body_html_raw, str) else ""

    score_value = post_data.get("score")
    score = int(score_value) if isinstance(score_value, int) else None

    comments: list[RedditComment] = []
    comments_data = comments_listing.get("data")
    if isinstance(comments_data, Mapping):
        comment_children = comments_data.get("children")
        if isinstance(comment_children, list):
            for entry in comment_children:
                if not isinstance(entry, Mapping):
                    continue
                if entry.get("kind") != "t1":
                    continue
                comment_payload = entry.get("data")
                if isinstance(comment_payload, Mapping):
                    comments.append(_parse_reddit_comment(comment_payload))

    return RedditThread(
        identifier=identifier,
        url=url,
        title=title,
        author=author,
        body_html=body_html,
        score=score,
        comments=comments,
    )


async def fetch_resource(
    url: str,
    client: httpx.AsyncClient,
    settings: Settings,
    *,
    semaphore: asyncio.Semaphore | None = None,
) -> FetchResult | None:
    if _is_reddit_url(url):
        thread = await fetch_reddit(
            url,
            client,
            settings,
            semaphore=semaphore,
        )
        if thread:
            return thread
        logger.info(
            "Falling back to HTML fetch for %s after Reddit parsing failure", url
        )

    html = await fetch_html(
        url,
        client,
        settings,
        semaphore=semaphore,
    )
    if not html:
        return None

    return HtmlDocument(url=url, html=html)


__all__ = [
    "HtmlDocument",
    "RedditComment",
    "RedditThread",
    "FetchResult",
    "fetch_html",
    "fetch_reddit",
    "fetch_resource",
]
