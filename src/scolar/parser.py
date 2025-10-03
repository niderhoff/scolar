from __future__ import annotations

import re
from bs4 import BeautifulSoup
import html2text
from urllib.parse import urljoin

from .config import Settings
from .models import LinkInfo, PageContent


def _sanitize_markdown(markdown: str) -> str:
    # Collapse repeated blank lines for readability.
    return re.sub(r"\n{3,}", "\n\n", markdown).strip()


def parse_html(url: str, html: str, settings: Settings) -> PageContent:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "img", "video", "source"]):
        tag.decompose()

    title = url
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    links: list[LinkInfo] = []
    seen: set[tuple[str, str]] = set()
    for link in soup.select("a[href]"):
        if len(links) >= settings.max_links_inspected:
            break
        href = link.get("href")
        if not href:
            continue
        absolute = urljoin(url, href.strip())
        if not absolute.startswith(("http://", "https://")):
            continue
        text = " ".join(link.get_text(" ", strip=True).split()) or absolute
        key = (text, absolute)
        if key in seen:
            continue
        seen.add(key)
        links.append(LinkInfo(title=text, url=absolute))

    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.ignore_images = True
    converter.body_width = 0

    markdown = converter.handle(str(soup.body or soup)).strip()
    markdown = _sanitize_markdown(markdown)

    truncated = False
    if len(markdown) > settings.max_markdown_chars:
        markdown = markdown[: settings.max_markdown_chars].rsplit("\n", 1)[0]
        truncated = True

    return PageContent(
        url=url,
        title=title or url,
        markdown=markdown,
        links=links,
        truncated=truncated,
    )


__all__ = ["parse_html"]
