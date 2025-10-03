from __future__ import annotations

from html import unescape

from bs4 import BeautifulSoup

from .fetcher import RedditComment, RedditThread


def clean_html_content(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    return " ".join(unescape(text).split())


def _append_comment_paths(
    lines: list[str],
    comment: RedditComment,
    path: str,
) -> None:
    author = comment.author or "Anonymous"
    body = clean_html_content(comment.body_html)
    lines.append(f"[{path}] {author}: {body}")

    for index, child in enumerate(comment.children, start=1):
        child_path = f"{path}.{index}"
        _append_comment_paths(lines, child, child_path)


def convert_to_thread_path(thread: RedditThread) -> list[str]:
    lines: list[str] = []
    op_author = thread.author or "Anonymous"
    op_body = clean_html_content(thread.body_html)

    if op_body:
        op_content = f"{thread.title} - {op_body}"
    else:
        op_content = thread.title

    lines.append(f"[1] {op_author}: {op_content}")

    for index, comment in enumerate(thread.comments, start=1):
        _append_comment_paths(lines, comment, f"1.{index}")

    return lines


__all__ = ["clean_html_content", "convert_to_thread_path"]
