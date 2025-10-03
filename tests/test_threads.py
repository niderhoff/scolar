from __future__ import annotations

from scolar.fetcher import RedditComment, RedditThread
from scolar.threads import clean_html_content, convert_to_thread_path


def test_clean_html_content_strips_markup() -> None:
    html = "<div>First &amp; second<br>line</div>"
    cleaned = clean_html_content(html)
    assert cleaned == "First & second line"


def test_convert_to_thread_path_nested_comments() -> None:
    thread = RedditThread(
        identifier="abc",
        url="https://www.reddit.com/r/test/comments/abc/thread/",
        title="Thread Title",
        author="op_user",
        body_html="<p>OP Body</p>",
        score=42,
        comments=[
            RedditComment(
                identifier="c1",
                author="user1",
                body_html="<p>Comment 1</p>",
                score=10,
                children=[
                    RedditComment(
                        identifier="c2",
                        author="user2",
                        body_html="<p>Reply <em>text</em></p>",
                        score=5,
                        children=[],
                    )
                ],
            )
        ],
    )

    paths = convert_to_thread_path(thread)

    assert paths == [
        "[1] op_user: Thread Title - OP Body",
        "[1.1] user1: Comment 1",
        "[1.1.1] user2: Reply text",
    ]
