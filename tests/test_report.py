"""Unit tests for scolar.report helpers."""

from __future__ import annotations

from scolar.models import (
    LinkInfo,
    PageAssessment,
    PageContent,
    RecommendedLink,
    Score,
)
from scolar.report import build_json_record


def test_build_json_record_serializes_page_and_assessment(tmp_path):
    """JSON record should include page metadata, scores, and link details."""

    page = PageContent(
        url="https://example.com",
        title="Example",
        markdown="Content",
        links=[LinkInfo(title="More", url="https://example.com/more")],
        truncated=False,
    )
    page.markdown_path = tmp_path / "example.md"
    page.markdown_path.write_text("Content", encoding="utf-8")

    assessment = PageAssessment(
        summary="Summary",
        technical_depth=Score(rating=4, justification="Deep"),
        prompt_fit=Score(rating=5, justification="Great"),
        recommended_links=[
            RecommendedLink(
                title="Follow", url="https://example.com/follow", reason="Reason"
            ),
        ],
    )

    record = build_json_record(page, assessment)

    assert record["title"] == "Example"
    assert record["url"] == "https://example.com"
    assert record["markdown_path"].endswith("example.md")
    assert record["technical_depth"]["rating"] == 4
    assert record["recommended_links"][0]["title"] == "Follow"
    assert record["outbound_links"][0]["url"] == "https://example.com/more"
