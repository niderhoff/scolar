from __future__ import annotations

from typing import Dict

from .models import PageAssessment, PageContent


def render_report(page: PageContent, assessment: PageAssessment) -> str:
    lines = [
        f"# {page.title}",
        f"Source: {page.url}",
        "",
        "## Summary",
        assessment.summary,
        "",
        "## Technical Depth",
        f"Rating: {assessment.technical_depth.rating}/5",
        assessment.technical_depth.justification,
        "",
        "## Prompt Fit",
        f"Rating: {assessment.prompt_fit.rating}/5",
        assessment.prompt_fit.justification,
    ]

    if assessment.recommended_links:
        lines.append("")
        lines.append("## Recommended Follow-up Links")
        for link in assessment.recommended_links:
            lines.append(f"- [{link.title}]({link.url}): {link.reason}")

    return "\n".join(lines)


def build_json_record(
    page: PageContent, assessment: PageAssessment
) -> Dict[str, object]:
    """Return a JSON-serializable dictionary representing the processed page."""

    return {
        "title": page.title,
        "url": page.url,
        "markdown_path": str(page.markdown_path) if page.markdown_path else None,
        "truncated": page.truncated,
        "outbound_links": [
            {"title": link.title, "url": link.url} for link in page.links
        ],
        "summary": assessment.summary,
        "technical_depth": {
            "rating": assessment.technical_depth.rating,
            "justification": assessment.technical_depth.justification,
        },
        "prompt_fit": {
            "rating": assessment.prompt_fit.rating,
            "justification": assessment.prompt_fit.justification,
        },
        "recommended_links": [
            {
                "title": link.title,
                "url": link.url,
                "reason": link.reason,
            }
            for link in assessment.recommended_links
        ],
    }


__all__ = ["render_report", "build_json_record"]
