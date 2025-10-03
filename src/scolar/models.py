from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, ValidationError


@dataclass(slots=True)
class LinkInfo:
    title: str
    url: str


@dataclass(slots=True)
class PageContent:
    url: str
    title: str
    markdown: str
    links: list[LinkInfo]
    truncated: bool
    markdown_path: Path | None = None


@dataclass(slots=True)
class Score:
    rating: int
    justification: str


@dataclass(slots=True)
class RecommendedLink:
    title: str
    url: str
    reason: str


@dataclass(slots=True)
class PageAssessment:
    summary: str
    technical_depth: Score
    prompt_fit: Score
    recommended_links: list[RecommendedLink]


class ScorePayload(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    justification: str = Field(..., min_length=1)


class RecommendedLinkPayload(BaseModel):
    title: str = Field(..., min_length=1)
    url: HttpUrl
    reason: str = Field(..., min_length=1)


class AssessmentPayload(BaseModel):
    summary: str = Field(..., min_length=1)
    technical_depth: ScorePayload
    prompt_fit: ScorePayload
    recommended_links: list[RecommendedLinkPayload] = Field(default_factory=list)

    @classmethod
    def parse_json(cls, data: str) -> "AssessmentPayload":
        return cls.model_validate_json(data)


def payload_to_assessment(payload: AssessmentPayload) -> PageAssessment:
    return PageAssessment(
        summary=payload.summary,
        technical_depth=Score(
            rating=int(payload.technical_depth.rating),
            justification=payload.technical_depth.justification,
        ),
        prompt_fit=Score(
            rating=int(payload.prompt_fit.rating),
            justification=payload.prompt_fit.justification,
        ),
        recommended_links=[
            RecommendedLink(title=item.title, url=str(item.url), reason=item.reason)
            for item in payload.recommended_links
        ],
    )


__all__ = [
    "LinkInfo",
    "PageContent",
    "Score",
    "RecommendedLink",
    "PageAssessment",
    "AssessmentPayload",
    "payload_to_assessment",
    "ValidationError",
    "page_to_dict",
    "assessment_to_dict",
    "dict_to_page",
    "dict_to_assessment",
]


def page_to_dict(page: PageContent) -> dict[str, object]:
    return {
        "url": page.url,
        "title": page.title,
        "markdown": page.markdown,
        "links": [{"title": link.title, "url": link.url} for link in page.links],
        "truncated": page.truncated,
        "markdown_path": str(page.markdown_path) if page.markdown_path else None,
    }


def assessment_to_dict(assessment: PageAssessment) -> dict[str, object]:
    return {
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


def dict_to_page(payload: Mapping[str, Any]) -> PageContent:
    markdown_path_raw = payload.get("markdown_path")
    markdown_path = Path(markdown_path_raw) if markdown_path_raw else None

    return PageContent(
        url=str(payload["url"]),
        title=str(payload["title"]),
        markdown=str(payload["markdown"]),
        links=[
            LinkInfo(title=str(item["title"]), url=str(item["url"]))
            for item in payload.get("links", [])
        ],
        truncated=bool(payload.get("truncated", False)),
        markdown_path=markdown_path,
    )


def dict_to_assessment(payload: Mapping[str, Any]) -> PageAssessment:
    technical = payload.get("technical_depth", {})
    prompt_fit = payload.get("prompt_fit", {})

    return PageAssessment(
        summary=str(payload.get("summary", "")),
        technical_depth=Score(
            rating=int(technical.get("rating", 0)),
            justification=str(technical.get("justification", "")),
        ),
        prompt_fit=Score(
            rating=int(prompt_fit.get("rating", 0)),
            justification=str(prompt_fit.get("justification", "")),
        ),
        recommended_links=[
            RecommendedLink(
                title=str(item["title"]),
                url=str(item["url"]),
                reason=str(item["reason"]),
            )
            for item in payload.get("recommended_links", [])
        ],
    )
