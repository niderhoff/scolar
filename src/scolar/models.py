from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    markdown_path: Optional[Path] = None


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
]
