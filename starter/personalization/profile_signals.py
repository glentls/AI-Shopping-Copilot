from __future__ import annotations

import re
from dataclasses import dataclass

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "prior", "purchases",
    "that", "the", "this", "to", "with", "you", "emphasize", "ratings", "usually",
    "positive", "critical", "mixed", "summary",
}

TAG_TO_ATTRIBUTE: dict[str, str | None] = {
    "material": "material",
    "fit": "feature",
    "comfort": "feature",
    "durability": "feature",
    "performance": "feature",
    "style": "style",
    "warmth": "use_case",
    "weather": "use_case",
    "general shopping": None,
}

@dataclass(frozen=True)
class ProfileSignals:
    tags: tuple[str, ...]
    preferred_attributes: tuple[str, ...]
    rating_style: str
    average_prior_rating: float | None
    summary_terms: tuple[str, ...]


def parse_profile(profile: dict | None) -> ProfileSignals:
    profile = profile or {}
    tags = tuple(
        dict.fromkeys(
            tag.strip().lower()
            for tag in profile.get("preference_tags", [])
            if isinstance(tag, str) and tag.strip()
        )
    )
    preferred: list[str] = []
    for tag in tags:
        attribute = TAG_TO_ATTRIBUTE.get(tag)
        if attribute and attribute not in preferred:
            preferred.append(attribute)
    rating_style = str(profile.get("rating_style") or "").strip().lower()
    average_prior_rating = _parse_float(profile.get("average_prior_rating"))
    summary_terms = tuple(_terms_from_summary(str(profile.get("summary") or "")))
    return ProfileSignals(
        tags=tags,
        preferred_attributes=tuple(preferred),
        rating_style=rating_style,
        average_prior_rating=average_prior_rating,
        summary_terms=summary_terms,
    )


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _terms_from_summary(summary: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(summary)
        if len(token) > 2 and token.lower() not in STOPWORDS
    ]
