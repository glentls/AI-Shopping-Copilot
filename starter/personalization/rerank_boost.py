from __future__ import annotations

from starter.personalization.profile_signals import ProfileSignals, parse_profile

TAG_MATCH_SCORE = 2.0
FEATURE_TAG_BONUS = 1.0
SUMMARY_TERM_SCORE = 1.0
MAX_PROFILE_BOOST = 2.0


def compute_profile_boost(product: object, profile: dict | ProfileSignals | None) -> float:
    signals = profile if isinstance(profile, ProfileSignals) else parse_profile(profile if isinstance(profile, dict) else None)
    if not signals.tags and not signals.summary_terms:
        return _rating_style_boost(product, signals)

    corpus = str(getattr(product, "searchable_text", "") or "").lower()
    features = str(getattr(product, "features", "") or "").lower()
    boost = 0.0

    for tag in signals.tags:
        if tag in corpus:
            boost += TAG_MATCH_SCORE
            if tag in features:
                boost += FEATURE_TAG_BONUS

    for term in signals.summary_terms:
        if term in corpus:
            boost += SUMMARY_TERM_SCORE

    boost += _rating_style_boost(product, signals)
    return min(boost, MAX_PROFILE_BOOST)


def _rating_style_boost(product: object, signals: ProfileSignals) -> float:
    rating = getattr(product, "average_rating", None)
    if rating is None:
        return 0.0
    try:
        rating_value = float(rating)
    except (TypeError, ValueError):
        return 0.0
    if signals.rating_style == "critical":
        if rating_value >= 4.0:
            return 1.0
        if rating_value < 3.5:
            return -1.0
    if signals.rating_style in ("usually positive", "mixed"):
        if rating_value >= 4.5:
            return 0.5
    if signals.average_prior_rating is not None and signals.average_prior_rating >= 4.0 and rating_value >= 4.0:
        return 0.5
    return 0.0
