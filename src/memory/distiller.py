"""Deterministic, intra-session preference distillation.

The evaluator exposes no stable user identity, so this module deliberately keeps no
persistent state.  It compresses the state that Agent already owns for one session into
the small ``soft_prefs`` structure consumed by retrieval.
"""

from __future__ import annotations

import re
from collections import defaultdict

from src.config import load_config
from src.contracts import MemoryProfile, SessionState

_NEGATIVE_RE = re.compile(
    r"\b(?:do not|is not|don't|isn't|not|no|never|without|avoid|reject(?:ed)?)\b\s+(?:a\s+)?([^.;,!]+)",
    re.IGNORECASE,
)
_VALUE_RE = re.compile(r"(?:what matters is:|key requirement is:|prefer(?:s)?)\s+([^.;!]+)", re.I)
_KNOWN_FIELDS = {"category", "material", "color", "size", "style", "brand", "budget", "feature", "use_case", "store"}


def _values(value: object) -> list[str]:
    if isinstance(value, dict):
        result: list[str] = []
        for nested in value.values():
            result.extend(_values(nested))
        return result
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if value in (None, ""):
        return []
    return [str(value).strip()]


def _add(mapping: defaultdict[str, dict[str, float]], field: str, value: str, weight: float) -> None:
    value = re.sub(r"\s+", " ", value).strip(" -:;,.")
    if field in _KNOWN_FIELDS and value:
        mapping[field].setdefault(value.lower(), 0.0)
        mapping[field][value.lower()] += weight


def _distill(state: SessionState, user_profile: dict, config: dict) -> MemoryProfile:
    cfg = config.get("memory", {})
    positive_weight = float(cfg.get("positive_weight", 1.0))
    profile_weight = float(cfg.get("profile_weight", 0.35))
    negative_weight = float(cfg.get("negative_weight", 1.0))
    max_values = int(cfg.get("max_values_per_field", 8))
    positive: defaultdict[str, dict[str, float]] = defaultdict(dict)
    negative_terms: dict[str, float] = {}

    for field, raw_value in (state.slots or {}).items():
        field_name = str(field).lower()
        for value in _values(raw_value)[:max_values]:
            _add(positive, field_name, value, positive_weight)

    # The anonymized profile is session input, not a cross-session identity. Only fields
    # with useful preference signal are distilled; purchase_frequency is non-discriminating.
    if isinstance(user_profile, dict):
        for value in _values(user_profile.get("preference_tags"))[:max_values]:
            _add(positive, "feature", value, profile_weight)
        for value in _values(user_profile.get("rating_style"))[:max_values]:
            _add(positive, "style", value, profile_weight)

    for entry in state.history or []:
        message = str(entry.get("user_message", "")) if isinstance(entry, dict) else str(entry)
        for match in _NEGATIVE_RE.findall(message):
            value = re.sub(r"\s+", " ", match).strip(" -:;,.").lower()
            value = re.sub(r"^(?:want|like|prefer)\s+", "", value)
            if value and "preference" not in value:
                negative_terms[value] = negative_terms.get(value, 0.0) + negative_weight
        for match in _VALUE_RE.findall(message):
            # Keep the raw phrase as a feature only when it is concise; slot extraction
            # remains the preferred and more precise representation.
            value = re.sub(r"\s+", " ", match).strip(" -:;,." )
            if value and len(value) <= 80:
                _add(positive, "feature", value, positive_weight * 0.5)

    rejected_asins = [str(item) for item in (state.negatives or []) if str(item).strip()]
    boosts: dict[str, dict[str, float] | list[str]] = {
        field: {value: round(weight, 4) for value, weight in values.items()}
        for field, values in positive.items()
        if values
    }
    if negative_terms:
        boosts["negative_terms"] = {value: round(-weight, 4) for value, weight in negative_terms.items()}
    if rejected_asins:
        boosts["rejected_asins"] = rejected_asins[: int(cfg.get("max_rejected_asins", 20))]

    summary_parts = []
    for field, values in positive.items():
        summary_parts.append(f"{field}: {', '.join(values)}")
    if negative_terms:
        summary_parts.append("avoid: " + ", ".join(negative_terms))
    if rejected_asins:
        summary_parts.append(f"rejected products: {len(rejected_asins)}")
    summary = "; ".join(summary_parts)[: int(cfg.get("summary_max_chars", 500))]
    return MemoryProfile(boosts=boosts, summary=summary)


def distill(state: SessionState, user_profile: dict) -> MemoryProfile:
    """Return safe session memory; malformed component inputs become empty memory."""
    try:
        return _distill(state, user_profile if isinstance(user_profile, dict) else {}, load_config())
    except Exception:
        return MemoryProfile(boosts={}, summary="")
