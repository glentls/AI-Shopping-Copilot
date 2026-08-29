from __future__ import annotations

OVERLOADED_THRESHOLD = 500

ATTRIBUTE_PRIORITY = (
    "category",
    "use_case",
    "budget",
    "size",
    "color",
    "material",
    "style",
    "brand",
    "feature",
)

SLOT_KEY_ALIASES = {
    "colour": "color",
    "use case": "use_case",
}


def normalize_attribute_name(name: str) -> str:
    lowered = name.strip().lower()
    return SLOT_KEY_ALIASES.get(lowered, lowered)


def build_retrieval_feedback(
    candidate_count: int,
    filters: dict,
    *,
    relaxed_search: bool = False,
    overloaded_threshold: int = OVERLOADED_THRESHOLD,
) -> dict[str, object]:
    missing = missing_attributes(filters)
    overloaded = candidate_count > overloaded_threshold
    if overloaded:
        missing = prioritize_missing_attributes(missing, candidate_count)
    feedback: dict[str, object] = {
        "candidate_count": candidate_count,
        "overloaded": overloaded,
        "missing_attributes": missing,
    }
    if relaxed_search:
        feedback["relaxed_search"] = True
    return feedback


def missing_attributes(filters: dict) -> list[str]:
    unconstrained = _attribute_set(filters.get("unconstrained"))
    asked = _attribute_set(filters.get("asked"))
    slot_status = _normalized_slot_status(filters.get("slot_status"))

    missing: list[str] = []
    for attribute in ATTRIBUTE_PRIORITY:
        if attribute in unconstrained or attribute in asked:
            continue
        status = slot_status.get(attribute)
        if status == "unconstrained":
            continue
        if _attribute_is_filled(filters, attribute) or status == "confirmed":
            continue
        missing.append(attribute)
    return missing


def prioritize_missing_attributes(missing: list[str], candidate_count: int) -> list[str]:
    if candidate_count <= OVERLOADED_THRESHOLD:
        return missing
    priority_index = {attribute: index for index, attribute in enumerate(ATTRIBUTE_PRIORITY)}
    return sorted(missing, key=lambda attribute: priority_index.get(attribute, len(ATTRIBUTE_PRIORITY)))


def _attribute_is_filled(filters: dict, attribute: str) -> bool:
    if attribute == "category":
        return bool(filters.get("category") or filters.get("category_tokens"))
    if attribute == "budget":
        return filters.get("max_price") is not None or bool(filters.get("budget"))
    return bool(filters.get(attribute))


def _attribute_set(raw: object) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, (set, frozenset)):
        items = raw
    elif isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    return {normalize_attribute_name(str(item)) for item in items if str(item).strip()}


def _normalized_slot_status(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, value in raw.items():
        normalized[normalize_attribute_name(str(key))] = str(value).strip().lower()
    return normalized
