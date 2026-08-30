"""Small, dependency-free dialog policy for the local evaluator's constraint language."""

from __future__ import annotations

import re

from src.contracts import ASK_ATTRIBUTES, DialogResult, SessionState

_MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
_COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange")
_MARKERS = re.compile(r"(?:what matters is:|key requirement is:|what i need is:|i need is:|\bprefer(?:s)?\b)\s*(.+?)(?:\.|$)", re.I)
_OVERRIDE = re.compile(r"\b(?:actually|instead|never mind|ignore my earlier|change of plans|different)\b", re.I)


def _classify(value: str) -> str:
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|under|less than|below)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in _MATERIALS):
        return "material"
    if any(color in lowered for color in _COLORS) or "color" in lowered:
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("style", "fit", "sleeve", "neck", "casual", "formal")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


def _extract(message: str) -> list[tuple[str, str]]:
    match = _MARKERS.search(message)
    if not match:
        return []
    values = [item.strip(" ;,") for item in re.split(r"\s*;\s*", match.group(1))]
    return [(_classify(value), value) for value in values if value]


def update(state: SessionState, user_message: str) -> DialogResult:
    message = str(user_message or "")
    override = bool(_OVERRIDE.search(message))
    slots = {} if override else dict(state.slots or {})
    for field, value in _extract(message):
        slots[field] = value

    query_parts = [message]
    for field, value in slots.items():
        if str(value) not in message:
            query_parts.append(str(value))
    canonical_query = " ".join(query_parts)

    asked = set(state.asked_attributes or [])
    preferred_order = ("material", "color", "size", "style", "feature", "use_case", "budget", "category")
    ask_attribute = next((attribute for attribute in preferred_order if attribute not in asked), None)
    if state.turn >= 10:
        ask_attribute = None
    message_text = "I can narrow these down further if you share one more preference."
    if ask_attribute is None:
        message_text = "Here are the closest matches I found."
    return DialogResult(
        canonical_query=canonical_query,
        ask_attribute=ask_attribute,
        slots=slots,
        message=message_text,
        intent="buy" if any(word in message.lower() for word in ("looking for", "need", "buy")) else "",
        intent_override=override,
    )
