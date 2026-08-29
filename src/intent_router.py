from __future__ import annotations

from src.message_parser import MessageParser, load_catalog_vocab

_categories, _brands = load_catalog_vocab("data/catalog.jsonl")
_parser = MessageParser(known_categories=_categories, known_brands=_brands)


def detect_scenario(message: str, history: list[dict] | None = None) -> str:
    """
    Returns one of: 'buying', 'browsing', 'intent_override', 'boundary'
    """
    parsed = _parser.parse(message)

    if parsed.is_override:
        return "intent_override"
    if parsed.is_no_preference:
        return "boundary"
    if parsed.is_vague:
        return "browsing"

    structured = {k: v for k, v in parsed.attributes.items() if k != "feature"}
    return "buying" if structured else "browsing"


def extract_attributes(message: str) -> dict:
    """
    Extract structured attributes from a user message.
    Returns a dict with keys matching KIV fields where detected.
    """
    parsed = _parser.parse(message)
    return {k: v for k, v in parsed.attributes.items() if k != "feature"}
