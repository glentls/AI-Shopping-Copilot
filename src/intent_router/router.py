from __future__ import annotations

from src.message_parser import LLMMessageParser, ParsedMessage, load_catalog_vocab

_categories, _brands = load_catalog_vocab("data/catalog.jsonl")
_parser = LLMMessageParser(known_categories=_categories, known_brands=_brands)


def parse_message(message: str) -> ParsedMessage:
    """Parse once and return the full ParsedMessage (intent, category, product, attributes)."""
    return _parser.parse(message)


def detect_scenario(message: str, history: list[dict] | None = None) -> str:
    """
    Returns one of: 'buying', 'browsing', 'intent_override', 'boundary'
    """
    return parse_message(message).intent


def extract_attributes(message: str) -> dict:
    """
    Extract structured attributes from a user message.
    Returns a dict with keys matching KIV fields where detected.
    """
    parsed = parse_message(message)
    return {k: v for k, v in parsed.attributes.items() if k != "feature"}


def build_search_key(session: dict) -> dict:
    search_key: dict = {}
    for attr, values in session["constraints"].items():
        search_key[attr] = values
    price_c = session.get("price_constraint")
    if price_c:
        if price_c["operator"] in ("<", "<=", "~"):
            search_key["price"] = [{"lte": price_c["amount"]}]
        else:
            search_key["price"] = [{"gte": price_c["amount"]}]
    return search_key
