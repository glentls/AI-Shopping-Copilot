from __future__ import annotations

import os

from src.message_parser import MessageParser, ParsedMessage, load_catalog_vocab

# Honour the evaluator's --catalog rather than hard-coding the path. The loader
# is lru_cached by path, so this shares the parse with every other consumer.
_CATALOG_PATH = os.environ.get("CATALOG_PATH", "data/catalog.jsonl")
_parser: MessageParser | None = None


def _get_parser() -> MessageParser:
    global _parser
    if _parser is None:
        categories, brands = load_catalog_vocab(_CATALOG_PATH)
        _parser = MessageParser(known_categories=categories, known_brands=brands)
    return _parser


def parse_message(message: str) -> ParsedMessage:
    """Parse once and return the full ParsedMessage (intent, category, product, attributes)."""
    return _get_parser().parse(message)


def detect_scenario(message: str, history: list[dict] | None = None) -> str:
    """
    Returns one of: 'buying', 'browsing', 'intent_override', 'boundary'
    """
    return parse_message(message).intent


def extract_attributes(message: str) -> dict:
    """
    Extract structured attributes from a user message.
    Returns a dict with keys matching KIV fields where detected.

    ``feature`` is still filtered here for the legacy BM25 path (keeping it
    byte-identical to the recorded 0.680 control). The bucket pipeline does not
    consume these taxonomy attributes at all -- it ranks against the verbatim
    ConstraintMemory, which is exactly where the dropped ``feature`` strings are
    now retained (see src/intent_router/constraint_memory.py).
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
