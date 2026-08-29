"""Raw customer message -> BM25 keywords + structured attributes + signals.
No session state; instantiate once and reuse across sessions.

    categories, brands = load_catalog_vocab("data/catalog.jsonl")
    parser = MessageParser(known_categories=categories, known_brands=brands)
    parsed = parser.parse("black leather boots, size 9, under $80")
    parsed.attributes   # {'material': 'leather', 'color': 'black', 'size': '9', ...}
    parsed.keywords      # BM25 query terms
    parsed.is_override / .is_no_preference / .is_vague
"""

from .catalog_vocab import load_catalog_vocab
from .parser import MessageParser, ParsedMessage
from .vocab import ALLOWED_ATTRIBUTES

__all__ = [
    "MessageParser",
    "ParsedMessage",
    "load_catalog_vocab",
    "ALLOWED_ATTRIBUTES",
]
