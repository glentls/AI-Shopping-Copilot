"""Interactive REPL to try the parser on your own text.

Run from the repo root:
    python3 -m src.message_parser.try_it

Type a message, see the extraction, repeat. Ctrl+D or "quit" to exit.
"""

from __future__ import annotations

import json
import sys

from .catalog_vocab import load_catalog_vocab
from .parser import MessageParser
from .llm_parser import LLMMessageParser


def main() -> None:
    try:
        categories, brands = load_catalog_vocab("data/catalog.jsonl")
        print(f"Loaded catalog vocab: {len(categories)} categories, {len(brands)} brands.")
    except FileNotFoundError:
        categories, brands = set(), set()
        print("data/catalog.jsonl not found — running without category/brand matching.")
        print("(gzip -dk data/catalog.jsonl.gz && mv data/catalog.jsonl data/catalog.jsonl to enable it)")

    # parser = MessageParser(known_categories=categories, known_brands=brands)
    parser = LLMMessageParser(known_categories=categories, known_brands=brands)
    print("Type a customer message (or 'quit'):\n")

    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if not text or text.lower() in {"quit", "exit"}:
            break
        parsed = parser.parse(text)
        print(json.dumps(parsed.to_dict(), indent=2))


if __name__ == "__main__":
    sys.exit(main())
