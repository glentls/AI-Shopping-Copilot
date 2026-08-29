# src/

Team implementation modules (see `docs/diagrams/architecture.md` for the full design).

## `message_parser.py` — Message Parser (owner: Shreya)

Pure function of message text: `MessageParser.parse(text) -> ParsedMessage`.
No session state — safe to instantiate once and reuse across all sessions.

```python
from src.message_parser import MessageParser, load_catalog_vocab

# Optional but recommended: catalog-derived vocab gives much higher-precision
# category/brand matching than heuristics alone.
categories, brands = load_catalog_vocab("data/catalog.jsonl")
parser = MessageParser(known_categories=categories, known_brands=brands)

parsed = parser.parse("I'm looking for black leather boots, size 9, under $80.")
parsed.attributes   # {'material': 'leather', 'color': 'black', 'size': '9', 'budget': '80', 'category': 'boots'}
parsed.keywords      # ['looking', 'black', 'leather', 'boots', 'size', '9', '80'] (BM25 query terms)
parsed.is_override      # False
parsed.is_no_preference # False
parsed.is_vague          # False

parsed.to_dict()  # JSON-serializable {"raw_text", "keywords", "attributes", "signals": {...}}
```

**Integration points:**
- **Ledger (Tiffany):** merge `parsed.attributes` into `ledger.attributes`. If `parsed.is_override` is set, clear prior attributes first (or otherwise mark the old ones as superseded) before merging in the new ones.
- **Intent Router (Sera):** `is_override` / `is_no_preference` / `is_vague` are ready-made classification signals — no need to re-derive scenario type from raw text.
- **BM25 Retriever (Nick):** `parsed.keywords` is the cleaned, deduped term list to build the search query from.

**Attributes it can extract:** `material`, `color`, `size`, `budget`, `brand`\*, `category`\*, `style`, `use_case`, and a `feature` catch-all for meaningful text that doesn't match a narrower bucket (\* brand/category need `known_brands`/`known_categories` passed in — falls back to skipping those two without it).

**Design notes:**
- Vocab-based (not exact string/template matching) so it's robust to the organizer paraphrasing customer messages differently in the private/final harness — see `docs/competition_specification.md`: *"If natural-language paraphrasing is added by the organizer, it cannot decide correctness."*
- When a token is claimed by a higher-priority attribute (e.g. "cotton" as `material`), it's excluded from later category/brand matching — some real catalog terms are ambiguous (`cotton`, `denim`, `fleece` are both materials *and* real Amazon leaf categories; `bamboo`, `canvas` are both materials *and* real store names) and would otherwise get double-assigned.
- Single-word brand/category vocab entries under 4 characters, or in a small blocklist of generic words, are excluded — a handful of real store names in the catalog are ordinary short English words (e.g. "Key", "Not") that would otherwise false-positive on unrelated sentences.

Tests: `tests/test_message_parser.py` (`python3 -m unittest tests.test_message_parser -v`).
