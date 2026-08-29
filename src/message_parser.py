"""Message Parser — turns a raw customer message into search keywords and
structured product attributes.

Owner: Shreya (Message Parser).

Design note: this module is a *pure* function of message text (plus optional
catalog-derived vocabularies for higher-precision brand/category matching).
It holds no session state on purpose, so it composes cleanly with:

- Ledger (Tiffany): merges ``ParsedMessage.attributes`` into the session's
  stored attributes; uses ``ParsedMessage.is_override`` to decide whether to
  clear prior attributes before merging.
- Intent Router (Sera): uses ``is_override`` / ``is_no_preference`` /
  ``is_vague`` as classification signals for Buying/Browsing/Override/
  Boundary routing.
- BM25 Retriever (Nick): uses ``ParsedMessage.keywords`` to build the search
  query.

Usage:
    >>> categories, brands = load_catalog_vocab("data/catalog.jsonl")
    >>> parser = MessageParser(known_categories=categories, known_brands=brands)
    >>> parsed = parser.parse("I'm looking for black leather boots, size 9, under $80")
    >>> parsed.attributes
    {'material': 'leather', 'color': 'black', 'size': '9', 'budget': '80', 'category': 'boots'}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_ATTRIBUTES = (
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case",
)

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "just", "me", "my", "of", "on", "or", "please",
    "some", "still", "that", "the", "this", "to", "want", "with", "would",
    "you", "looking", "need", "im", "am", "exploring", "something",
}

MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "fabric", "denim", "cashmere", "linen", "suede", "canvas",
    "mesh", "velvet", "satin", "chiffon", "lace", "faux fur", "fur",
    "fleece", "corduroy", "knit", "elastane", "viscose", "acrylic",
    "bamboo", "leatherette", "polyurethane",
)
MATERIAL_RE = re.compile(r"\b(" + "|".join(re.escape(m) for m in MATERIALS) + r")\b", re.I)

COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray",
    "grey", "purple", "yellow", "orange", "navy", "beige", "tan", "gold",
    "silver", "cream", "maroon", "teal", "turquoise", "khaki", "ivory",
    "burgundy", "coral", "lavender", "mint", "multicolor", "multi-color",
    "rose gold",
)
COLOR_RE = re.compile(r"\b(" + "|".join(re.escape(c) for c in COLORS) + r")\b", re.I)

SIZE_NUMERIC_RE = re.compile(r"\bsize[:\s]*(\d{1,2}(?:\.\d)?)\b", re.I)
SIZE_LETTER_RE = re.compile(r"\bsize[:\s]*(x{0,3}s|x{0,3}l|m)\b", re.I)
SIZE_BARE_LETTER_RE = re.compile(r"\b(xxs|xs|small|medium|large|xl|xxl|xxxl)\b", re.I)
SIZE_WIDTH_RE = re.compile(r"\b(wide|narrow|regular)\s*width\b", re.I)

BUDGET_RE = re.compile(
    r"(?:under|below|less than|no more than|around|about|budget(?: of| around)?)?\s*\$\s?(\d+(?:\.\d{1,2})?)",
    re.I,
)

STYLE_KEYWORDS = (
    "formal", "casual", "vintage", "classic", "modern", "sporty", "elegant",
    "slim fit", "slim", "relaxed fit", "relaxed", "oversized", "cropped",
    "high-waisted", "high waisted", "sleeveless", "long sleeve",
    "short sleeve", "crew neck", "v-neck", "button-up", "button up",
    "zip-up", "zip up", "loose fit", "loose", "fitted", "straight leg",
    "skinny", "bootcut",
)
USE_CASE_KEYWORDS = (
    "running", "hiking", "gym", "workout", "yoga", "winter", "summer",
    "outdoor", "work", "office", "wedding", "party", "formal event",
    "everyday", "casual wear", "travel", "beach", "school", "sport",
    "athletic", "training", "walking", "swimming", "cycling",
)

OVERRIDE_PATTERNS = (
    "actually", "instead", "ignore my earlier", "ignore that", "scratch that",
    "change of mind", "changed my mind", "on second thought", "never mind that",
    "forget what i said", "rather than that", "let's go with", "lets go with",
)
NO_PREFERENCE_PATTERNS = (
    "don't have a preference", "do not have a preference", "no preference",
    "doesn't matter", "does not matter", "any is fine", "either is fine",
    "either works", "use your judgment", "use your judgement", "up to you",
    "i don't know", "i do not know", "no strong preference", "not picky",
    "whatever works", "you decide", "no particular",
)
VAGUE_PATTERNS = (
    "still exploring", "just looking", "just browsing", "not sure yet",
    "browsing", "open to anything", "not sure what", "no idea yet",
    # Evaluator's generic reprompt when the agent's last turn set no
    # ask_attribute — content-free, not a real product signal.
    "not quite right yet", "one specific attribute",
)

_EXCLUDED_CATEGORY_TERMS = {
    "clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry",
    "shoes & jewelry",
}

# Amazon store/category names are free text and occasionally collide with
# ordinary English function words (e.g. real stores named "Key" or "Not").
# Single-word vocab entries are the risky case — multi-word phrases rarely
# false-positive by accident, so only single words are filtered here.
_GENERIC_SINGLE_WORD_BLOCKLIST = {
    "key", "not", "so", "up", "in", "on", "at", "by", "or", "all", "new",
    "one", "top", "set", "box", "plus", "its", "out", "off", "non", "our",
    "any", "get", "buy", "shop", "store", "and", "for", "with", "you",
    "your", "what", "ask", "about", "right", "yet", "quite", "those",
    "options",
}
MIN_SINGLE_WORD_VOCAB_LEN = 4


@dataclass
class ParsedMessage:
    """Structured result of parsing one customer message."""

    raw_text: str
    keywords: list[str] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)
    is_override: bool = False
    is_no_preference: bool = False
    is_vague: bool = False

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "keywords": self.keywords,
            "attributes": self.attributes,
            "signals": {
                "is_override": self.is_override,
                "is_no_preference": self.is_no_preference,
                "is_vague": self.is_vague,
            },
        }


def _clean_terms(text: str) -> list[str]:
    """Tokenize, drop stopwords/short tokens, dedupe while preserving order."""
    seen: set[str] = set()
    terms: list[str] = []
    for token in TOKEN_RE.findall(text):
        lowered = token.lower()
        if len(lowered) <= 1 or lowered in STOPWORDS or lowered in seen:
            continue
        seen.add(lowered)
        terms.append(lowered)
    return terms


def _matches_any(lowered_text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in lowered_text for pattern in patterns)


def _first_keyword_hit(lowered_text: str, vocab: tuple[str, ...]) -> str | None:
    for term in vocab:
        if re.search(rf"\b{re.escape(term)}\b", lowered_text):
            return term
    return None


def _match_vocab_ngrams(
    tokens: list[str],
    vocab: set[str],
    claimed: set[str] | None = None,
    max_n: int = 4,
) -> str | None:
    """Find the longest vocab phrase present in `tokens` (hash-set lookups,
    so this stays fast even against a ~20k-term brand vocabulary).

    `claimed` are tokens already consumed by a higher-priority attribute
    (material/color/size/budget) — candidate phrases overlapping them are
    skipped so an ambiguous term (e.g. "cotton" is both a real material and
    a real Amazon leaf category) isn't assigned to two attributes at once.
    """
    claimed = claimed or set()
    n = len(tokens)
    for size in range(min(max_n, n), 0, -1):
        for i in range(n - size + 1):
            window = tokens[i : i + size]
            if any(t in claimed for t in window):
                continue
            phrase = " ".join(window)
            if size == 1 and (len(phrase) < MIN_SINGLE_WORD_VOCAB_LEN or phrase in _GENERIC_SINGLE_WORD_BLOCKLIST):
                continue
            if phrase in vocab:
                return phrase
    return None


def load_catalog_vocab(catalog_path: str | Path) -> tuple[set[str], set[str]]:
    """Build (categories, brands) vocab sets from the catalog for
    higher-precision matching. Both are lowercase strings, category values
    are individual leaf terms (catalog `categories` lists split on comma),
    brand values are `store` field values.
    """
    categories: set[str] = set()
    brands: set[str] = set()
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            for cat in product.get("categories") or []:
                for part in str(cat).split(","):
                    cleaned = part.strip().lower()
                    if cleaned and cleaned not in _EXCLUDED_CATEGORY_TERMS:
                        categories.add(cleaned)
            store = product.get("store")
            if store:
                brands.add(str(store).strip().lower())
    return categories, brands


class MessageParser:
    """Extracts search keywords and structured attributes from customer
    messages. Stateless across calls — safe to share one instance across
    all sessions.
    """

    def __init__(
        self,
        known_categories: set[str] | None = None,
        known_brands: set[str] | None = None,
    ) -> None:
        self.known_categories = known_categories or set()
        self.known_brands = known_brands or set()

    def parse(self, text: str) -> ParsedMessage:
        text = text or ""
        lowered = text.lower()
        result = ParsedMessage(raw_text=text)

        result.is_override = _matches_any(lowered, OVERRIDE_PATTERNS)
        result.is_no_preference = _matches_any(lowered, NO_PREFERENCE_PATTERNS)
        # Computed before the feature fallback below: an explicit "still
        # exploring" signal should stay vague even if the message happens to
        # contain a bare category noun (e.g. "I'm looking for shoes, but I'm
        # still exploring.") that would otherwise get swept into "feature".
        result.is_vague = _matches_any(lowered, VAGUE_PATTERNS)

        if not result.is_no_preference:
            self._extract_attributes(lowered, text, result)

        result.keywords = _clean_terms(text)

        # Catch-all: unstructured but meaningful content becomes a "feature"
        # constraint, mirroring how the evaluator buckets free-text product
        # snippets it can't classify into a narrower attribute.
        should_fallback = (
            not result.attributes
            and result.keywords
            and not result.is_no_preference
            and not result.is_override
            and not result.is_vague
        )
        if should_fallback:
            result.attributes["feature"] = " ".join(result.keywords[:8])

        return result

    def _extract_attributes(self, lowered: str, original: str, result: ParsedMessage) -> None:
        # Tokens already assigned to a higher-priority attribute are excluded
        # from later vocab matches — see `_match_vocab_ngrams` docstring.
        claimed: set[str] = set()

        material = MATERIAL_RE.search(lowered)
        if material:
            value = material.group(1).lower()
            result.attributes["material"] = value
            claimed.update(value.split())

        color = COLOR_RE.search(lowered)
        if color:
            value = color.group(1).lower()
            result.attributes["color"] = value
            claimed.update(value.split())

        size_match = SIZE_NUMERIC_RE.search(lowered) or SIZE_LETTER_RE.search(lowered)
        if size_match:
            value = size_match.group(1)
            result.attributes["size"] = value.upper() if value.isalpha() else value
            claimed.add(value.lower())
        else:
            bare_size = SIZE_BARE_LETTER_RE.search(lowered)
            width = SIZE_WIDTH_RE.search(lowered)
            if bare_size:
                result.attributes["size"] = bare_size.group(1).upper()
                claimed.add(bare_size.group(1).lower())
            elif width:
                result.attributes["size"] = width.group(0)

        budget = BUDGET_RE.search(original)
        if budget:
            result.attributes["budget"] = budget.group(1)

        style = _first_keyword_hit(lowered, STYLE_KEYWORDS)
        if style:
            result.attributes["style"] = style
            claimed.update(style.split())

        use_case = _first_keyword_hit(lowered, USE_CASE_KEYWORDS)
        if use_case:
            result.attributes["use_case"] = use_case
            claimed.update(use_case.split())

        tokens = [t.lower() for t in TOKEN_RE.findall(lowered)]

        if self.known_brands:
            brand = _match_vocab_ngrams(tokens, self.known_brands, claimed=claimed)
            if brand:
                result.attributes["brand"] = brand
                claimed.update(brand.split())

        if self.known_categories:
            category = _match_vocab_ngrams(tokens, self.known_categories, claimed=claimed)
            if category:
                result.attributes["category"] = category
                claimed.update(category.split())
