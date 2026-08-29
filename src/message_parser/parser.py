"""Core extraction: raw message -> keywords + structured attributes + signals."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .vocab import (
    BUDGET_RE,
    BUDGET_WORDS_RE,
    COLOR_RE,
    COMPOUND_ALIASES,
    GENERIC_SINGLE_WORD_BLOCKLIST,
    MATERIAL_RE,
    MIN_SINGLE_WORD_VOCAB_LEN,
    NO_PREFERENCE_PATTERNS,
    OVERRIDE_PATTERNS,
    SIZE_BARE_LETTER_RE,
    SIZE_LETTER_RE,
    SIZE_NUMERIC_RE,
    SIZE_WIDTH_RE,
    STOPWORDS,
    STYLE_KEYWORDS,
    TOKEN_RE,
    USE_CASE_KEYWORDS,
    VAGUE_PATTERNS,
)


@dataclass
class ParsedMessage:
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


def _match_compound_alias(
    tokens: list[str],
    vocab: set[str],
    claimed: set[str] | None = None,
) -> tuple[str, list[str]] | None:
    """Checks each raw token's compound-alias phrase (e.g. "crossbody" ->
    "cross body") against `vocab` directly. Deliberately does NOT inject the
    split fragments ("cross", "body") into the general token stream for
    n-gram matching — treating them as independent words would let an
    unrelated single-word match (e.g. a store literally named "Cross")
    fire on a fragment that only exists because of the alias split."""
    claimed = claimed or set()
    for token in tokens:
        if token in claimed:
            continue
        alias = COMPOUND_ALIASES.get(token)
        if not alias:
            continue
        if alias in vocab:
            return alias, [token]
        plural = alias if alias.endswith("s") else alias + "s"
        if plural in vocab:
            return plural, [token]
    return None


def _match_vocab_ngrams(
    tokens: list[str],
    vocab: set[str],
    claimed: set[str] | None = None,
    max_n: int = 4,
) -> tuple[str, list[str]] | None:
    """Longest vocab phrase present in `tokens`. Returns (matched_vocab_term,
    original_tokens_matched) — callers must add the *original* tokens (not
    the matched/pluralized term) to `claimed`, or a singular customer word
    ("jacket") and its plural catalog match ("jackets") won't be recognized
    as the same token and a later matcher (e.g. category, after brand) could
    double-assign it.

    `claimed` tokens (already consumed by a higher-priority attribute) are
    skipped so an ambiguous term like "cotton" (a real material AND a real
    catalog category) isn't double-assigned. Single-word matches under
    MIN_SINGLE_WORD_VOCAB_LEN or in the generic blocklist are skipped (some
    real store names are plain English words, e.g. "Key", "Not")."""
    claimed = claimed or set()
    n = len(tokens)
    for size in range(min(max_n, n), 0, -1):
        for i in range(n - size + 1):
            window = tokens[i : i + size]
            if any(t in claimed for t in window):
                continue
            phrase = " ".join(window)
            if size == 1 and (len(phrase) < MIN_SINGLE_WORD_VOCAB_LEN or phrase in GENERIC_SINGLE_WORD_BLOCKLIST):
                continue
            if phrase in vocab:
                return phrase, window
            # Plural-insensitive fallback: catalog terms are frequently
            # plural ("Shirts", "Jackets") while a customer says singular.
            plural_variant = phrase if phrase.endswith("s") else phrase + "s"
            singular_variant = phrase[:-1] if phrase.endswith("s") and len(phrase) > 3 else None
            if plural_variant in vocab:
                return plural_variant, window
            if singular_variant and singular_variant in vocab:
                return singular_variant, window
    return None


class MessageParser:
    """Stateless — one instance can be reused across all sessions."""

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
        result.is_vague = _matches_any(lowered, VAGUE_PATTERNS)

        if not result.is_no_preference:
            self._extract_attributes(lowered, text, result)

        result.keywords = _clean_terms(text)

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

        budget = BUDGET_RE.search(original) or BUDGET_WORDS_RE.search(original)
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

        # Category before brand: a word coincidentally matching both a real
        # store name and a real category (e.g. "jacket"/"Jackets") almost
        # always reflects category intent, not a store mention. Compound
        # alias ("crossbody" -> "cross body") is tried first since it's a
        # more specific signal than generic n-gram matching.
        if self.known_categories:
            category_match = _match_compound_alias(tokens, self.known_categories, claimed=claimed) \
                or _match_vocab_ngrams(tokens, self.known_categories, claimed=claimed)
            if category_match:
                value, window = category_match
                result.attributes["category"] = value
                claimed.update(window)

        if self.known_brands:
            brand_match = _match_compound_alias(tokens, self.known_brands, claimed=claimed) \
                or _match_vocab_ngrams(tokens, self.known_brands, claimed=claimed)
            if brand_match:
                value, window = brand_match
                result.attributes["brand"] = value
                claimed.update(window)
