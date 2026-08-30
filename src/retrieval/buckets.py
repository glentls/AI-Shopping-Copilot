"""Category bucketing over the catalog.

The customer's opening line is generated as ``I'm looking for {category}...``
where ``{category}`` is the *coarse category* of the hidden target -- the last
two comma-separated segments of that product's own ``categories`` path. So the
first message names, verbatim, a bucket that is guaranteed to contain the
target. The catalog splits into 1,115 such buckets with a median size of 8
(median ~182 for the buckets targets actually fall in), which turns a 50,000
product ranking problem into a ~180 product one.

``coarse_category`` is reimplemented here rather than imported from
``evaluator/`` -- the shipped agent must not depend on the evaluator package.

``resolve`` never fails: it degrades through exact match, containment, and
token overlap before giving up and returning ``None`` (meaning "search the
whole catalog"). The inexact rungs are paraphrase insurance for the private
set, where the opening template may be reworded.
"""

from __future__ import annotations

import re
from collections import defaultdict

# Top-level category strings that carry no discriminative signal -- every
# product in this catalog is under Clothing, Shoes & Jewelry.
_EXCLUDED = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}

_FALLBACK_CATEGORY = "clothing item"

# The opening line, whose tail is the coarse category. The browsing/boundary
# variant ends ", but I'm still exploring."; buying continues ". A key
# requirement is: ..."; intent_override continues ". <preference>".
# Kept deliberately loose on the verb: the private set may reword the wrapper,
# and the category tail is what matters. If no verb matches at all, `parse_category`
# falls back to the whole message so the fuzzy rung still has something to chew on.
_OPENING_RE = re.compile(
    r"(?:looking|searching|hunting|shopping)\s+for\s+|"
    r"(?:i\s+(?:want|need)|show\s+me|interested\s+in|after)\s+",
    re.IGNORECASE,
)

_OPENING_TAIL_RE = re.compile(
    r"(?:,\s*but\s+i'?m\s+still\s+exploring|\.\s|\.$)",
    re.IGNORECASE,
)

# Filler that appears around a paraphrased category but never inside a real
# catalog category key.
_FILLER = {"a", "an", "the", "some", "any", "new", "today", "please", "me", "i",
           "for", "of", "in", "and", "to", "my", "looking", "want", "need",
           "hi", "hello", "hey", "there", "recommendations", "recommendation",
           "suggestions", "on", "about", "something", "anything", "im", "am"}

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Minimum token overlap for the fuzzy rung to fire. Below this the match is
# noise and searching the whole catalog is the safer degradation.
_MIN_JACCARD = 0.34


def coarse_category(values: list[str] | None) -> str:
    """The last two meaningful segments of a product's category path."""
    cleaned: list[str] = []
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in _EXCLUDED:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else _FALLBACK_CATEGORY


def _tokens(text: str, drop_filler: bool = False) -> set[str]:
    found = set(_TOKEN_RE.findall(text.lower()))
    return found - _FILLER if drop_filler else found


def parse_category(message: str) -> str:
    """Pull the category fragment out of an opening message.

    Falls back to the whole message when no opening verb is recognised, so a
    fully reworded greeting still reaches the token-overlap rung rather than
    silently degrading to a whole-catalog scan.
    """
    text = message or ""
    match = _OPENING_RE.search(text)
    body = text[match.end():] if match else text
    tail = _OPENING_TAIL_RE.search(body)
    if tail:
        body = body[: tail.start()]
    return re.sub(r"\s+", " ", body).strip(" .,;:")


class BucketIndex:
    """Maps a coarse-category string to the parent_asins that live in it."""

    def __init__(self, rows) -> None:
        self._buckets: dict[str, list[str]] = defaultdict(list)
        for product in rows:
            key = coarse_category(product.get("categories"))
            self._buckets[key.lower()].append(str(product["parent_asin"]))
        self._buckets = dict(self._buckets)
        self._token_sets = {key: _tokens(key) for key in self._buckets}

    def __len__(self) -> int:
        return len(self._buckets)

    def get(self, key: str) -> list[str]:
        return self._buckets.get(key.lower(), [])

    def resolve(self, message: str) -> tuple[str | None, str]:
        """Resolve an opening message to a bucket key.

        Returns ``(key, how)`` where ``how`` names the rung that fired, for the
        A/B log. ``key`` is ``None`` when nothing matched well enough, which
        callers must read as "fall back to the whole catalog".
        """
        fragment = parse_category(message)
        if not fragment:
            return None, "no-category-parsed"

        lowered = fragment.lower()
        if lowered in self._buckets:
            return lowered, "exact"

        # Containment: the fragment survived a reworded wrapper, or the
        # organizer trimmed/extended the path. Longest wins -- a longer key
        # shares more of the path and so is the more specific bucket.
        contained = [
            key for key in self._buckets
            if key and (key in lowered or lowered in key)
        ]
        if contained:
            return max(contained, key=len), "containment"

        # Token overlap: word order changed or a connector was dropped.
        fragment_tokens = _tokens(lowered, drop_filler=True)
        if fragment_tokens:
            best_key, best_score = None, 0.0
            for key, key_tokens in self._token_sets.items():
                if not key_tokens:
                    continue
                overlap = len(fragment_tokens & key_tokens)
                if not overlap:
                    continue
                score = overlap / len(fragment_tokens | key_tokens)
                if score > best_score:
                    best_key, best_score = key, score
            if best_key is not None and best_score >= _MIN_JACCARD:
                return best_key, f"jaccard:{best_score:.2f}"

        return None, "unresolved"
