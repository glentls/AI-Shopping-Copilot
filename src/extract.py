"""Turn one customer message into normalized conversational constraints."""

from __future__ import annotations

import re

from src.contracts import SLOTS, ConversationState, SlotValue
from src.lexicons import (
    IDIOM_RE,
    NEGATION_RE,
    NO_PREFERENCE_RE,
    OVERRIDE_RE,
    PATTERNS,
    RECIPIENT_RE,
    SLOT_NAME_PATTERNS,
    WORD_NUMBERS,
)


_NUMBER = r"\d{1,6}(?:,\d{3})*(?:\.\d{1,2})?"
_CURRENCY = r"(?:US\s*)?(?:[$\u00a3\u20ac]|USD\s*|EUR\s*|GBP\s*)?"
_CURRENCY_REQUIRED = r"(?:US\s*\$|[$\u00a3\u20ac]|USD\s*|EUR\s*|GBP\s*)"
# Currency named after the number: "50 dollars", "40 euros", "50 quid".
_CURRENCY_WORD = r"(?:dollars?|usd|euros?|eur|pounds?|gbp|quid|bucks?)"

# Range patterns come before the one-sided patterns: "$50-100" means a maximum
# of 100, not 50. A currency marker, "dollars", or explicit budget wording is
# required for a bare range so clothing sizes such as 8-10 are not prices.
BUDGET_RANGE_RE = re.compile(
    rf"(?:"
    rf"(?:between|from)\s*{_CURRENCY}(?P<lo1>{_NUMBER})\s*(?:and|to|-)\s*{_CURRENCY}(?P<hi1>{_NUMBER})"
    rf"|{_CURRENCY_REQUIRED}(?P<lo2>{_NUMBER})\s*(?:-|to)\s*{_CURRENCY}(?P<hi2>{_NUMBER})"
    rf"|(?P<lo3>{_NUMBER})\s*(?:-|to)\s*(?P<hi3>{_NUMBER})\s*{_CURRENCY_WORD}"
    rf"|budget(?:\s+(?:is|of|around|about|roughly|approximately))*\s*{_CURRENCY}(?P<lo4>{_NUMBER})\s*(?:-|to)\s*{_CURRENCY}(?P<hi4>{_NUMBER})"
    rf")",
    re.IGNORECASE,
)
# "under 80" is unambiguous budget language; a bare "max 90" is not, because
# product names are full of it -- "air max 90" was being read as a $90 ceiling.
# So the strong triggers stand alone, while max/maximum needs a currency marker
# or an explicit "dollars".
_STRONG_MAX = r"under|below|less than|no more than|not over|up to|within|at most"
BUDGET_MAX_RE = re.compile(
    rf"(?:"
    rf"(?:{_STRONG_MAX})\s*{_CURRENCY}(?P<maximum>{_NUMBER})(?:\s*{_CURRENCY_WORD})?"
    rf"|max(?:imum)?(?: of)?\s*(?:{_CURRENCY_REQUIRED}(?P<max_cur>{_NUMBER})"
    rf"|(?P<max_word>{_NUMBER})\s*{_CURRENCY_WORD})"
    rf")",
    re.IGNORECASE,
)
# "budget" is an unambiguous trigger and stands alone. "about" and "around"
# are not: "how about 10 to 20 litres" and "what about size 8" are ordinary
# conversation, and reading them as a $10 ceiling silently distorts the whole
# ranking. The vague triggers therefore require a currency marker or an
# explicit currency word, exactly like bare "max".
BUDGET_AROUND_RE = re.compile(
    rf"(?:"
    rf"budget(?:\s+(?:is|of|around|about|roughly|approximately))*\s*{_CURRENCY}(?P<around>{_NUMBER})"
    rf"(?:\s*{_CURRENCY_WORD})?"
    rf"|(?:around|about|roughly|approximately)\s*"
    rf"(?:{_CURRENCY_REQUIRED}(?P<around_cur>{_NUMBER})"
    rf"|(?P<around_word>{_NUMBER})\s*{_CURRENCY_WORD})"
    rf")",
    re.IGNORECASE,
)
_WORDS = "|".join(sorted((re.escape(w) for w in WORD_NUMBERS), key=len, reverse=True))
# "under fifty dollars", "no more than eighty", "a couple of hundred at most".
BUDGET_WORD_RE = re.compile(
    rf"(?:{_STRONG_MAX}|budget(?:\s+(?:is|of|around|about))*|around|about|roughly)\s*"
    rf"(?P<worded>{_WORDS})(?:\s*{_CURRENCY_WORD})?",
    re.IGNORECASE,
)
# The ceiling stated after the number: "50 quid max", "EUR 40 or less".
BUDGET_TRAILING_RE = re.compile(
    rf"{_CURRENCY}(?P<trailing>{_NUMBER})\s*(?:{_CURRENCY_WORD}\s*)?"
    rf"(?:max(?:imum)?|or less|at most|tops)\b",
    re.IGNORECASE,
)
BUDGET_WORD_TRAILING_RE = re.compile(
    rf"(?P<wtrailing>{_WORDS})(?:\s*{_CURRENCY_WORD})?\s*"
    rf"(?:max(?:imum)?|or less|at most|tops)\b",
    re.IGNORECASE,
)

PRICE_RE = re.compile(rf"(?:US\s*)?\$\s*(?P<price>{_NUMBER})", re.IGNORECASE)

# Numeric sizes need context; unconstrained numbers are commonly prices,
# quantities, percentages, or product model numbers.
SIZE_RANGE_RE = re.compile(
    r"\bsizes?\s*(?P<lo>\d{1,2}(?:\.5)?)\s*(?:-|to|through)\s*(?P<hi>\d{1,2}(?:\.5)?)\b",
    re.IGNORECASE,
)
SIZE_NUMBER_RE = re.compile(
    r"(?:\b(?:shoe|dress|clothing|waist|size|sized)\s*(?:size\s*)?(?P<n1>\d{1,3}(?:\.5)?)\b"
    r"|\b(?P<n2>\d{1,2}(?:\.5)?)\s*(?:US\s*)?(?:shoe\s*)?size\b"
    r"|\bUS\s*(?P<n3>\d{1,2}(?:\.5)?)\b)",
    re.IGNORECASE,
)
BRA_SIZE_RE = re.compile(r"\b(?P<band>2[6-9]|[3-5]\d)(?P<cup>aa|a|b|c|d{1,3}|e|f|g|h)\b", re.IGNORECASE)

_CLAUSE_BREAK_RE = re.compile(r"[,;.!?]|\b(?:but|however|though|although|instead)\b", re.IGNORECASE)


def _number(value: str) -> float:
    return float(value.replace(",", ""))


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.2f}".rstrip("0").rstrip(".")


def parse_budget(text: str) -> float | None:
    """Return the customer's numeric maximum, including the high end of ranges."""
    source = text or ""
    match = BUDGET_RANGE_RE.search(source)
    if match:
        high = match.group("hi1") or match.group("hi2") or match.group("hi3") or match.group("hi4")
        return _number(high)
    match = BUDGET_MAX_RE.search(source)
    if match:
        ceiling = (match.group("maximum") or match.group("max_cur")
                   or match.group("max_word"))
        return _number(ceiling)
    match = BUDGET_TRAILING_RE.search(source)
    if match:
        return _number(match.group("trailing"))
    match = BUDGET_AROUND_RE.search(source)
    if match:
        approximate = (match.group("around") or match.group("around_cur")
                       or match.group("around_word"))
        return _number(approximate)
    for pattern, group in ((BUDGET_WORD_RE, "worded"),
                           (BUDGET_WORD_TRAILING_RE, "wtrailing")):
        match = pattern.search(source)
        if match:
            return float(WORD_NUMBERS[re.sub(r"\s+", " ", match.group(group).lower())])
    match = PRICE_RE.search(source)
    return _number(match.group("price")) if match else None


def _clause_before(text: str, start: int) -> str:
    """Text in the current clause before a value, used for negation scope."""
    prefix = text[max(0, start - 80):start]
    breaks = [
        match
        for match in _CLAUSE_BREAK_RE.finditer(prefix)
        if not (
            match.group(0).casefold() == "but"
            and re.search(r"\banything\s+$", prefix[:match.start()], re.IGNORECASE)
        )
    ]
    return prefix[breaks[-1].end():] if breaks else prefix


def _is_negated(text: str, start: int) -> bool:
    clause = _clause_before(text, start)
    # "No preference for black" is boundary behavior, not an exclusion.
    if NO_PREFERENCE_RE.search(clause):
        return False
    return bool(NEGATION_RE.search(clause))


def _is_no_preference_clause(text: str, start: int) -> bool:
    left = max(text.rfind(mark, 0, start) for mark in (",", ";", ".", "!", "?"))
    right_candidates = [text.find(mark, start) for mark in (",", ";", ".", "!", "?")]
    right_candidates = [position for position in right_candidates if position >= 0]
    right = min(right_candidates) if right_candidates else len(text)
    return bool(NO_PREFERENCE_RE.search(text[left + 1:right]))


def _append(
    found: dict[str, list[SlotValue]],
    seen: set[tuple[str, str, bool]],
    slot: str,
    value: str,
    confidence: float,
    turn: int,
    polarity: bool,
) -> None:
    key = (slot, value, polarity)
    if key in seen:
        return
    seen.add(key)
    found.setdefault(slot, []).append(SlotValue(value, confidence, turn, polarity))


def _dynamic_sizes(text: str) -> list[tuple[str, int]]:
    sizes: list[tuple[str, int]] = []
    occupied: list[tuple[int, int]] = []
    for match in SIZE_RANGE_RE.finditer(text):
        sizes.append((f"{match.group('lo')}-{match.group('hi')}", match.start()))
        occupied.append(match.span())
    for match in SIZE_NUMBER_RE.finditer(text):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        sizes.append((match.group("n1") or match.group("n2") or match.group("n3"), match.start()))
    for match in BRA_SIZE_RE.finditer(text):
        sizes.append(((match.group("band") + match.group("cup")).lower(), match.start()))
    return sizes


def extract_slots(text: str, turn: int, state: ConversationState) -> dict[str, list[SlotValue]]:
    """Extract every explicit or normalized preference from one customer turn.

    The function also sets ``state.budget_max``. This small, intentional side
    effect keeps the fixed interface useful to callers which invoke extraction
    directly rather than through ``src.policy.state.update``.
    """
    found: dict[str, list[SlotValue]] = {}
    if not text:
        return found

    seen: set[tuple[str, str, bool]] = set()
    idioms = [(m.start(), m.end()) for m in IDIOM_RE.finditer(text)]
    for slot, entries in PATTERNS.items():
        for canonical, pattern in entries:
            for match in pattern.finditer(text):
                if _is_no_preference_clause(text, match.start()):
                    continue
                # "feeling blue" is not a colour preference.
                if any(lo <= match.start() < hi for lo, hi in idioms):
                    continue
                polarity = not _is_negated(text, match.start())
                surface = re.sub(r"[\s-]+", " ", match.group(0).casefold()).strip()
                confidence = 0.95 if surface == canonical.casefold() else 0.90
                _append(found, seen, slot, canonical, confidence, turn, polarity)

    for value, pattern in RECIPIENT_RE.items():
        if pattern.search(text):
            # Softer than a stated preference: inferred, not said.
            _append(found, seen, "category", value, 0.75, turn, True)

    for value, start in _dynamic_sizes(text):
        _append(found, seen, "size", value, 0.94, turn, not _is_negated(text, start))

    budget = parse_budget(text)
    if budget is not None:
        state.budget_max = budget
        _append(found, seen, "budget", _format_number(budget), 0.97, turn, True)

    return found


def _named_slots(text: str) -> set[str]:
    return {slot for slot, pattern in SLOT_NAME_PATTERNS.items() if pattern.search(text)}


def detect_override(text: str) -> list[str]:
    """Return slots affected by a customer's mid-conversation retraction.

    ``["*"]`` means an override cue was present but the affected slot could not
    be identified; Lane C resolves that against newly extracted values.
    """
    source = text or ""
    if not OVERRIDE_RE.search(source):
        return []
    affected = _named_slots(source)
    for slot, entries in PATTERNS.items():
        if any(pattern.search(source) for _, pattern in entries):
            affected.add(slot)
    if parse_budget(source) is not None:
        affected.add("budget")
    if _dynamic_sizes(source):
        affected.add("size")
    ordered = [slot for slot in SLOTS if slot in affected]
    return ordered or ["*"]


def detect_no_preference(text: str) -> list[str]:
    """Return named unconstrained slots, or ``["*"]`` for the last question."""
    source = text or ""
    if not NO_PREFERENCE_RE.search(source):
        return []
    named = _named_slots(source)
    ordered = [slot for slot in SLOTS if slot in named]
    return ordered or ["*"]
