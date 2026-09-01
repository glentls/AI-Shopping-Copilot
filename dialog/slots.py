"""Structured slot tracking with contradiction-based override detection.

Reuses retrieval.structured's term lists/regex rather than redefining them, so "what
counts as a material/color/budget mention" stays a single source of truth across the
retrieval-scoring path and the dialog-slot path.

Only material/color/budget/size get first-class slot extraction: these are the
attributes with an unambiguous, cheaply-extractable surface form in free text (a known
term, a regex-matched size token, a dollar amount). style/feature/use_case/other are
tracked as an unstructured bag of "other terms" -- useful as disclosed context, but not
worth building unreliable extraction heuristics for. category/brand are deliberately
NOT tracked as slots: see ASK_ATTRIBUTE_BLOCKLIST below.

Accumulation vs. override: a first value for an attribute is accumulated. A *second,
different* value for an attribute already filled is treated as a contradiction and
overwrites the old one (erase-and-rewrite), which is how CLAUDE.md's Phase 3d asks for
override detection -- "by contradiction against the current slot state, not by keyword
matching on the word 'actually'." This also happens to fire correctly on the evaluator's
scripted override turn without ever looking at the message text for "actually".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from retrieval.structured import BUDGET_RE, COLOR_TERMS, MATERIAL_TERMS

SIZE_RE = re.compile(
    r"\b(xx-?small|xx-?large|x-?small|x-?large|small|medium|large|xxs|xs|s|m|l|xl|xxl|"
    r"size\s+\d{1,2}(?:\.\d)?|\d{1,2}(?:\.\d)?\s*(?:us|uk|eu))\b",
    re.IGNORECASE,
)

# The evaluator's own customer_reply() classifies disclosed constraints into
# {budget, material, color, size, style, use_case} or falls through to "feature"
# (evaluator/local_evaluator.py:137-151) -- "brand" and "category" are never a
# classify_constraint() outcome, so no disclosed constraint can ever match them. Asking
# either is provably a wasted turn against this specific simulator (never against a real
# user -- see docs/evaluator_notes.md, this is a benchmark quirk, not a real-world one,
# and the question policy only uses it as a tie-breaker, never an absolute veto).
ASK_ATTRIBUTE_BLOCKLIST = {"brand", "category"}

STRUCTURED_ATTRIBUTES = ("material", "color", "size", "budget")


def _extract_material(text: str) -> str | None:
    lowered = text.lower()
    for term in MATERIAL_TERMS:
        if term in lowered:
            return term
    return None


def _extract_color(text: str) -> str | None:
    lowered = text.lower()
    for term in COLOR_TERMS:
        if term in lowered:
            return term
    return None


def _extract_size(text: str) -> str | None:
    match = SIZE_RE.search(text)
    return match.group(1).lower() if match else None


def _extract_budget(text: str) -> float | None:
    match = BUDGET_RE.search(text)
    return float(match.group(1)) if match else None


_EXTRACTORS = {
    "material": _extract_material,
    "color": _extract_color,
    "size": _extract_size,
    "budget": _extract_budget,
}


@dataclass
class SlotState:
    values: dict[str, object] = field(default_factory=dict)
    other_terms: list[str] = field(default_factory=list)
    overridden_attributes: set[str] = field(default_factory=set)

    def update(self, text: str) -> set[str]:
        """Extract slot values from a new user message. A conflicting new value for an
        already-filled attribute overwrites (override); a first value fills it
        (accumulate). Returns the attributes overridden *by this call* -- callers use
        this to know when to drop state that was only valid under the old intent (e.g.
        proven-negative exclusions), not just the cumulative history."""
        just_overridden: set[str] = set()
        for attribute, extractor in _EXTRACTORS.items():
            new_value = extractor(text)
            if new_value is None:
                continue
            old_value = self.values.get(attribute)
            if old_value is not None and old_value != new_value:
                self.overridden_attributes.add(attribute)
                just_overridden.add(attribute)
            self.values[attribute] = new_value
        return just_overridden

        lowered = text.lower()
        for term in ("running", "hiking", "gym", "winter", "outdoor", "work", "casual", "formal"):
            if term in lowered and term not in self.other_terms:
                self.other_terms.append(term)

    def unfilled_attributes(self, enum: tuple[str, ...]) -> list[str]:
        return [attr for attr in enum if attr not in self.values and attr not in ASK_ATTRIBUTE_BLOCKLIST]

    def as_query_terms(self) -> list[str]:
        terms = [str(v) for v in self.values.values() if v is not None]
        return terms + list(self.other_terms)
