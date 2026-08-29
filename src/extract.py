"""Customer message -> structured slots.

LANE A OWNS THIS FILE. Thin but working; expanding coverage is Lane A's job.
Signatures are fixed by src/contracts.py and other lanes code against them.
"""

from __future__ import annotations

import re

from src.contracts import ConversationState, SlotValue
from src.lexicons import NEGATION_CUES, NO_PREFERENCE_CUES, OVERRIDE_CUES, PATTERNS

# "under $80", "below 80 dollars", "less than $50"
BUDGET_RE = re.compile(
    r"(?:under|below|less than|no more than|max(?:imum)?|up to|within)\s*\$?\s*(\d+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
PRICE_RE = re.compile(r"\$\s*(\d+(?:\.\d{1,2})?)")

_NEG_WINDOW = 24  # characters before a match to scan for a negation cue

# Word-bounded, or "no" matches inside "ignore" and "not" inside "another".
_NEGATION_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(cue) for cue in NEGATION_CUES) + r")\b",
    re.IGNORECASE,
)


def _is_negated(text: str, start: int) -> bool:
    return bool(_NEGATION_RE.search(text[max(0, start - _NEG_WINDOW):start]))


def extract_slots(text: str, turn: int, state: ConversationState) -> dict[str, list[SlotValue]]:
    """Pull every slot value we can find out of one customer message."""
    found: dict[str, list[SlotValue]] = {}
    if not text:
        return found

    for slot, entries in PATTERNS.items():
        for canonical, pattern in entries:
            match = pattern.search(text)
            if not match:
                continue
            found.setdefault(slot, []).append(
                SlotValue(
                    value=canonical,
                    confidence=0.9,
                    turn=turn,
                    polarity=not _is_negated(text, match.start()),
                )
            )

    budget = BUDGET_RE.search(text) or PRICE_RE.search(text)
    if budget:
        found.setdefault("budget", []).append(
            SlotValue(value=budget.group(1), confidence=0.95, turn=turn)
        )

    return found


def detect_override(text: str) -> list[str]:
    """Slot names the customer is retracting. Empty list means no override.

    Returning ["*"] means "we saw an override cue but cannot tell which slot" --
    Lane C treats that as: retract whatever slots the new message overwrites.
    """
    lowered = (text or "").lower()
    if not any(cue in lowered for cue in OVERRIDE_CUES):
        return []
    replaced = [slot for slot in PATTERNS if any(p.search(text) for _, p in PATTERNS[slot])]
    return replaced or ["*"]


def detect_no_preference(text: str) -> list[str]:
    """Non-empty when the customer has no preference for what we just asked.

    Returns the named slot when the message names one, otherwise ["*"] meaning
    "the slot we asked about last turn" -- Lane C resolves that via
    state.last_asked.
    """
    lowered = (text or "").lower()
    if not any(cue in lowered for cue in NO_PREFERENCE_CUES):
        return []
    from src.contracts import SLOTS

    named = [slot for slot in SLOTS if re.search(rf"\b{slot.replace('_', '[ _]')}\b", lowered)]
    return named or ["*"]


def parse_budget(text: str) -> float | None:
    match = BUDGET_RE.search(text or "")
    return float(match.group(1)) if match else None
