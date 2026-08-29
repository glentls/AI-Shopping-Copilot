"""Per-turn result of the confidence check."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidencePayload:
    """Result of the confidence function for a single turn.

    ``reason`` is a human-readable string for the demo/report, e.g.
    "2 of 4 constraints known, 87 products tie at full coverage".
    """

    score: float
    clarify: bool
    ask_attribute: str | None
    reason: str
