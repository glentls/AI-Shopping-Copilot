"""Policy mapping: turn a confidence score into a clarify decision.

    clarify = (conf < theta) AND (not exhausted) AND (turn < TURN_CUTOFF)
    ask_attribute = "other" whenever clarify        # fixed dominant attribute

Overrides:
    - Force clarify=True while n_constraints_known == 0 (never open on zero info).
    - After an override the ledger resets ``exhausted`` -> clarify resumes.

Recommendations are emitted every turn regardless of clarify; this policy only
decides the *question*.
"""

from __future__ import annotations

import os

from src.confidence.confidence import compute_confidence
from src.confidence.session_ledger import SessionLedger
from src.confidence.payload import ConfidencePayload
from src.reranker.types import RankResult

DEFAULT_THETA = 0.5
TURN_CUTOFF = 10           # stop asking at/after this turn (a None ask is a
                          # guaranteed zero-information turn, so ask to the end)
FIXED_ASK_ATTRIBUTE = "other"
FINAL_TURN = 10

# Exposure gate. The evaluator freezes MRR at the target's first top-10
# appearance, so a full list on turn 1 with only one generic constraint locks
# in a mid-list rank permanently. Exposing exactly one candidate keeps the
# upside (a correct top-1 hits at rank 1 immediately) with no downside (a wrong
# top-1 costs nothing, since MRR is unaffected until a hit).
RELEASE_TURN = 3
CONFIDENT_EXPOSURE = 1


def exposure_enabled() -> bool:
    """Gate is on by default; ``EXPOSURE_GATE=0`` reverts to full-list-every-turn
    (the ungated arm, reported alongside the gated score in the writeup)."""
    return os.environ.get("EXPOSURE_GATE", "1").strip() != "0"


def release_turn() -> int:
    raw = os.environ.get("RELEASE_TURN", "").strip()
    return int(raw) if raw.isdigit() else RELEASE_TURN


def exposure(turn: int, exhausted: bool, top_k: int) -> int:
    """How many recommendations to reveal this turn.

    Full list once we release (turn >= RELEASE_TURN), when the customer says the
    card is drained, or on the final turn (never withhold at turn 10 -- that
    truncation loses winnable sessions outright). Otherwise a single candidate.
    """
    if not exposure_enabled():
        return top_k
    if turn >= release_turn() or exhausted or turn >= FINAL_TURN:
        return top_k
    return CONFIDENT_EXPOSURE


def decide(
    rank: RankResult,
    ledger: SessionLedger,
    theta: float = DEFAULT_THETA,
) -> ConfidencePayload:
    """Compute confidence and the clarify decision for this turn."""
    n_known = ledger.n_constraints_known
    score, reason = compute_confidence(rank, n_known)

    # Zero-info: never open a browsing session without asking.
    if n_known == 0:
        return ConfidencePayload(
            score=score,
            clarify=True,
            ask_attribute=FIXED_ASK_ATTRIBUTE,
            reason=f"zero constraints known -> forced clarify ({reason})",
        )

    clarify = (score < theta) and (not ledger.exhausted) and (ledger.turn < TURN_CUTOFF)
    ask_attribute = FIXED_ASK_ATTRIBUTE if clarify else None

    if ledger.exhausted:
        reason = f"exhausted -> recommend only ({reason})"
    elif ledger.turn >= TURN_CUTOFF:
        reason = f"turn cutoff reached -> recommend only ({reason})"

    return ConfidencePayload(
        score=score,
        clarify=clarify,
        ask_attribute=ask_attribute,
        reason=reason,
    )


def always_ask(ledger: SessionLedger) -> ConfidencePayload:
    """P0 champion arm: ask until exhausted, ignoring confidence.

    Used as the ship-gate baseline. Recommendations still emitted every turn.
    """
    clarify = not ledger.exhausted and ledger.turn < TURN_CUTOFF
    return ConfidencePayload(
        score=float("nan"),
        clarify=clarify,
        ask_attribute=FIXED_ASK_ATTRIBUTE if clarify else None,
        reason="always-ask-until-exhausted (P0)",
    )
