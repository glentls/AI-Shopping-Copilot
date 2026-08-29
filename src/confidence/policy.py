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

from src.confidence.confidence import compute_confidence
from src.confidence.session_ledger import SessionLedger
from src.confidence.payload import ConfidencePayload
from src.reranker.types import RankResult

DEFAULT_THETA = 0.5
TURN_CUTOFF = 8            # stop asking at/after this turn
FIXED_ASK_ATTRIBUTE = "other"


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
