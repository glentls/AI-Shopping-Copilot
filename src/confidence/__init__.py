"""Confidence check component.

Confidence gates the *question*, never the products: every turn still returns a
top-10; the confidence score only decides whether to attach a clarifying
``ask_attribute``.

Public API:
    compute_confidence(rank, n_constraints_known) -> (score, reason)
    decide(rank, ledger, theta)                   -> ConfidencePayload
    always_ask(ledger)                            -> ConfidencePayload  # P0 arm
    safe_decide(rank_fn, ledger, fallback, theta) -> (payload, recs)    # no-raise
    SessionLedger                                 # per-session state
    ConfidencePayload                             # per-turn result
"""

from src.confidence.confidence import compute_confidence
from src.confidence.fallback import popularity_top10, safe_decide
from src.confidence.session_ledger import SessionLedger
from src.confidence.payload import ConfidencePayload
from src.confidence.policy import DEFAULT_THETA, always_ask, decide

__all__ = [
    "compute_confidence",
    "decide",
    "always_ask",
    "safe_decide",
    "popularity_top10",
    "SessionLedger",
    "ConfidencePayload",
    "DEFAULT_THETA",
]
