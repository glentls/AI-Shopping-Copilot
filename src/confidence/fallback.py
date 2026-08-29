"""Fallback path: never raise, always return a top-10.

When the reranker/retrieval yields an empty pool or throws, the agent must
still emit recommendations. We fall back to a popularity ordering computed once
from the catalog (rating_number desc, then average_rating desc).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.confidence.session_ledger import SessionLedger
from src.confidence.payload import ConfidencePayload
from src.confidence.policy import FIXED_ASK_ATTRIBUTE


def popularity_top10(catalog_path: str | Path) -> list[str]:
    """Compute the popularity fallback list (top 10 parent_asin)."""
    rows: list[tuple[float, float, str]] = []
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            p = json.loads(line)
            rating_number = float(p.get("rating_number") or 0)
            average_rating = float(p.get("average_rating") or 0)
            rows.append((rating_number, average_rating, str(p["parent_asin"])))
    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
    return [asin for _, _, asin in rows[:10]]


def safe_decide(
    rank_fn,
    ledger: SessionLedger,
    fallback_recs: list[str],
    theta: float,
    policy: str = "always_ask",
) -> tuple[ConfidencePayload, list[str]]:
    """Run ranking + policy, guaranteeing no raise.

    ``rank_fn`` is a zero-arg callable returning a ``RankResult``. On any
    exception or empty pool we return the popularity fallback with conf=0 and
    clarify=True. ``policy`` selects the clarify decision: ``"always_ask"``
    (the ship-gate champion arm, see ``scripts/sweep_confidence.py``) or
    ``"confidence"`` (the coverage-based ``decide`` heuristic, gated by
    ``theta``). Returns ``(payload, recommendations)``.
    """
    # Local import to avoid cycles at import time.
    from src.confidence.policy import always_ask, decide

    try:
        rank = rank_fn()
    except Exception:
        rank = None

    if rank is None or rank.pool_size <= 0 or not rank.ranked:
        payload = ConfidencePayload(
            score=0.0,
            clarify=True,
            ask_attribute=FIXED_ASK_ATTRIBUTE,
            reason="empty pool / rank failure -> popularity fallback",
        )
        return payload, list(fallback_recs[:10])

    payload = always_ask(ledger) if policy == "always_ask" else decide(rank, ledger, theta=theta)
    return payload, list(rank.ranked[:10])
