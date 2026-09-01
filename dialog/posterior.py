"""Tracks (material, color) value frequency across previously shown-and-rejected
batches, per session. A session reaching turn N is proof turn N-1's items were rejected
(the evaluator stops on a hit -- evaluator/local_evaluator.py:252-255), so recording the
previous turn's shown batch here, before computing turn N's recommendations, is a valid
causal signal, not a guess."""

from __future__ import annotations

from collections import Counter

from dialog.portfolio import _feature_values


class RejectionTracker:
    def __init__(self) -> None:
        self.counts: Counter = Counter()

    def record_rejected_batch(self, asins: list[str], products: dict[str, dict]) -> None:
        for asin in asins:
            self.counts.update(_feature_values(products.get(asin, {})))
