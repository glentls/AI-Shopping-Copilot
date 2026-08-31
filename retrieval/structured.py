"""Structured route: soft score penalties/boosts over category, price band, and store
(used as the brand signal -- see retrieval/catalog.py docstring for why `store`, not
`details.Brand`). Deliberately never excludes: with 79% of the catalog missing `price`,
a hard price filter would delete most of the catalog outright, and CLAUDE.md's Phase 2
spec calls for soft filtering by default -- reserve hard exclusion for constraints
stated explicitly and unambiguously, which this route doesn't attempt to detect.

Only contributes a ranked list to fusion when at least one constraint actually matched
something; an all-zero-score route would otherwise inject arbitrary catalog order as a
fourth fusion signal.
"""

from __future__ import annotations

import re

from retrieval.catalog import price_of, store_of

MATERIAL_TERMS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon",
    "denim", "suede", "canvas", "linen", "fleece", "cashmere", "velvet",
)
COLOR_TERMS = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "navy", "beige", "tan", "gold", "silver", "cream",
)
BUDGET_RE = re.compile(r"\$\s?(\d+(?:\.\d+)?)")


def extract_constraints(query_text: str) -> dict:
    lowered = query_text.lower()
    budget_match = BUDGET_RE.search(query_text)
    return {
        "materials": [term for term in MATERIAL_TERMS if term in lowered],
        "colors": [term for term in COLOR_TERMS if term in lowered],
        "budget": float(budget_match.group(1)) if budget_match else None,
        "lowered_text": lowered,
    }


class StructuredRetriever:
    def __init__(self, products: dict[str, dict]):
        self._products = products
        self._searchable = {
            asin: f"{product.get('title', '')} {' '.join(product.get('features') or [])}".lower()
            for asin, product in products.items()
        }
        self._price = {asin: price_of(product) for asin, product in products.items()}
        self._store_lower = {asin: store_of(product).lower() for asin, product in products.items()}

    def _score(self, asin: str, constraints: dict) -> float:
        score = 0.0
        text = self._searchable[asin]
        for term in constraints["materials"]:
            if term in text:
                score += 1.0
        for term in constraints["colors"]:
            if term in text:
                score += 1.0

        budget = constraints.get("budget")
        price = self._price[asin]
        if budget is not None and price is not None:
            if price <= budget:
                score += 0.5
            else:
                overage = (price - budget) / budget
                score -= min(overage, 1.0) * 0.5  # capped soft penalty, never excludes

        store = self._store_lower[asin]
        if store and store in constraints["lowered_text"]:
            score += 1.0

        return score

    def search(self, query_text: str, k: int) -> list[tuple[str, float]]:
        constraints = extract_constraints(query_text)
        if not constraints["materials"] and not constraints["colors"] and constraints["budget"] is None:
            return []
        scored = [(asin, self._score(asin, constraints)) for asin in self._products]
        scored = [pair for pair in scored if pair[1] > 0]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
