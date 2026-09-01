"""Portfolio ranking: the Top-10 is scored on P(target present), not mean item quality,
so under attribute uncertainty ten near-duplicates is a strictly worse bet than a
covering set. Rank 1 is always the top fused item (protects MRR -- CLAUDE.md Phase 3b).
Ranks 2..top_k are filled by greedy marginal-coverage over (material, color, price
bucket), a submodular proxy for "how much of the unresolved hypothesis space does adding
this item explain that the selected set doesn't already."

Explore/exploit schedule (CLAUDE.md Phase 3c): diversity weight starts high (partition
the hypothesis space early, so a rejection carries maximum information) and decays with
turn (exploit the narrowed posterior later). Linear, clamped -- there's no ground truth
posterior-entropy signal available cheaply enough to justify a fancier schedule.

Proven-negative downweighting (CLAUDE.md Phase 3a #2): rejected_value_counts is a
frequency count over (material, color) values seen in *previously shown, still-rejected*
batches -- passed in by the caller (dialog/posterior.py), not computed here. A candidate
matching values that dominated a rejected batch is scored down, since the session
continuing after that batch was shown is itself evidence against those values.
"""

from __future__ import annotations

from collections import Counter

from dialog.slots import _extract_color, _extract_material
from retrieval.catalog import price_of


FIXED_DIVERSITY_WEIGHT = 0.35  # used when the turn-based schedule (explore/exploit) is off


def _diversity_weight(turn: int, use_schedule: bool) -> float:
    if not use_schedule:
        return FIXED_DIVERSITY_WEIGHT
    return max(0.1, min(0.6, 0.65 - 0.05 * turn))


def _price_bucket(price: float | None) -> str | None:
    if price is None:
        return None
    if price < 25:
        return "budget"
    if price < 75:
        return "mid"
    return "premium"


def _feature_values(product: dict) -> set[str]:
    text = f"{product.get('title', '')} {' '.join(product.get('features') or [])}"
    values = {_extract_material(text), _extract_color(text), _price_bucket(price_of(product))}
    values.discard(None)
    return values  # type: ignore[return-value]


def portfolio_rerank(
    fused_asins: list[str],
    products: dict[str, dict],
    top_k: int,
    turn: int,
    rejected_value_counts: Counter | None = None,
    use_schedule: bool = True,
) -> list[str]:
    if not fused_asins:
        return []
    rejected_value_counts = rejected_value_counts or Counter()
    max_rejection = max(rejected_value_counts.values(), default=0) or 1

    selected = [fused_asins[0]]
    selected_features: set[str] = _feature_values(products.get(fused_asins[0], {}))
    remaining = fused_asins[1:]
    diversity_weight = _diversity_weight(turn, use_schedule)

    while len(selected) < top_k and remaining:
        best_asin = None
        best_score = float("-inf")
        for rank, asin in enumerate(remaining, start=1):
            product = products.get(asin, {})
            features = _feature_values(product)
            relevance = 1.0 / rank
            novelty = len(features - selected_features)
            rejection_penalty = sum(rejected_value_counts.get(v, 0) for v in features) / max_rejection
            score = relevance * (1 - diversity_weight) + diversity_weight * novelty - 0.3 * rejection_penalty
            if score > best_score:
                best_score = score
                best_asin = asin
        selected.append(best_asin)
        selected_features |= _feature_values(products.get(best_asin, {}))
        remaining.remove(best_asin)

    return selected
