"""Grid-sweep retrieval scoring weights against the public set.

Reuses a single catalog index so each configuration costs one evaluation pass.

Usage:
    python3 scripts/sweep.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import Agent  # noqa: E402
from starter.personalization import rerank_boost  # noqa: E402
from starter.retrieval import catalog_store, reranker, search  # noqa: E402

GRID = {
    "TOP_TIER_BONUS": (0.0, 2.0, 4.0, 8.0),
    "TOP_TIER_SIZE": (15, 25, 40),
}
TARGETS = {
    "MAX_PROFILE_BOOST": rerank_boost,
    "CANDIDATE_POOL_SIZE": search,
}


def _apply(name: str, value: object) -> None:
    if name == "BM25_WEIGHTS":
        weights = ", ".join(str(weight) for weight in value)  # type: ignore[union-attr]
        catalog_store.BM25_ORDER_BY = f"bm25(products, {weights})"
        return
    setattr(TARGETS.get(name, reranker), name, value)


def main() -> None:
    samples = load_jsonl("data/public_set.jsonl")
    catalog_ids, categories, products = catalog_index("data/catalog.jsonl")
    agent = Agent("data/catalog.jsonl")

    names = list(GRID)
    rows: list[tuple[float, str, dict]] = []
    for combo in itertools.product(*(GRID[name] for name in names)):
        config = dict(zip(names, combo))
        for name, value in config.items():
            _apply(name, value)
        result = evaluate(agent, samples, catalog_ids, categories, products)
        label = " ".join(f"{name.split('_')[0].lower()}={value}" for name, value in config.items())
        rows.append((result["recommended_technical_score"], label, result))
        print(
            f"{result['recommended_technical_score']:.4f}  "
            f"hit={result['hit_rate_at_10']:.3f} mrr={result['mrr']:.3f} mttc={result['mttc']:.2f}  {label}",
            flush=True,
        )

    rows.sort(key=lambda row: row[0], reverse=True)
    print("\n=== TOP 5 ===")
    for score, label, result in rows[:5]:
        print(f"{score:.4f}  hit={result['hit_rate_at_10']:.3f} mrr={result['mrr']:.3f}  {label}")


if __name__ == "__main__":
    main()
