"""Sanity probe: does the reranker actually find targets on the 200 public sessions?

Reconstructs each session's intent card (same logic as the evaluator) and feeds
the FULL set of disclosed constraints to the reranker, then reports target
HitRate@10 and median target rank. This is a read-only measurement to confirm
the reranker produces real hit-signal before we calibrate confidence.

Run: python3 -m scripts.probe_reranker
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from evaluator.local_evaluator import intent_card
from src.reranker import build_reranker, default_query

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"


def main() -> None:
    products: dict[str, dict] = {}
    with Path(CATALOG).open(encoding="utf-8") as fh:
        for line in fh:
            p = json.loads(line)
            products[str(p["parent_asin"])] = p

    rr = build_reranker(CATALOG)
    sessions = [json.loads(l) for l in Path(DATASET).open(encoding="utf-8") if l.strip()]

    ranks: list[int | None] = []
    by_scenario: dict[str, list[int | None]] = {}
    for s in sessions:
        target = str(s["ground_truth"]["parent_asin"])
        card = intent_card(products[target])
        constraints = list(dict.fromkeys(card["hard_constraints"] + card["soft_preferences"]))
        query = default_query(constraints, card["target_category"])
        res = rr.rank(query, constraints)
        rank = res.ranked.index(target) + 1 if target in res.ranked else None
        ranks.append(rank)
        by_scenario.setdefault(s["scenario_type"], []).append(rank)

    def summarize(rs: list[int | None]) -> str:
        hits = [r for r in rs if r is not None]
        hr = len(hits) / len(rs) if rs else 0.0
        med = statistics.median(hits) if hits else None
        return f"n={len(rs):3d}  HitRate@10={hr:.3f}  median_rank={med}"

    print("== Reranker sanity probe (full-constraint disclosure) ==")
    print("overall :", summarize(ranks))
    for name in sorted(by_scenario):
        print(f"{name:8}:", summarize(by_scenario[name]))


if __name__ == "__main__":
    main()
