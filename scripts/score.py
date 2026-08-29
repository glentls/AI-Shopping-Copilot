"""Compact evaluator runner for A/B comparing agent variants.

Usage:
    python3 scripts/score.py [label]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import Agent  # noqa: E402


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    samples = load_jsonl("data/public_set.jsonl")
    catalog_ids, categories, products = catalog_index("data/catalog.jsonl")
    result = evaluate(Agent("data/catalog.jsonl"), samples, catalog_ids, categories, products)

    print(f"\n=== {label} ===")
    print(
        f"score={result['recommended_technical_score']:.4f}  "
        f"hit={result['hit_rate_at_10']:.4f}  "
        f"mrr={result['mrr']:.4f}  "
        f"mttc={result['mttc']:.3f}"
    )
    for name, metrics in sorted(result["scenario_metrics"].items()):
        print(
            f"  {name:<16} hit={metrics['hit_rate_at_10']:.3f} "
            f"mrr={metrics['mrr']:.3f} mttc={metrics['mttc']:.2f} (n={metrics['sample_count']})"
        )

    sessions = result["sessions"]
    ranks = Counter(s["best_rank"] if s["hit"] else "miss" for s in sessions)
    top1 = ranks.get(1, 0)
    print(f"  rank1={top1}  rank2-10={sum(v for k, v in ranks.items() if k != 'miss') - top1}  miss={ranks.get('miss', 0)}")
    turns = Counter(s["first_hit_turn"] for s in sessions if s["hit"])
    print("  hits by turn: " + ", ".join(f"t{k}={turns[k]}" for k in sorted(turns)))

    Path(f"results_{label}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
