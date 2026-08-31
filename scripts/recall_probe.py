"""Recall@K probe: for each session, was the target ever in the retrieval candidate
pool at any turn, independent of ranking? Splits official misses into retrieval
failures (target never entered the pool) vs ranking failures (it entered the pool but
never made the scored Top-10).

Usage:
    python -m scripts.recall_probe --name bm25_baseline --agent-import starter.agent:Agent --k 1000

Probes by asking the agent for `top_k` recommendations instead of the official 10 (see
scripts/harness.py for the top_k-invariance assumption this relies on). Never edits
evaluator/ or data/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, load_jsonl

from scripts.harness import load_agent, run_session


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recall@K probe: is the target ever in the candidate pool, independent of ranking?"
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--agent-import", default="starter.agent:Agent")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--k", type=int, default=1000, help="Candidate pool depth to probe")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = load_agent(args.agent_import, args.catalog)

    sessions = [run_session(agent, sample, catalog_ids, categories, products, top_k=args.k) for sample in samples]

    total = len(sessions)
    hits = sum(1 for s in sessions if s["hit"])
    misses = [s for s in sessions if not s["hit"]]
    retrieval_failures = [s for s in misses if not s["recall_hit"]]
    ranking_failures = [s for s in misses if s["recall_hit"]]

    summary = {
        "config": args.name,
        "k": args.k,
        "sample_count": total,
        "official_hit_rate_at_10": hits / total if total else 0.0,
        "recall_at_k": sum(1 for s in sessions if s["recall_hit"]) / total if total else 0.0,
        "miss_count": len(misses),
        "retrieval_failure_count": len(retrieval_failures),
        "retrieval_failure_fraction_of_misses": (len(retrieval_failures) / len(misses)) if misses else 0.0,
        "ranking_failure_count": len(ranking_failures),
        "ranking_failure_fraction_of_misses": (len(ranking_failures) / len(misses)) if misses else 0.0,
    }

    output_path = Path(args.output) if args.output else Path("runs") / f"recall_probe_{args.name}.json"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps({**summary, "sessions": sessions}, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
