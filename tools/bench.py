"""Run the public set and report metrics by scenario.

LANE C OWNS THIS FILE, and owns it first -- the other two lanes measure with
it, so it needs to land on main ahead of everything else.

    python3 -m tools.bench
    python3 -m tools.bench --failures 5      # dump failed session transcripts
    python3 -m tools.bench --compare a.json b.json

Still to build (Lane C): per-turn first-hit distribution, transcript dumps for
failures, and side-by-side A/B of two result files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent

SCENARIOS = ("buying", "browsing", "intent_override", "boundary")


def run(catalog: str, dataset: str, output: str) -> dict:
    samples = load_jsonl(dataset)
    catalog_ids, categories, products = catalog_index(catalog)
    result = evaluate(Agent(catalog), samples, catalog_ids, categories, products)
    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def report(result: dict) -> None:
    print(f"\n  TechnicalScore  {result['recommended_technical_score']:.4f}")
    print(f"  HitRate@10      {result['hit_rate_at_10']:.4f}")
    print(f"  MRR             {result['mrr']:.4f}")
    print(f"  MTTC            {result['mttc']:.3f}   (efficiency {result['efficiency']:.4f})")
    print(f"\n  {'scenario':16} {'n':>4} {'hit':>8} {'mrr':>8} {'mttc':>7}")
    for name in SCENARIOS:
        m = result["scenario_metrics"].get(name)
        if not m:
            continue
        print(f"  {name:16} {m['sample_count']:4} {m['hit_rate_at_10']:8.3f} "
              f"{m['mrr']:8.3f} {m['mttc']:7.2f}")
    usage = result["reported_token_usage"]
    print(f"\n  tokens          {usage['total_tokens']}")


def compare(left: str, right: str) -> None:
    a = json.loads(Path(left).read_text())
    b = json.loads(Path(right).read_text())
    print(f"\n  {'metric':16} {Path(left).stem:>12} {Path(right).stem:>12} {'delta':>10}")
    for key in ("recommended_technical_score", "hit_rate_at_10", "mrr", "mttc"):
        print(f"  {key:16} {a[key]:12.4f} {b[key]:12.4f} {b[key] - a[key]:+10.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Public-set benchmark")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--compare", nargs=2, metavar=("BASE", "NEW"))
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    report(run(args.catalog, args.dataset, args.output))


if __name__ == "__main__":
    main()
