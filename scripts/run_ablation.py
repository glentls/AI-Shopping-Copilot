"""Run one named agent config through the unmodified official evaluator, append its row
to docs/ablations.md, and archive the raw results.json under runs/<name>.json.

Usage:
    python -m scripts.run_ablation --name bm25_baseline --agent-import starter.agent:Agent

Never edits evaluator/ or data/: it imports evaluator.local_evaluator.evaluate() as-is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

from scripts.harness import load_agent

RUNS_DIR = Path("runs")
ABLATIONS_PATH = Path("docs/ablations.md")

MAIN_HEADER = "| Config | Hit@10 | MRR | MTTC | Efficiency | TechnicalScore | N |"
MAIN_SEP = "|---|---|---|---|---|---|---|"
SCENARIO_HEADER = "| Config | Buying Hit@10 | Browsing Hit@10 | Intent Override Hit@10 | Boundary Hit@10 |"
SCENARIO_SEP = "|---|---|---|---|---|"


def format_row(name: str, result: dict) -> str:
    return (
        f"| {name} | {result['hit_rate_at_10']:.4f} | {result['mrr']:.4f} | "
        f"{result['mttc']:.4f} | {result['efficiency']:.4f} | "
        f"{result['recommended_technical_score']:.4f} | {result['sample_count']} |"
    )


def format_scenario_row(name: str, scenario_metrics: dict) -> str:
    def hit(scenario: str) -> str:
        metrics = scenario_metrics.get(scenario)
        return f"{metrics['hit_rate_at_10']:.4f}" if metrics else "-"

    return f"| {name} | {hit('buying')} | {hit('browsing')} | {hit('intent_override')} | {hit('boundary')} |"


def upsert_table(existing: str, header: str, sep: str, new_row: str, config_name: str) -> str:
    """Insert new_row under `header`, replacing any prior row for the same config name
    (so reruns of the same --name update in place instead of duplicating)."""
    row_key = f"| {config_name} |"
    lines = existing.splitlines()
    if header not in lines:
        prefix = existing.rstrip("\n")
        block = "\n".join([header, sep, new_row])
        return (prefix + "\n\n" + block + "\n") if prefix else (block + "\n")
    header_idx = lines.index(header)
    row_idx = header_idx + 2
    while row_idx < len(lines) and lines[row_idx].startswith("|"):
        if lines[row_idx].startswith(row_key):
            lines[row_idx] = new_row
            return "\n".join(lines) + "\n"
        row_idx += 1
    lines.insert(row_idx, new_row)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one named ablation config through the official evaluator")
    parser.add_argument("--name", required=True, help="Config label, e.g. 'bm25_baseline'")
    parser.add_argument(
        "--agent-import", default="starter.agent:Agent", help="module:ClassName, constructed as Class(catalog_path)"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--agent-kwargs", default="{}", help="JSON kwargs passed to the Agent constructor, e.g. '{\"config_path\": \"configs/retrieval_lexical_only.json\"}'")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = load_agent(args.agent_import, args.catalog, **json.loads(args.agent_kwargs))
    result = evaluate(agent, samples, catalog_ids, categories, products)

    RUNS_DIR.mkdir(exist_ok=True)
    run_path = RUNS_DIR / f"{args.name}.json"
    run_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    ABLATIONS_PATH.parent.mkdir(exist_ok=True)
    existing = ABLATIONS_PATH.read_text(encoding="utf-8") if ABLATIONS_PATH.exists() else "# Ablations\n"
    existing = upsert_table(existing, MAIN_HEADER, MAIN_SEP, format_row(args.name, result), args.name)
    existing = upsert_table(existing, SCENARIO_HEADER, SCENARIO_SEP, format_scenario_row(args.name, result["scenario_metrics"]), args.name)
    ABLATIONS_PATH.write_text(existing, encoding="utf-8")

    summary = {key: value for key, value in result.items() if key != "sessions"}
    print(json.dumps(summary, indent=2))
    print(f"\nArchived raw results -> {run_path}")
    print(f"Updated {ABLATIONS_PATH}")


if __name__ == "__main__":
    main()
