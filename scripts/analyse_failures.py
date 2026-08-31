"""For every session the agent misses, dump the target's catalog record, the full
per-turn transcript, and the top-10 actually shown -- dense enough to read fifty
failures in ten minutes.

Usage:
    python -m scripts.analyse_failures --name bm25_baseline --agent-import starter.agent:Agent

Runs at top_k=10, reproducing the official evaluator's dialogue exactly (see
scripts/harness.py). Never edits evaluator/ or data/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluator.local_evaluator import catalog_index, load_jsonl

from scripts.harness import load_agent, run_session


def format_product(product: dict) -> str:
    if not product:
        return "*(target parent_asin not found in catalog)*"
    title = product.get("title", "?")
    price = product.get("price", "?")
    categories = product.get("categories") or []
    features = product.get("features") or []
    cat_str = " > ".join(str(c) for c in categories[-2:]) if categories else "-"
    feat_str = "; ".join(str(f) for f in features[:2]) if features else "-"
    return f"**{title}** (${price}) [{cat_str}] -- {feat_str}"


def format_turn(turn: dict) -> str:
    shown = ", ".join(turn["recommendations_shown"][:5]) or "-"
    ask = turn["ask_attribute"] or "-"
    message = turn["message"][:80].replace("\n", " ")
    user_message = turn["user_message"][:80].replace("\n", " ")
    return f"  - turn {turn['turn']}: asked=`{ask}` said=\"{message}\" user=\"{user_message}\" top5=[{shown}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump per-session detail for every missed session")
    parser.add_argument("--name", required=True)
    parser.add_argument("--agent-import", default="starter.agent:Agent")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = load_agent(args.agent_import, args.catalog)

    output_path = Path(args.output) if args.output else Path("runs") / f"failures_{args.name}.md"
    output_path.parent.mkdir(exist_ok=True)

    body_lines: list[str] = []
    total = 0
    miss_count = 0
    for sample in samples:
        total += 1
        session = run_session(agent, sample, catalog_ids, categories, products, top_k=10)
        if session["hit"]:
            continue
        miss_count += 1
        target_product = products.get(session["target"], {})
        body_lines.append(f"## {session['sample_id']} ({session['scenario_type']}) -- target `{session['target']}`")
        body_lines.append(f"- target: {format_product(target_product)}")
        body_lines.extend(format_turn(turn) for turn in session["turns"])
        body_lines.append("")

    header = f"# Failure dump: {args.name}\n\n{miss_count}/{total} sessions missed.\n\n"
    output_path.write_text(header + "\n".join(body_lines), encoding="utf-8")
    print(f"{miss_count}/{total} missed. Wrote {output_path}")


if __name__ == "__main__":
    main()
