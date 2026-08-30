"""Fast/full/holdout eval loops. Reuses evaluator.local_evaluator.evaluate() (never edits it --
docs/submission_rules.md forbids modifying evaluator files) against the committed dev/holdout
split (eval/dev_holdout_split.json) so nobody can tune on the holdout by accident.

Usage:
    py -m eval.run_eval --mode fast      # first 50 of the 150 dev sessions, for quick iteration
    py -m eval.run_eval --mode full      # all 150 dev sessions
    py -m eval.run_eval --mode holdout   # all 50 holdout sessions -- use sparingly, see below
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from src.config import load_config
from src.agent import Agent

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_split() -> dict:
    with (REPO_ROOT / "eval" / "dev_holdout_split.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def select_samples(mode: str, samples_by_id: dict[str, dict], config: dict) -> list[dict]:
    split = _load_split()
    if mode == "fast":
        ids = split["dev"][: config["eval"]["fast_sample_count"]]
    elif mode == "full":
        ids = split["dev"]
    elif mode == "holdout":
        ids = split["holdout"]
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    return [samples_by_id[sample_id] for sample_id in ids]


def main() -> None:
    parser = argparse.ArgumentParser(description="TechJam skeleton eval loop")
    parser.add_argument("--mode", choices=["fast", "full", "holdout"], default="fast")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--log", default="eval/results_log.jsonl")
    args = parser.parse_args()

    if args.mode == "holdout":
        print("WARNING: running on the holdout split. This should be rare -- see CLAUDE.md.")

    config = load_config()
    all_samples = load_jsonl(args.dataset)
    samples_by_id = {sample["sample_id"]: sample for sample in all_samples}
    samples = select_samples(args.mode, samples_by_id, config)

    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog)

    start = time.perf_counter()
    result = evaluate(agent, samples, catalog_ids, categories, products)
    wall_clock_seconds = round(time.perf_counter() - start, 3)

    summary = {key: value for key, value in result.items() if key != "sessions"}
    summary["mode"] = args.mode
    summary["wall_clock_seconds"] = wall_clock_seconds
    print(json.dumps(summary, indent=2))

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        **summary,
    }
    log_path = REPO_ROOT / args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(log_entry) + "\n")


if __name__ == "__main__":
    main()
