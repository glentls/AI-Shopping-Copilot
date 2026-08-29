"""One-time generator for eval/dev_holdout_split.json. Not run as part of any eval loop -- the
committed output file is the source of truth, so nobody can accidentally re-shuffle the holdout
by re-running this script. Stratifies by scenario_type so both splits keep the same 40/40/15/5
Buying/Browsing/Override/Boundary mix as the full public set (docs/competition_specification.md:23-29).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from src.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    config = load_config()
    split_cfg = config["eval"]["dev_holdout"]
    seed = split_cfg["split_seed"]
    dev_count = split_cfg["dev_count"]
    holdout_count = split_cfg["holdout_count"]

    samples = []
    with (REPO_ROOT / "data" / "public_set.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                samples.append(json.loads(line))

    by_scenario: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        by_scenario[sample["scenario_type"]].append(sample["sample_id"])

    rng = random.Random(seed)
    dev_ids: list[str] = []
    holdout_ids: list[str] = []
    total = len(samples)
    for scenario, ids in sorted(by_scenario.items()):
        ids = sorted(ids)  # deterministic order before shuffling
        rng.shuffle(ids)
        scenario_holdout_n = round(len(ids) * holdout_count / total)
        holdout_ids.extend(ids[:scenario_holdout_n])
        dev_ids.extend(ids[scenario_holdout_n:])

    # Reconcile rounding drift against the exact configured counts by moving IDs deterministically.
    dev_ids.sort()
    holdout_ids.sort()
    while len(dev_ids) > dev_count and holdout_ids:
        holdout_ids.append(dev_ids.pop())
        holdout_ids.sort()
    while len(holdout_ids) > holdout_count and dev_ids:
        dev_ids.append(holdout_ids.pop())
        dev_ids.sort()

    assert len(dev_ids) == dev_count, (len(dev_ids), dev_count)
    assert len(holdout_ids) == holdout_count, (len(holdout_ids), holdout_count)
    assert set(dev_ids).isdisjoint(holdout_ids)
    assert set(dev_ids) | set(holdout_ids) == {s["sample_id"] for s in samples}

    output = {
        "seed": seed,
        "dev_count": len(dev_ids),
        "holdout_count": len(holdout_ids),
        "dev": dev_ids,
        "holdout": holdout_ids,
    }
    out_path = REPO_ROOT / "eval" / "dev_holdout_split.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path} -- dev={len(dev_ids)} holdout={len(holdout_ids)}")


if __name__ == "__main__":
    main()
