"""Build every artifact the agent needs, from a clean checkout.

LANE B OWNS THIS FILE. Right now it builds Lane A's attribute table only.
Lane B extends it to fetch/vendor the ONNX embedding model and precompute the
50,000 product vectors.

    python3 -m tools.build_index

Artifacts go to artifacts/ which is gitignored. Commit this script, never its
output.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.attributes import build_attribute_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build agent artifacts")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    started = time.perf_counter()
    table = build_attribute_table(args.catalog, args.artifacts)
    elapsed = time.perf_counter() - started

    print(f"attribute table built in {elapsed:.1f}s -> {Path(args.artifacts) / 'attributes.json'}")
    for slot in table.slots():
        print(f"  {slot:10} coverage {table.coverage(slot):6.1%}")


if __name__ == "__main__":
    main()
