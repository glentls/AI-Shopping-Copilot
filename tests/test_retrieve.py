"""
test_retrieve.py — manual validation of Agent.retrieve() against known catalog entries.

Run from the project root:
    python -m starter.test_retrieve

Each test picks a real product from catalog.jsonl, builds a search_key from its
own fields, and asserts the product's parent_asin appears somewhere in the results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starter.agent import Agent

CATALOG = Path("data/catalog.jsonl")
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def load_product(parent_asin: str) -> dict:
    with CATALOG.open(encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["parent_asin"] == parent_asin:
                return p
    raise KeyError(f"{parent_asin} not found in catalog")


def run_test(label: str, agent: Agent, search_key: dict, expected_asin: str, top_k: int = 20) -> bool:
    results = agent.retrieve(search_key, top_k=top_k)
    hit = expected_asin in results
    rank = results.index(expected_asin) + 1 if hit else None
    status = PASS if hit else FAIL
    rank_str = f"rank {rank}/{top_k}" if hit else f"not in top {top_k}"
    print(f"  [{status}] {label}")
    print(f"         expected : {expected_asin}")
    print(f"         result   : {rank_str}")
    print(f"         top-5    : {results[:5]}")
    return hit


def main() -> None:
    print("Loading agent (building in-memory index)...")
    agent = Agent()
    print("Ready.\n")

    results: list[bool] = []

    # ------------------------------------------------------------------
    # Test 1: Columbia Men's T-Shirt  (B07KCFS4VC)
    #   known fields: store=Columbia, categories includes T-Shirts, Men
    # ------------------------------------------------------------------
    results.append(run_test(
        label="Columbia Men's T-shirt — store + category",
        agent=agent,
        search_key={
            "store": ["Columbia"],
            "category": ["T-Shirts", "Men"],
        },
        expected_asin="B07KCFS4VC",
    ))

    # ------------------------------------------------------------------
    # Test 2: Spirit Hoops Earrings  (B07K34RX5J)
    #   known fields: store=Spirit Hoops, categories includes Earrings, Hoop
    # ------------------------------------------------------------------
    results.append(run_test(
        label="Spirit Hoops Earrings — store + category",
        agent=agent,
        search_key={
            "store": ["Spirit Hoops"],
            "category": ["Earrings", "Hoop"],
        },
        expected_asin="B07K34RX5J",
    ))

    # ------------------------------------------------------------------
    # Test 3: Hylaea Running Socks  (B095PZG4SR)
    #   known fields: title words 'running socks', 'cushion', 'moisture wicking'
    # ------------------------------------------------------------------
    results.append(run_test(
        label="Hylaea Running Socks — title keywords",
        agent=agent,
        search_key={
            "product_type": ["running socks"],
            "feature": ["cushion", "moisture wicking"],
        },
        expected_asin="B095PZG4SR",
    ))

    # ------------------------------------------------------------------
    # Test 4: Nupuyai Green Aventurine Necklace  (B08LDFVQXV)
    #   known fields: title includes 'aventurine necklace', 'pendant'
    # ------------------------------------------------------------------
    results.append(run_test(
        label="Nupuyai Green Aventurine Necklace — material + type",
        agent=agent,
        search_key={
            "material": ["aventurine"],
            "type": ["necklace", "pendant"],
        },
        expected_asin="B08LDFVQXV",
    ))

    # ------------------------------------------------------------------
    # Test 5: Pink Satin Costume Jacket  (B08VDM4G8B)
    #   known fields: title includes 'pink satin jacket', 'halloween costume'
    # ------------------------------------------------------------------
    results.append(run_test(
        label="Pink Satin Halloween Jacket — color + material + occasion",
        agent=agent,
        search_key={
            "color": ["pink"],
            "material": ["satin"],
            "occasion": ["halloween"],
        },
        expected_asin="B08VDM4G8B",
    ))

    # ------------------------------------------------------------------
    # Test 6: Empty search_key — should return []
    # ------------------------------------------------------------------
    empty_result = agent.retrieve({}, top_k=10)
    ok = empty_result == []
    status = PASS if ok else FAIL
    print(f"  [{status}] Empty search_key returns empty list — got: {empty_result}")
    results.append(ok)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
