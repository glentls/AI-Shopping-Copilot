"""
test_retrieve.py — validation of Agent.retrieve() against known catalog entries.

Run from the project root:
    python -m tests.test_retrieve

Covers:
  - Text-only search (existing behaviour)
  - Numeric-only filters  (price / average_rating / rating_number)
  - Combined text + numeric
  - Edge cases: empty search_key, null-price exclusion, range (gte + lte)
"""

from __future__ import annotations

import json
import operator
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starter.agent import Agent

CATALOG = Path("data/catalog.jsonl")
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_PY_OPS = {
    "lt":  operator.lt,
    "lte": operator.le,
    "gt":  operator.gt,
    "gte": operator.ge,
    "eq":  operator.eq,
}


# ── helpers ──────────────────────────────────────────────────────────────────

def run_test(label: str, agent: Agent, search_key: dict, expected_asin: str, top_k: int = 20) -> bool:
    """Assert that expected_asin appears somewhere in the top-k results."""
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


def run_numeric_test(
    label: str,
    agent: Agent,
    search_key: dict,
    checks: list[tuple[str, str, float]],
    top_k: int = 10,
    require_nonempty: bool = True,
) -> bool:
    """
    Assert that every returned ASIN satisfies all numeric checks.

    checks: list of (field, op, threshold) tuples
            e.g. [("price", "lte", 50.0), ("average_rating", "gte", 4.0)]
    """
    results = agent.retrieve(search_key, top_k=top_k)
    violations: list[str] = []

    if require_nonempty and not results:
        print(f"  [{FAIL}] {label}")
        print(f"         result   : empty (expected at least one hit)")
        return False

    for asin in results:
        meta = agent._numeric.get(asin, {})
        for field, op, threshold in checks:
            val = meta.get(field)
            if val is None:
                violations.append(f"{asin}: {field}=null failed {op} {threshold}")
                continue
            if not _PY_OPS[op](float(val), float(threshold)):
                violations.append(f"{asin}: {field}={val} failed {op} {threshold}")

    ok = len(violations) == 0
    status = PASS if ok else FAIL
    print(f"  [{status}] {label}")
    print(f"         returned : {len(results)} results, checked {len(checks)} condition(s) each")
    if violations:
        for v in violations[:3]:
            print(f"         VIOLATION: {v}")
    else:
        # Show a sample of what was returned with the relevant field values
        sample = results[:3]
        for asin in sample:
            vals = {f: agent._numeric.get(asin, {}).get(f) for f, _, _ in checks}
            print(f"         {asin} -> {vals}")
    return ok


# ── test suite ────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading agent (building in-memory index)...")
    agent = Agent()
    print("Ready.\n")

    results: list[bool] = []

    print("── Text search ──────────────────────────────────────────────────")

    # T1: Columbia Men's T-Shirt (B07KCFS4VC) — price=27.99, rating=4.7
    results.append(run_test(
        label="Columbia Men's T-shirt — store + category",
        agent=agent,
        search_key={"store": ["Columbia"], "category": ["T-Shirts", "Men"]},
        expected_asin="B07KCFS4VC",
    ))

    # T2: Spirit Hoops Earrings (B07K34RX5J) — price=null, rating=4.1
    results.append(run_test(
        label="Spirit Hoops Earrings — store + category",
        agent=agent,
        search_key={"store": ["Spirit Hoops"], "category": ["Earrings", "Hoop"]},
        expected_asin="B07K34RX5J",
    ))

    # T3: Hylaea Running Socks (B095PZG4SR) — price=22.99, rating=4.5
    results.append(run_test(
        label="Hylaea Running Socks — title keywords",
        agent=agent,
        search_key={"product_type": ["running socks"], "feature": ["cushion", "moisture wicking"]},
        expected_asin="B095PZG4SR",
    ))

    # T4: Nupuyai Aventurine Necklace (B08LDFVQXV) — price=15.49, rating=4.6
    results.append(run_test(
        label="Nupuyai Aventurine Necklace — material + type",
        agent=agent,
        search_key={"material": ["aventurine"], "type": ["necklace", "pendant"]},
        expected_asin="B08LDFVQXV",
    ))

    # T5: Pink Satin Halloween Jacket (B08VDM4G8B) — price=22.99, rating=4.6
    results.append(run_test(
        label="Pink Satin Halloween Jacket — color + material + occasion",
        agent=agent,
        search_key={"color": ["pink"], "material": ["satin"], "occasion": ["halloween"]},
        expected_asin="B08VDM4G8B",
    ))

    print("\n── Numeric-only filters ─────────────────────────────────────────")

    # N1: price <= 20 — every result must have price ≤ 20
    results.append(run_numeric_test(
        label="price <= 20 (numeric-only, no text)",
        agent=agent,
        search_key={"price": [{"lte": 20.0}]},
        checks=[("price", "lte", 20.0)],
        top_k=20,
    ))

    # N2: average_rating >= 4.8 — only highly-rated products
    results.append(run_numeric_test(
        label="average_rating >= 4.8 (numeric-only)",
        agent=agent,
        search_key={"average_rating": [{"gte": 4.8}]},
        checks=[("average_rating", "gte", 4.8)],
        top_k=20,
    ))

    # N3: price range 10 <= price <= 20
    results.append(run_numeric_test(
        label="price range 10–20 (gte + lte on same field)",
        agent=agent,
        search_key={"price": [{"gte": 10.0}, {"lte": 20.0}]},
        checks=[("price", "gte", 10.0), ("price", "lte", 20.0)],
        top_k=20,
    ))

    print("\n── Combined text + numeric ──────────────────────────────────────")

    # C1: jackets under $30 — text "jacket" + price <= 30
    results.append(run_numeric_test(
        label="jacket under $30 — text + price filter",
        agent=agent,
        search_key={"type": ["jacket"], "price": [{"lte": 30.0}]},
        checks=[("price", "lte", 30.0)],
        top_k=10,
    ))

    # C2: Columbia T-shirt (B07KCFS4VC, $27.99) must survive price <= 30 filter
    results.append(run_test(
        label="Columbia T-shirt ($27.99) survives price <= 30 filter",
        agent=agent,
        search_key={"store": ["Columbia"], "category": ["T-Shirts"], "price": [{"lte": 30.0}]},
        expected_asin="B07KCFS4VC",
        top_k=20,
    ))

    # C3: Nupuyai necklace ($15.49, rating 4.6) — text + price < 20 + rating >= 4.5
    results.append(run_test(
        label="Nupuyai necklace survives price < 20 AND rating >= 4.5",
        agent=agent,
        search_key={
            "material": ["aventurine"],
            "type": ["necklace"],
            "price": [{"lt": 20.0}],
            "average_rating": [{"gte": 4.5}],
        },
        expected_asin="B08LDFVQXV",
        top_k=20,
    ))

    # C4: Spirit Hoops earrings have price=null — must be excluded by any price filter
    spirit_with_price_filter = agent.retrieve(
        {"store": ["Spirit Hoops"], "price": [{"lte": 999.0}]}, top_k=20
    )
    excluded = "B07K34RX5J" not in spirit_with_price_filter
    status = PASS if excluded else FAIL
    print(f"  [{status}] Null-price product excluded when price filter applied")
    print(f"         B07K34RX5J (price=null) in results: {not excluded}")
    results.append(excluded)

    print("\n── Edge cases ───────────────────────────────────────────────────")

    # E1: empty search_key
    empty_result = agent.retrieve({}, top_k=10)
    ok = empty_result == []
    print(f"  [{PASS if ok else FAIL}] Empty search_key returns [] — got: {empty_result}")
    results.append(ok)

    # E2: unknown attribute name (treated as text search, produces valid results or [])
    unknown = agent.retrieve({"xyzzy_nonexistent": ["foo bar"]}, top_k=5)
    ok = isinstance(unknown, list)
    print(f"  [{PASS if ok else FAIL}] Unknown attribute key returns a list — got: {unknown[:3]}")
    results.append(ok)

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*55}")
    print(f"Results: {passed}/{total} tests passed")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
