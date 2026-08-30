"""Build an 800-session synthetic development set from the frozen catalog.

A session row carries only a target ``parent_asin`` plus a safe profile; the
customer's words are produced at run time (see ``tools/customer_sim.py`` and
``evaluator/local_evaluator.py``). So a valid extra test case is just a target
plus labels, and the interesting question is *which* targets to pick.

Difficulty here is measured, not asserted. For each candidate product we score
four signals and split the pool into exact terciles:

    popularity        log1p(rating_number) - obscure products are harder
    distinctiveness   max IDF over title tokens - a rare token is a handle
    category room     -log(products sharing the coarse category) - crowded is harder
    facet richness    how many typed constraints the customer can even state

Targets are drawn disjoint from the 200 public sessions, so this set is an
honest generalization check rather than a re-run of what you already tuned on.

Usage::

    python3 -m tools.build_synth_set                       # 800 rows
    python3 -m tools.build_synth_set --count 2000 --seed 7
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from evaluator.local_evaluator import coarse_category, load_jsonl
from tools.customer_sim import build_persona, extract_facets


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

SCENARIO_MIX = (("buying", 0.40), ("browsing", 0.40), ("intent_override", 0.15), ("boundary", 0.05))

DIFFICULTIES = ("easy", "medium", "hard")

# Anonymised aggregate profiles, same shape as the public set's user_profile.
PROFILE_TAGS = (
    ("fit", "comfort", "durability"),
    ("fit", "comfort", "style"),
    ("value", "quality", "durability"),
    ("style", "colour accuracy", "fit"),
    ("comfort", "warmth", "value"),
    ("quality", "craftsmanship", "style"),
)
PROFILE_FREQUENCY = ("1-2 prior purchases", "3-4 prior purchases", "5+ prior purchases")
PROFILE_RATING_STYLE = ("usually positive", "critical", "balanced")


def _iter_catalog(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _safe_int(value: object) -> int:
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return 0


def _title_tokens(title: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(title) if len(token) > 2]


def difficulty_features(
    product: dict,
    document_frequency: Counter,
    total_documents: int,
    category_size: Counter,
) -> dict[str, float]:
    """Higher score = easier to find."""
    title = str(product.get("title") or "")
    tokens = _title_tokens(title)
    idf_values = [
        math.log(total_documents / (1 + document_frequency[token]))
        for token in set(tokens)
    ]
    coarse = coarse_category([str(value) for value in product.get("categories") or []])
    facets = extract_facets(product, random.Random(0))

    return {
        "popularity": math.log1p(_safe_int(product.get("rating_number"))),
        "distinctiveness": max(idf_values) if idf_values else 0.0,
        "category_room": -math.log(1 + category_size[coarse]),
        "facet_richness": float(len(facets)),
    }


def _zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    deviation = math.sqrt(variance) or 1.0
    return [(value - mean) / deviation for value in values]


def allocate(total: int, weights: tuple[tuple[str, float], ...]) -> list[str]:
    """Largest-remainder allocation, so proportions are exact."""
    exact = {name: total * weight for name, weight in weights}
    counts = {name: int(value) for name, value in exact.items()}
    remaining = total - sum(counts.values())
    for name in sorted(exact, key=lambda name: -(exact[name] - counts[name])):
        if remaining <= 0:
            break
        counts[name] += 1
        remaining -= 1
    result: list[str] = []
    for name, _ in weights:
        result.extend([name] * counts[name])
    return result


def build(
    catalog_path: Path,
    public_path: Path,
    count: int,
    seed: int,
    pool_size: int,
) -> list[dict]:
    excluded = {
        str(json.loads(line)["ground_truth"]["parent_asin"])
        for line in public_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    document_frequency: Counter = Counter()
    category_size: Counter = Counter()
    products: dict[str, dict] = {}
    total_documents = 0

    for product in _iter_catalog(catalog_path):
        total_documents += 1
        title = str(product.get("title") or "")
        document_frequency.update(set(_title_tokens(title)))
        category_size[coarse_category([str(value) for value in product.get("categories") or []])] += 1
        parent_asin = str(product["parent_asin"])
        if parent_asin not in excluded:
            products[parent_asin] = product

    rng = random.Random(seed)
    candidates = sorted(products)
    rng.shuffle(candidates)
    candidates = candidates[: max(pool_size, count * 3)]

    features = [
        difficulty_features(products[parent_asin], document_frequency, total_documents, category_size)
        for parent_asin in candidates
    ]
    normalized = {
        key: _zscore([item[key] for item in features])
        for key in ("popularity", "distinctiveness", "category_room", "facet_richness")
    }
    scores = [
        1.0 * normalized["popularity"][index]
        + 1.0 * normalized["distinctiveness"][index]
        + 0.8 * normalized["category_room"][index]
        + 0.6 * normalized["facet_richness"][index]
        for index in range(len(candidates))
    ]

    ordered = [parent_asin for _, parent_asin in sorted(zip(scores, candidates), reverse=True)]
    third = len(ordered) // 3
    tiers = {
        "easy": ordered[:third],
        "medium": ordered[third : 2 * third],
        "hard": ordered[2 * third :],
    }

    per_difficulty = allocate(count, tuple((name, 1 / 3) for name in DIFFICULTIES))
    wanted = Counter(per_difficulty)

    rows: list[dict] = []
    index = 0
    for difficulty in DIFFICULTIES:
        take = wanted[difficulty]
        tier = tiers[difficulty]
        rng.shuffle(tier)
        scenarios = allocate(take, SCENARIO_MIX)
        rng.shuffle(scenarios)
        for offset in range(take):
            parent_asin = tier[offset]
            scenario = scenarios[offset]
            row_rng = random.Random(f"{seed}\0{parent_asin}")
            tags = row_rng.choice(PROFILE_TAGS)
            rating_style = row_rng.choice(PROFILE_RATING_STYLE)
            rows.append(
                {
                    "sample_id": f"synth_{index:04d}",
                    "scenario_type": scenario,
                    "difficulty_bucket": difficulty,
                    "category_bucket": "clothing",
                    "ground_truth": {"parent_asin": parent_asin},
                    "user_profile": {
                        "purchase_frequency": row_rng.choice(PROFILE_FREQUENCY),
                        "average_prior_rating": round(row_rng.uniform(1.0, 5.0), 1),
                        "rating_style": rating_style,
                        "preference_tags": list(tags),
                        "summary": f"Prior purchases emphasize {', '.join(tags)}; ratings are {rating_style}.",
                    },
                    "persona": build_persona(row_rng, difficulty),
                }
            )
            index += 1

    rows.sort(key=lambda row: row["sample_id"])
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a synthetic development set from the frozen catalog")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public", default="data/public_set.jsonl")
    parser.add_argument("--output", default="data/synth_set_800.jsonl")
    parser.add_argument("--count", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--pool-size", type=int, default=9000)
    args = parser.parse_args(argv)

    rows = build(Path(args.catalog), Path(args.public), args.count, args.seed, args.pool_size)
    Path(args.output).write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    by_difficulty = Counter(row["difficulty_bucket"] for row in rows)
    crossed: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        crossed[row["difficulty_bucket"]][row["scenario_type"]] += 1
    print(json.dumps({
        "output": args.output,
        "count": len(rows),
        "difficulty": dict(by_difficulty),
        "scenario_by_difficulty": {key: dict(value) for key, value in crossed.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
