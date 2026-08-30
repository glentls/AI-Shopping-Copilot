"""Offline retrieval-recall harness for R1. Deliberately does NOT import
`evaluator.local_evaluator` or `src.agent` -- it is a standalone, sub-10-second loop the
retrieval work can lean on after every change without spinning up the whole agent or the
scored evaluator.

What it measures
---------------
For each of the 150 dev sessions (`eval/dev_holdout_split.json` -> "dev"; the holdout list is
never touched here): reconstruct the turn-1 customer message, run it through the current
retrieval `search()`, and check where the gold `parent_asin` lands.

    py -m eval.recall_probe                 # headline: turn-1-message queries
    py -m eval.recall_probe --with-profile  # append user_profile.preference_tags to the query
    py -m eval.recall_probe --json          # machine-readable summary on stdout

Reported: Recall@{10,50,100,500}, median rank of gold when found, overall and per scenario,
plus the ORACLE diagnostic -- the same retrieval fed the gold item's own title. Oracle
Recall@100 that is not ~1.0 means the index itself is broken, not the query strategy; keep
that line green.

Query reconstruction (`_turn1_message`) mirrors `evaluator/local_evaluator.py`'s
`intent_card` / `coarse_category` / `behavior_for` / `initial_message` as of 2026-08-30. It is
vendored, not imported, on purpose (see module docstring). If the evaluator's generation logic
changes, re-sync this block -- the `--audit` flag prints a few reconstructed messages so a
drift check against a real eval run is a two-minute eyeball.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from src.config import load_config
from src.contracts import RetrievalRequest
from src.retrieval import build_index, search

REPO_ROOT = Path(__file__).resolve().parent.parent
RECALL_KS = (10, 50, 100, 500)
PROBE_TOP_K = max(RECALL_KS)

# --------------------------------------------------------------------------- #
# Vendored query reconstruction -- mirrors evaluator/local_evaluator.py (2026-08-30).
# --------------------------------------------------------------------------- #

_MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
_COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I
)
_SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")


def _searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field in _SEARCH_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _clean_constraint(value: str, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def _coarse_category(values: list[str]) -> str:
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def _intent_card(product: dict, limit: int = 180) -> dict:
    title = _clean_constraint(str(product.get("title") or "product"), limit)
    candidates = [*_flatten_values(product.get("features")), *_flatten_values(product.get("details"))]
    corpus = _searchable_text(product)
    material = _MATERIAL_RE.search(corpus)
    color = _COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    cleaned = list(
        dict.fromkeys(_clean_constraint(item, limit) for item in candidates if _clean_constraint(item, limit))
    )
    if not cleaned:
        cleaned = [title]
    return {
        "target_category": title,
        "hard_constraints": cleaned[:2],
        "soft_preferences": cleaned[2:4] or cleaned[:1],
    }


def _override_old_value(card: dict) -> str:
    soft = card["soft_preferences"]
    return soft[-1] if soft else "I prefer a different style."


def _turn1_message(sample: dict, product: dict) -> str:
    """The exact string retrieval receives on turn 1 (NullDialog passes it through verbatim)."""
    scenario = sample["scenario_type"]
    category = _coarse_category(product.get("categories") or [])
    card = _intent_card(product)
    if scenario == "buying" and card["hard_constraints"]:
        return f"I'm looking for {category}. A key requirement is: {card['hard_constraints'][0]}."
    if scenario == "intent_override":
        return f"I'm looking for {category}. {_override_old_value(card)}"
    return f"I'm looking for {category}, but I'm still exploring."


def _oracle_turn_stream(sample: dict, product: dict, max_turns: int = 4) -> list[str]:
    """The turn-1..N user messages under a PERFECT dialog -- one that asks exactly the right
    attribute each turn, so the customer discloses one held-back constraint per turn in the
    evaluator's own `customer_reply` phrasing. This is the ceiling: a real dialog extracts
    fewer, later. Boundary/override turn dynamics are out of scope here (buying+browsing are
    where accumulation matters); their turn-1 line is still measured."""
    card = _intent_card(product)
    turn1 = _turn1_message(sample, product)
    disclosed_in_turn1 = set()
    if sample["scenario_type"] == "buying" and card["hard_constraints"]:
        disclosed_in_turn1.add(card["hard_constraints"][0])
    held_back = [
        c for c in [*card["hard_constraints"], *card["soft_preferences"]]
        if c and c not in disclosed_in_turn1
    ]
    stream = [turn1]
    for constraint in held_back:
        if len(stream) >= max_turns:
            break
        stream.append(f"For that, what matters is: {constraint}.")
    return stream


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #


def _load_catalog(catalog_path: Path) -> dict[str, dict]:
    products: dict[str, dict] = {}
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            products[str(product["parent_asin"])] = product
    return products


def _dev_samples(dataset_path: Path) -> list[dict]:
    split = json.loads((REPO_ROOT / "eval" / "dev_holdout_split.json").read_text(encoding="utf-8"))
    dev_ids = set(split["dev"])
    with dataset_path.open(encoding="utf-8") as handle:
        return [
            sample
            for sample in (json.loads(line) for line in handle if line.strip())
            if sample["sample_id"] in dev_ids
        ]


def _rank_of_gold(index, query: str, gold: str, config: dict) -> int | None:
    request = RetrievalRequest(
        canonical_query=query, intent="unknown", hard_filters={}, soft_prefs={}, top_k=PROBE_TOP_K
    )
    return _position(search(index, request, config), gold)


def _position(results, gold: str) -> int | None:
    for position, candidate in enumerate(results, start=1):
        if candidate.parent_asin == gold:
            return position
    return None


def _union(*rank_lists: list[int | None]) -> list[int | None]:
    """Best (smallest) rank across methods per session -- the fusion recall ceiling."""
    out: list[int | None] = []
    for ranks in zip(*rank_lists):
        found = [r for r in ranks if r is not None]
        out.append(min(found) if found else None)
    return out


def _summarize(ranks: list[int | None]) -> dict:
    found = [r for r in ranks if r is not None]
    summary = {"n": len(ranks), "found": len(found)}
    for k in RECALL_KS:
        summary[f"recall@{k}"] = round(sum(1 for r in found if r <= k) / len(ranks), 4) if ranks else 0.0
    summary["median_rank_when_found"] = statistics.median(found) if found else None
    return summary


def _dense_ranks(catalog_path: Path, config: dict, queries: list[str], golds: list[str]) -> dict:
    """Build the dense index, batch-encode all queries, and return per-session gold ranks plus
    the pure vector-search timing (encoding excluded -- that is the transformer, not the search).
    Imports torch lazily via the encoder; only touched when --dense is passed."""
    import time as _t

    from src.retrieval import DenseIndex, dense_search_batch, load_products
    from src.retrieval.embed_index import build_or_load, make_encoder

    emb_cfg = config["retrieval"]["embedding"]
    t0 = _t.perf_counter()
    embedding = build_or_load(catalog_path, config)
    products = load_products(catalog_path)
    index = DenseIndex(embedding=embedding, products=products,
                       query_prefix=str(emb_cfg.get("query_prefix", "")))
    index._encoder = make_encoder(embedding.model_name, batch_size=int(emb_cfg.get("batch_size", 64)),
                                  max_seq_length=int(emb_cfg.get("max_seq_length", 160)))
    setup = _t.perf_counter() - t0

    t0 = _t.perf_counter()
    query_vectors = index.encode_queries(queries)
    encode_s = _t.perf_counter() - t0

    t0 = _t.perf_counter()
    results = dense_search_batch(index, query_vectors, k=PROBE_TOP_K)
    search_s = _t.perf_counter() - t0

    ranks = [_position(results[i], golds[i]) for i in range(len(golds))]
    return {
        "ranks": ranks,
        "timing": {
            "setup_s": round(setup, 2),
            "encode_s": round(encode_s, 2),
            "vector_search_s": round(search_s, 3),
            "vector_search_ms_per_query": round(1000 * search_s / max(1, len(queries)), 2),
        },
    }


def run(
    with_profile: bool,
    catalog_path: Path,
    dataset_path: Path,
    audit: bool,
    oracle_limit: int = 50,
    config: dict | None = None,
    dense: bool = False,
) -> dict:
    config = config if config is not None else load_config()
    started = time.perf_counter()
    products = _load_catalog(catalog_path)
    samples = _dev_samples(dataset_path)
    index = build_index(catalog_path, config)
    index_ready = time.perf_counter()

    golds = [str(s["ground_truth"]["parent_asin"]) for s in samples]
    scenarios = [s["scenario_type"] for s in samples]
    queries: list[str] = []
    for sample, gold in zip(samples, golds):
        query = _turn1_message(sample, products[gold])
        if with_profile:
            tags = sample.get("user_profile", {}).get("preference_tags") or []
            query = f"{query} {' '.join(str(t) for t in tags)}"
        queries.append(query)

    bm25_ranks = [_rank_of_gold(index, q, g, config) for q, g in zip(queries, golds)]

    # Oracle is a fixed-stride subsample by default: enough to catch a broken index without
    # doubling the query budget. `--oracle-full` runs all 150.
    oracle_every = max(1, len(samples) // oracle_limit) if oracle_limit else 1
    oracle = [
        _rank_of_gold(index, str(products[golds[i]].get("title") or ""), golds[i], config)
        for i in range(len(samples))
        if i % oracle_every == 0
    ]
    bm25_done = time.perf_counter()

    def by_scenario(ranks: list[int | None]) -> dict:
        grouped: dict[str, list[int | None]] = {}
        for scn, r in zip(scenarios, ranks):
            grouped.setdefault(scn, []).append(r)
        return {name: _summarize(v) for name, v in sorted(grouped.items())}

    methods = {"bm25": {"overall": _summarize(bm25_ranks), "by_scenario": by_scenario(bm25_ranks)}}
    timing = {
        "index_build": round(index_ready - started, 2),
        "bm25_queries": round(bm25_done - index_ready, 2),
    }

    if dense:
        d = _dense_ranks(catalog_path, config, queries, golds)
        dense_ranks = d["ranks"]
        union_ranks = _union(bm25_ranks, dense_ranks)
        methods["dense"] = {"overall": _summarize(dense_ranks), "by_scenario": by_scenario(dense_ranks)}
        methods["bm25_union_dense"] = {
            "overall": _summarize(union_ranks), "by_scenario": by_scenario(union_ranks)
        }
        methods["_dense_extra"] = {
            "recovered_by_dense_at_100": sum(
                1 for b, x in zip(bm25_ranks, dense_ranks)
                if (b is None or b > 100) and x is not None and x <= 100
            ),
            "lost_by_dense_at_100": sum(
                1 for b, x in zip(bm25_ranks, dense_ranks)
                if (x is None or x > 100) and b is not None and b <= 100
            ),
        }
        timing["dense"] = d["timing"]

    total = time.perf_counter()
    timing["total"] = round(total - started, 2)
    return {
        "mode": ("with_profile" if with_profile else "turn1_message") + ("+dense" if dense else ""),
        "methods": methods,
        "overall": methods["bm25"]["overall"],            # back-compat: bare "overall" == bm25
        "by_scenario": methods["bm25"]["by_scenario"],
        "oracle_title_query": _summarize(oracle),
        "timing_seconds": timing,
        "_audit": [f"  [{scn:15s}] {q!r}" for scn, q in list(zip(scenarios, queries))[:8]] if audit else [],
    }


def run_multiturn(catalog_path: Path, dataset_path: Path, config: dict | None = None) -> dict:
    """Measure the Phase-7 multi-turn query build against an oracle turn stream. For each turn
    depth t, three query strategies are scored:

      latest      -- only turn t's message (what a stateless retriever sees)
      accumulate  -- de-duplicated terms of turns 1..t (src.retrieval.multiturn.accumulate_query)
      accum+prof  -- accumulate, plus user_profile.preference_tags

    `search()` runs with multiturn.enabled forced OFF; the accumulated string is passed straight
    in as canonical_query, so this isolates the accumulation logic from the live SessionMemory
    path. Popularity / fusion stay at whatever the config says.
    """
    from src.retrieval.multiturn import accumulate_query, blend_profile

    config = config if config is not None else load_config()
    config = json.loads(json.dumps(config))  # local copy
    config["retrieval"].setdefault("multiturn", {})["enabled"] = False
    stopwords = frozenset(config["retrieval"].get("stopwords") or [])

    products = _load_catalog(catalog_path)
    samples = _dev_samples(dataset_path)
    index = build_index(catalog_path, config)
    golds = [str(s["ground_truth"]["parent_asin"]) for s in samples]
    scenarios = [s["scenario_type"] for s in samples]
    streams = [_oracle_turn_stream(s, products[g]) for s, g in zip(samples, golds)]
    profiles = [s.get("user_profile", {}) or {} for s in samples]
    max_t = max(len(s) for s in streams)

    def rank(query: str, gold: str) -> int | None:
        req = RetrievalRequest(canonical_query=query, intent="unknown", hard_filters={},
                               soft_prefs={}, top_k=PROBE_TOP_K)
        return _position(search(index, req, config), gold)

    strategies = ("latest", "accumulate", "accum+prof")
    per_turn: dict[int, dict[str, list[int | None]]] = {}
    for t in range(1, max_t + 1):
        rows: dict[str, list[int | None]] = {s: [] for s in strategies}
        for stream, gold, prof in zip(streams, golds, profiles):
            depth = min(t, len(stream))  # a short session holds at its final turn
            turns_so_far = [(i + 1, msg) for i, msg in enumerate(stream[:depth])]
            latest_q = stream[depth - 1]
            accum_q = accumulate_query(turns_so_far, config, stopwords) if depth > 1 else latest_q
            prof_q = blend_profile(accum_q, {"preference_tags": prof.get("preference_tags", [])},
                                   {"retrieval": {"multiturn": {"profile_blend": True}}})
            rows["latest"].append(rank(latest_q, gold))
            rows["accumulate"].append(rank(accum_q, gold))
            rows["accum+prof"].append(rank(prof_q, gold))
        per_turn[t] = rows

    def rec(ranks: list[int | None], k: int) -> float:
        return round(sum(1 for r in ranks if r is not None and r <= k) / len(ranks), 4)

    return {
        "mode": "multiturn (oracle dialog)",
        "max_turn": max_t,
        "by_turn": {
            t: {s: {"recall@10": rec(rows[s], 10), "recall@100": rec(rows[s], 100)}
                for s in strategies}
            for t, rows in per_turn.items()
        },
        "by_turn_scenario": {
            t: {
                scn: {
                    s: rec([r for r, sc in zip(rows[s], scenarios) if sc == scn], 10)
                    for s in strategies
                }
                for scn in sorted(set(scenarios))
            }
            for t, rows in per_turn.items()
        },
    }


def _print_multiturn(report: dict) -> None:
    print(f"\n=== recall_probe [{report['mode']}] ===")
    print(f"  {'turn':>4}  {'latest R@10/100':>18}  {'accumulate R@10/100':>20}  {'accum+prof R@10/100':>20}")
    for t, row in report["by_turn"].items():
        def cell(s: str) -> str:
            return f"{row[s]['recall@10']:.3f} / {row[s]['recall@100']:.3f}"
        print(f"  {t:>4}  {cell('latest'):>18}  {cell('accumulate'):>20}  {cell('accum+prof'):>20}")
    print("\n  by scenario, hit@10 (latest -> accumulate -> accum+prof):")
    for t, scn_rows in report["by_turn_scenario"].items():
        for scn, vals in scn_rows.items():
            print(f"    t{t} {scn:16s} {vals['latest']:.3f} -> {vals['accumulate']:.3f} -> {vals['accum+prof']:.3f}")


def _print_report(report: dict) -> None:
    def line(label: str, s: dict) -> None:
        cells = "  ".join(f"R@{k}={s[f'recall@{k}']:.3f}" for k in RECALL_KS)
        print(f"  {label:18s} n={s['n']:3d} found={s['found']:3d}  {cells}  med_rank={s['median_rank_when_found']}")

    print(f"\n=== recall_probe [{report['mode']}] ===")
    for method, data in report["methods"].items():
        if method.startswith("_"):
            continue
        print(f"  --- {method} ---")
        line("OVERALL", data["overall"])
        for name, s in data["by_scenario"].items():
            line(name, s)
    print("  --- oracle (gold title as query; R@100 must be ~1.0) ---")
    line("oracle", report["oracle_title_query"])

    extra = report["methods"].get("_dense_extra")
    if extra:
        print(f"  dense @100: recovered {extra['recovered_by_dense_at_100']} golds BM25 misses, "
              f"lost {extra['lost_by_dense_at_100']}")

    t = report["timing_seconds"]
    print(f"  timing: index {t['index_build']}s + bm25 {t['bm25_queries']}s"
          + (f" + dense {t['dense']}" if "dense" in t else "")
          + f" = {t['total']}s")
    if report["_audit"]:
        print("  --- reconstructed turn-1 messages (audit) ---")
        for entry in report["_audit"]:
            print(entry)

    print(f"\n  >>> BM25 Recall@100 (turn-1 queries) = {report['methods']['bm25']['overall']['recall@100']:.4f} <<<")
    if "bm25_union_dense" in report["methods"]:
        print(f"  >>> BM25 U dense Recall@100 (fusion ceiling) = "
              f"{report['methods']['bm25_union_dense']['overall']['recall@100']:.4f} <<<")
    oracle100 = report["oracle_title_query"]["recall@100"]
    if oracle100 < 0.95:
        print(f"  !! WARNING: oracle Recall@100 = {oracle100:.3f} < 0.95 -- the index is suspect.")


def main() -> None:
    parser = argparse.ArgumentParser(description="R1 offline recall probe")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--with-profile", action="store_true", help="append preference_tags to the query")
    parser.add_argument("--audit", action="store_true", help="print a sample of reconstructed messages")
    parser.add_argument("--oracle-full", action="store_true", help="run the oracle diagnostic on all 150 dev sessions")
    parser.add_argument("--dense", action="store_true",
                        help="also measure the dense route + BM25 U dense (loads the embedding cache + encoder; ~90s)")
    parser.add_argument("--multiturn", action="store_true",
                        help="Phase 7: score latest-turn vs accumulated query against an oracle turn stream")
    parser.add_argument("--json", action="store_true", help="emit the summary as JSON only")
    args = parser.parse_args()

    if args.multiturn:
        report = run_multiturn(Path(args.catalog), Path(args.dataset))
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_multiturn(report)
        return

    report = run(
        with_profile=args.with_profile,
        catalog_path=Path(args.catalog),
        dataset_path=Path(args.dataset),
        audit=args.audit,
        oracle_limit=0 if args.oracle_full else 50,
        dense=args.dense,
    )
    if args.json:
        report.pop("_audit", None)
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)


if __name__ == "__main__":
    main()
