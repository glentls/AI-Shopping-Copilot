"""Catalog -> attribute table and inverted index.

LANE A OWNS THIS FILE. Thin but working; raising per-slot coverage without
introducing false positives is the bulk of Lane A's work.

The method other lanes depend on most is distribution(): Lane C's question
picker calls it every turn to work out which question splits the remaining
candidates most evenly. Keep it fast -- under 10ms for 5,000 candidates.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.lexicons import PATTERNS

ARTIFACT_NAME = "attributes.json"
_EXCLUDED_CATEGORIES = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}


def _product_text(product: dict) -> str:
    parts = [str(product.get("title") or "")]
    parts += [str(x) for x in (product.get("features") or [])]
    parts += [str(x) for x in (product.get("description") or [])]
    details = product.get("details")
    if isinstance(details, dict):
        parts += [f"{k} {v}" for k, v in details.items()]
    return " ".join(parts)


def coarse_category(values: list[str]) -> str:
    cleaned: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in _EXCLUDED_CATEGORIES:
                cleaned.append(part)
    return " ".join(cleaned[-2:]).lower() if cleaned else "clothing item"


class AttributeTable:
    """slot -> value -> set of parent_asin, plus the forward lookup."""

    def __init__(self, inverted: dict[str, dict[str, set[str]]], total: int) -> None:
        self._inverted = inverted
        self._total = max(1, total)
        self._forward: dict[str, dict[str, list[str]]] = {}
        for slot, values in inverted.items():
            for value, asins in values.items():
                for asin in asins:
                    self._forward.setdefault(asin, {}).setdefault(slot, []).append(value)
        self._coverage = {
            slot: len(set().union(*values.values())) / self._total if values else 0.0
            for slot, values in inverted.items()
        }

    def values(self, asin: str, slot: str) -> list[str]:
        return self._forward.get(asin, {}).get(slot, [])

    def matching(self, slot: str, value: str) -> set[str]:
        return self._inverted.get(slot, {}).get(value, set())

    def coverage(self, slot: str) -> float:
        return self._coverage.get(slot, 0.0)

    def distribution(self, slot: str, candidates: set[str]) -> dict[str, int]:
        """{value: how many of `candidates` carry it}. Lane C's entropy input."""
        values = self._inverted.get(slot)
        if not values or not candidates:
            return {}
        counts = {}
        for value, asins in values.items():
            overlap = len(candidates & asins) if len(candidates) < len(asins) else len(asins & candidates)
            if overlap:
                counts[value] = overlap
        return counts

    def slots(self) -> list[str]:
        return list(self._inverted)


def build_attribute_table(catalog_path: str | Path, artifacts_dir: str | Path) -> AttributeTable:
    """One pass over the catalog. Writes a cache; returns the table."""
    inverted: dict[str, dict[str, set[str]]] = {}
    total = 0
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            asin = str(product["parent_asin"])
            total += 1
            text = _product_text(product)
            for slot, entries in PATTERNS.items():
                for canonical, pattern in entries:
                    if pattern.search(text):
                        inverted.setdefault(slot, {}).setdefault(canonical, set()).add(asin)
            store = product.get("store")
            if store:
                inverted.setdefault("brand", {}).setdefault(str(store).strip().lower(), set()).add(asin)
            inverted.setdefault("category", {}).setdefault(
                coarse_category([str(v) for v in (product.get("categories") or [])]), set()
            ).add(asin)

    directory = Path(artifacts_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "total": total,
        "inverted": {slot: {v: sorted(a) for v, a in values.items()} for slot, values in inverted.items()},
    }
    (directory / ARTIFACT_NAME).write_text(json.dumps(payload), encoding="utf-8")
    return AttributeTable(inverted, total)


def load_attribute_table(artifacts_dir: str | Path, catalog_path: str | Path) -> AttributeTable:
    """Load the cache, building it first if it is missing or unreadable."""
    cache = Path(artifacts_dir) / ARTIFACT_NAME
    if cache.exists():
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            inverted = {
                slot: {value: set(asins) for value, asins in values.items()}
                for slot, values in payload["inverted"].items()
            }
            return AttributeTable(inverted, int(payload["total"]))
        except (ValueError, KeyError, OSError):
            pass
    return build_attribute_table(catalog_path, artifacts_dir)
