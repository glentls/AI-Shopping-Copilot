"""Build and load the catalog's normalized attribute table.

The artifact stores both an inverted index (slot/value -> ASINs) and enough
information to construct a forward index at load time. ``distribution`` uses
that forward index, touching only the supplied candidates instead of scanning
all values in a slot.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from src.contracts import SLOTS
from src.lexicons import PATTERNS


ARTIFACT_NAME = "attributes.json"
ARTIFACT_VERSION = 2

SOURCE_CONFIDENCE = {
    "details": 0.99,
    "categories": 0.98,
    "title": 0.95,
    "features": 0.82,
    "description": 0.65,
    "store": 1.00,
    "price": 1.00,
}

BUDGET_BANDS = ("under 25", "25-50", "50-100", "100-200", "200+")
_PLACEHOLDER_BRANDS = {"", "generic", "unknown", "unbranded", "no brand", "n/a"}
_ROOT_CATEGORIES = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
_SIZE_DETAIL_KEYS = {
    "size", "size name", "size map", "shoe size", "band size", "cup size",
    "waist size", "inseam", "item size", "product size",
}
_SIZE_RANGE_RE = re.compile(
    r"\bsizes?\s*(?P<lo>\d{1,3}(?:\.5)?)\s*(?:-|to|through)\s*(?P<hi>\d{1,3}(?:\.5)?)\b",
    re.IGNORECASE,
)
_SIZE_NUMBER_RE = re.compile(
    r"(?:\b(?:shoe|dress|clothing|waist|size|sized)\s*(?:size\s*)?(?P<n1>\d{1,3}(?:\.5)?)\b"
    r"|\b(?P<n2>\d{1,2}(?:\.5)?)\s*(?:US\s*)?(?:shoe\s*)?size\b"
    r"|\bUS\s*(?P<n3>\d{1,2}(?:\.5)?)\b)",
    re.IGNORECASE,
)
_BRA_SIZE_RE = re.compile(r"\b(?P<band>2[6-9]|[3-5]\d)(?P<cup>aa|a|b|c|d{1,3}|e|f|g|h)\b", re.IGNORECASE)
_DIRECT_SIZE_RE = re.compile(
    r"^(?:US\s*)?(?P<size>\d{1,3}(?:\.5)?|2[6-9](?:aa|a|b|c|d{1,3}|e|f|g|h)|[3-5]\d(?:aa|a|b|c|d{1,3}|e|f|g|h))$",
    re.IGNORECASE,
)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _normalize_label(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).replace("’", "'").strip()).casefold()


def _category_text(value: object) -> str:
    categories = value if isinstance(value, list) else (value,)
    return " ".join(
        str(category)
        for category in categories
        if _normalize_label(category) not in _ROOT_CATEGORIES
    )


def _matched_values(slot: str, text: str) -> set[str]:
    if not text:
        return set()
    return {
        canonical
        for canonical, pattern in PATTERNS.get(slot, ())
        if pattern.search(text)
    }


def _dynamic_sizes(text: str, direct: bool = False) -> set[str]:
    values: set[str] = set()
    if direct:
        match = _DIRECT_SIZE_RE.fullmatch(text.strip())
        if match:
            values.add(match.group("size").casefold())
    values.update(
        f"{match.group('lo')}-{match.group('hi')}"
        for match in _SIZE_RANGE_RE.finditer(text)
    )
    values.update(
        (match.group("n1") or match.group("n2") or match.group("n3")).casefold()
        for match in _SIZE_NUMBER_RE.finditer(text)
    )
    values.update(
        (match.group("band") + match.group("cup")).casefold()
        for match in _BRA_SIZE_RE.finditer(text)
    )
    return values


def _price(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        price = float(value)
    elif isinstance(value, str):
        try:
            price = float(value.strip().replace("$", "").replace(",", ""))
        except ValueError:
            return None
    else:
        return None
    return price if 0.0 <= price < 1_000_000 else None


def _budget_band(price: float) -> str:
    if price < 25:
        return BUDGET_BANDS[0]
    if price < 50:
        return BUDGET_BANDS[1]
    if price < 100:
        return BUDGET_BANDS[2]
    if price < 200:
        return BUDGET_BANDS[3]
    return BUDGET_BANDS[4]


def _brand_values(product: dict) -> set[str]:
    candidates: list[object] = [product.get("store")]
    details = product.get("details")
    if isinstance(details, dict):
        candidates.extend(details.get(key) for key in ("Brand", "Brand Name"))
        if not any(candidates):
            candidates.append(details.get("Manufacturer"))

    result: set[str] = set()
    for candidate in candidates:
        if candidate in (None, ""):
            continue
        text = str(candidate)
        known = _matched_values("brand", text)
        normalized = _normalize_label(text)
        if known:
            result.update(known)
        elif normalized not in _PLACEHOLDER_BRANDS:
            result.add(normalized)
    return result


def _tag_product(product: dict) -> tuple[dict[str, dict[str, float]], float | None]:
    """Return slot values with their strongest source confidence."""
    tagged: dict[str, dict[str, float]] = defaultdict(dict)

    def record(slot: str, values: Iterable[str], source: str) -> None:
        confidence = SOURCE_CONFIDENCE[source]
        for value in values:
            previous = tagged[slot].get(value, 0.0)
            if confidence > previous:
                tagged[slot][value] = confidence

    title = _text(product.get("title"))
    features = _text(product.get("features"))
    description = _text(product.get("description"))
    categories = _category_text(product.get("categories"))
    details = product.get("details") if isinstance(product.get("details"), dict) else {}
    details_text = _text(details)

    # Taxonomy and title are dependable category sources. Product prose often
    # mentions related products, so it is intentionally excluded here.
    record("category", _matched_values("category", categories), "categories")
    record("category", _matched_values("category", title), "title")

    for value in _brand_values(product):
        record("brand", (value,), "store")

    for slot in ("material", "color", "style", "feature", "use_case"):
        record(slot, _matched_values(slot, details_text), "details")
        record(slot, _matched_values(slot, title), "title")
        record(slot, _matched_values(slot, features), "features")
        record(slot, _matched_values(slot, description), "description")

    # Unqualified "small" and "large" are noisy in product prose. Restrict
    # static sizes to titles and dedicated structured fields, while still
    # mining contextual numeric sizes from titles/features/descriptions.
    record("size", _matched_values("size", title), "title")
    record("size", _dynamic_sizes(title), "title")
    record("size", _dynamic_sizes(features), "features")
    record("size", _dynamic_sizes(description), "description")
    for key, value in details.items():
        if _normalize_label(key) in _SIZE_DETAIL_KEYS:
            size_text = _text(value)
            record("size", _matched_values("size", size_text), "details")
            record("size", _dynamic_sizes(size_text, direct=True), "details")

    price = _price(product.get("price"))
    if price is not None:
        record("budget", (_budget_band(price),), "price")

    return dict(tagged), price


class AttributeTable:
    """Forward and inverted normalized catalog attributes."""

    def __init__(
        self,
        inverted: dict[str, dict[str, set[str]]],
        total: int,
        confidence: dict[str, dict[str, dict[str, float]]] | None = None,
        prices: dict[str, float] | None = None,
    ) -> None:
        self._inverted = inverted
        self._total = max(0, int(total))
        self._confidence = confidence or {}
        self._prices = prices or {}

        forward: dict[str, dict[str, list[str]]] = {}
        for slot, values in inverted.items():
            slot_forward: dict[str, list[str]] = defaultdict(list)
            for value, asins in values.items():
                for asin in asins:
                    slot_forward[asin].append(value)
            forward[slot] = {
                asin: sorted(product_values)
                for asin, product_values in slot_forward.items()
            }
        self._forward = forward
        self._coverage = {
            slot: (len(products) / self._total if self._total else 0.0)
            for slot, products in forward.items()
        }

    def values(self, asin: str, slot: str) -> list[str]:
        return list(self._forward.get(slot, {}).get(asin, ()))

    def matching(self, slot: str, value: str) -> set[str]:
        normalized = _normalize_label(value).replace("-", " ")
        values = self._inverted.get(slot, {})
        if normalized in values:
            return values[normalized]
        # Canonical values such as "v neck" already use spaces, while brand
        # names and range values legitimately retain punctuation.
        return values.get(_normalize_label(value), set())

    def coverage(self, slot: str) -> float:
        return self._coverage.get(slot, 0.0)

    def distribution(self, slot: str, candidates: set[str]) -> dict[str, int]:
        """Count values over candidates only, using the precomputed forward map."""
        if not candidates:
            return {}
        forward = self._forward.get(slot)
        if not forward:
            return {}
        counts: dict[str, int] = defaultdict(int)
        for asin in candidates:
            for value in forward.get(asin, ()):
                counts[value] += 1
        return dict(counts)

    def confidence(self, asin: str, slot: str, value: str) -> float:
        """Strongest catalog-source confidence for one normalized value."""
        return self._confidence.get(slot, {}).get(value, {}).get(asin, 0.0)

    def price(self, asin: str) -> float | None:
        """Raw numeric price for soft budget ranking; never a filter."""
        return self._prices.get(asin)

    def slots(self) -> list[str]:
        return [slot for slot in SLOTS if slot in self._inverted]


def _payload(
    inverted: dict[str, dict[str, set[str]]],
    confidence: dict[str, dict[str, dict[str, float]]],
    prices: dict[str, float],
    total: int,
) -> dict:
    serialized_inverted: dict[str, dict[str, list[str]]] = {}
    serialized_confidence: dict[str, dict[str, list[int]]] = {}
    for slot, values in inverted.items():
        serialized_inverted[slot] = {}
        serialized_confidence[slot] = {}
        for value, asins in values.items():
            ordered = sorted(asins)
            serialized_inverted[slot][value] = ordered
            by_asin = confidence.get(slot, {}).get(value, {})
            serialized_confidence[slot][value] = [
                round(100 * by_asin.get(asin, 0.0)) for asin in ordered
            ]
    return {
        "version": ARTIFACT_VERSION,
        "total": total,
        "inverted": serialized_inverted,
        "confidence": serialized_confidence,
        "prices": prices,
    }


def build_attribute_table(
    catalog_path: str | Path,
    artifacts_dir: str | Path,
) -> AttributeTable:
    """Build the artifact in one catalog pass and return the loaded table.

    Returning the table preserves the repository's build-script integration;
    callers relying on the specified side-effect-only API may ignore it.
    """
    inverted: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    confidence: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    prices: dict[str, float] = {}
    total = 0

    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            asin = str(product["parent_asin"])
            total += 1
            tagged, price = _tag_product(product)
            if price is not None:
                prices[asin] = price
            for slot, values in tagged.items():
                for value, score in values.items():
                    inverted[slot][value].add(asin)
                    confidence[slot][value][asin] = score

    plain_inverted = {
        slot: {value: set(asins) for value, asins in values.items()}
        for slot, values in inverted.items()
    }
    plain_confidence = {
        slot: {value: dict(scores) for value, scores in values.items()}
        for slot, values in confidence.items()
    }

    directory = Path(artifacts_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = _payload(plain_inverted, plain_confidence, prices, total)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, prefix=".attributes-", delete=False
        ) as handle:
            temporary_path = handle.name
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary_path, directory / ARTIFACT_NAME)
        temporary_path = None
    finally:
        if temporary_path:
            try:
                Path(temporary_path).unlink()
            except FileNotFoundError:
                pass

    return AttributeTable(plain_inverted, total, plain_confidence, prices)


def _table_from_payload(payload: dict) -> AttributeTable:
    if payload.get("version") != ARTIFACT_VERSION:
        raise ValueError("attribute artifact version is stale")
    raw_inverted = payload["inverted"]
    raw_confidence = payload.get("confidence", {})
    inverted: dict[str, dict[str, set[str]]] = {}
    confidence: dict[str, dict[str, dict[str, float]]] = {}
    for slot, values in raw_inverted.items():
        inverted[slot] = {}
        confidence[slot] = {}
        for value, ordered_asins in values.items():
            inverted[slot][value] = set(ordered_asins)
            scores = raw_confidence.get(slot, {}).get(value, ())
            confidence[slot][value] = {
                asin: score / 100.0
                for asin, score in zip(ordered_asins, scores)
            }
    prices = {asin: float(price) for asin, price in payload.get("prices", {}).items()}
    return AttributeTable(inverted, int(payload["total"]), confidence, prices)


def load_attribute_table(
    artifacts_dir: str | Path,
    catalog_path: str | Path | None = None,
) -> AttributeTable:
    """Load a built table, optionally rebuilding a missing or stale cache.

    The one-argument form is the public API. The optional catalog path keeps the
    frozen starter's cache-on-miss call backward compatible.
    """
    cache = Path(artifacts_dir) / ARTIFACT_NAME
    try:
        with cache.open(encoding="utf-8") as handle:
            return _table_from_payload(json.load(handle))
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as error:
        if catalog_path is None:
            raise RuntimeError(
                f"No usable attribute table at {cache}; run python3 -m tools.build_index"
            ) from error
    return build_attribute_table(catalog_path, artifacts_dir)
