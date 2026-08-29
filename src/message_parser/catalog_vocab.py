"""Builds category/brand vocab from the real catalog for higher-precision
attribute matching than static word lists alone."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .vocab import EXCLUDED_CATEGORY_TERMS

# Query-side tokenization (TOKEN_RE) splits on any non-alphanumeric
# character and rejoins tokens with a plain space, so catalog terms must be
# normalized the same way or a literal hyphen ("t-shirts") never matches
# space-joined query tokens ("t shirts"). 46 catalog category terms contain
# a hyphen, including "t-shirts" (a top-10 category by product count).
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalize(text: str) -> str:
    return _NON_ALNUM_RE.sub(" ", text.lower()).strip()


def load_catalog_vocab(catalog_path: str | Path) -> tuple[set[str], set[str]]:
    """Returns (categories, brands), both lowercase and space-normalized (see
    `_normalize`). Categories are individual leaf terms from the catalog's
    `categories` lists; brands are `store` values."""
    categories: set[str] = set()
    brands: set[str] = set()
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            for cat in product.get("categories") or []:
                for part in str(cat).split(","):
                    cleaned = _normalize(part)
                    if cleaned and cleaned not in EXCLUDED_CATEGORY_TERMS:
                        categories.add(cleaned)
            store = product.get("store")
            if store:
                normalized_store = _normalize(str(store))
                if normalized_store:
                    brands.add(normalized_store)
    return categories, brands
