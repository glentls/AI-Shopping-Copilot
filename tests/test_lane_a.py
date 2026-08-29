"""Regression and performance tests for Lane A understanding."""

from __future__ import annotations

import json
import statistics
import tempfile
import time
import unittest
from pathlib import Path

from src.attributes import AttributeTable, build_attribute_table, load_attribute_table
from src.contracts import ConversationState
from src.extract import (
    detect_no_preference,
    detect_override,
    extract_slots,
    parse_budget,
)


def _values(found: dict, slot: str, polarity: bool | None = None) -> set[str]:
    return {
        value.value
        for value in found.get(slot, ())
        if polarity is None or value.polarity is polarity
    }


class CustomerExtractionTest(unittest.TestCase):
    def extract(self, text: str) -> tuple[dict, ConversationState]:
        state = ConversationState("session", {})
        return extract_slots(text, 3, state), state

    def test_word_boundaries_do_not_find_red_inside_embroidered(self) -> None:
        found, _ = self.extract("an embroidered, zippered jacket")
        self.assertNotIn("red", _values(found, "color"))
        self.assertIn("jacket", _values(found, "category"))

        found, _ = self.extract("a red jacket")
        self.assertIn("red", _values(found, "color"))

    def test_negation_is_scoped_to_the_relevant_clause(self) -> None:
        found, _ = self.extract(
            "Not leather, no heels, and anything but black or red; blue is fine."
        )
        self.assertEqual(_values(found, "material", False), {"leather"})
        self.assertEqual(_values(found, "category", False), {"heels"})
        self.assertEqual(_values(found, "color", False), {"black", "red"})
        self.assertEqual(_values(found, "color", True), {"blue"})

    def test_synonyms_share_canonical_values(self) -> None:
        for phrase in ("water resistant", "water-resistant", "waterproof"):
            found, _ = self.extract(phrase)
            self.assertEqual(_values(found, "feature"), {"waterproof"})

        for phrase in ("womens shoes", "women's shoes", "for her shoes"):
            found, _ = self.extract(phrase)
            self.assertIn("women", _values(found, "category"))

        found, _ = self.extract("an elastane top in navy")
        self.assertIn("spandex", _values(found, "material"))
        self.assertIn("blue", _values(found, "color"))

    def test_budget_parsing_updates_state_with_range_maximum(self) -> None:
        cases = {
            "under $80": 80.0,
            "below 80 dollars": 80.0,
            "$50-100": 100.0,
            "between $40 and $75": 75.0,
            "budget 50-100": 100.0,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                found, state = self.extract(text)
                self.assertEqual(parse_budget(text), expected)
                self.assertEqual(state.budget_max, expected)
                self.assertEqual(_values(found, "budget"), {str(int(expected))})
        self.assertIsNone(parse_budget("size 8-10"))

    def test_override_phrasings_identify_affected_slots(self) -> None:
        cases = {
            "Actually, make it blue instead.": ["color"],
            "On second thought, I want cotton.": ["material"],
            "Ignore what I said about size.": ["size"],
            "I changed my mind about the budget; under $60.": ["budget"],
            "Scratch that.": ["*"],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(detect_override(text), expected)
        self.assertEqual(detect_override("I would like blue."), [])

    def test_no_preference_returns_named_or_last_asked_marker(self) -> None:
        self.assertEqual(detect_no_preference("No preference for color."), ["color"])
        self.assertEqual(
            detect_no_preference("Material or brand doesn't matter."),
            ["material", "brand"],
        )
        self.assertEqual(detect_no_preference("You decide."), ["*"])


class AttributeTableTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.catalog = root / "catalog.jsonl"
        self.artifacts = root / "artifacts"
        products = [
            {
                "parent_asin": "A",
                "title": "Red waterproof Women's hiking boot size 8",
                "features": ["Comfortable genuine leather upper"],
                "description": ["Built for wet trails"],
                "details": {},
                "categories": ["Clothing, Shoes & Jewelry", "Women", "Shoes", "Boots"],
                "store": "Nike",
                "price": 79.99,
            },
            {
                "parent_asin": "B",
                "title": "Embroidered cotton dress",
                "features": [],
                "description": [],
                "details": {},
                "categories": ["Clothing, Shoes & Jewelry", "Women", "Dresses"],
                "store": "Other Brand",
                "price": 30,
            },
            {
                "parent_asin": "C",
                "title": "Plain men's shirt",
                "features": [],
                "description": ["A breathable shirt in red."],
                "details": {},
                "categories": ["Clothing, Shoes & Jewelry", "Men", "Shirts"],
                "store": None,
                "price": None,
            },
            {
                "parent_asin": "D",
                "title": "Blue summer sandals",
                "features": ["Water-resistant and lightweight"],
                "description": [],
                "details": {"Size": "10"},
                "categories": ["Clothing, Shoes & Jewelry", "Women", "Shoes", "Sandals"],
                "store": "Skechers",
                "price": 22.5,
            },
        ]
        self.catalog.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        self.table = build_attribute_table(self.catalog, self.artifacts)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_catalog_word_boundaries_and_synonyms(self) -> None:
        self.assertEqual(self.table.matching("color", "red"), {"A", "C"})
        self.assertNotIn("B", self.table.matching("color", "red"))
        self.assertEqual(self.table.matching("feature", "waterproof"), {"A", "D"})

    def test_root_taxonomy_is_not_a_category_value(self) -> None:
        self.assertEqual(self.table.matching("category", "shoes"), {"A", "D"})
        self.assertEqual(self.table.matching("category", "jewelry"), set())

    def test_source_confidence_is_lower_for_description_prose(self) -> None:
        self.assertEqual(self.table.confidence("A", "color", "red"), 0.95)
        self.assertEqual(self.table.confidence("C", "color", "red"), 0.65)

    def test_budget_keeps_price_for_soft_ranking(self) -> None:
        self.assertEqual(self.table.values("A", "budget"), ["50-100"])
        self.assertEqual(self.table.price("A"), 79.99)
        self.assertIsNone(self.table.price("C"))
        self.assertEqual(self.table.coverage("budget"), 0.75)

    def test_distribution_counts_only_requested_candidates(self) -> None:
        self.assertEqual(
            self.table.distribution("color", {"A", "B", "D", "missing"}),
            {"red": 1, "blue": 1},
        )
        self.assertEqual(
            self.table.distribution("category", {"A", "B"}),
            {"women": 2, "boots": 1, "shoes": 1, "dress": 1},
        )
        self.assertEqual(self.table.distribution("color", set()), {})

    def test_one_argument_load_restores_values_and_confidence(self) -> None:
        loaded = load_attribute_table(self.artifacts)
        self.assertEqual(loaded.values("D", "size"), ["10"])
        self.assertEqual(loaded.matching("brand", "skechers"), {"D"})
        self.assertEqual(loaded.confidence("C", "color", "red"), 0.65)

    def test_distribution_meets_5000_candidate_latency_target(self) -> None:
        inverted: dict[str, dict[str, set[str]]] = {"brand": {}}
        for number in range(10_000):
            value = f"brand-{number % 1000}"
            inverted["brand"].setdefault(value, set()).add(f"A{number:05d}")
        table = AttributeTable(inverted, 10_000)
        candidates = {f"A{number:05d}" for number in range(0, 10_000, 2)}
        timings = []
        for _ in range(7):
            started = time.perf_counter()
            result = table.distribution("brand", candidates)
            timings.append(time.perf_counter() - started)
        self.assertEqual(sum(result.values()), 5_000)
        self.assertLess(statistics.median(timings), 0.010)


if __name__ == "__main__":
    unittest.main()
