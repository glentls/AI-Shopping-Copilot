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
from src.lexicons import LEXICON


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

    def test_closure_features_do_not_duplicate_style_values(self) -> None:
        closure_cases = {
            "Closure type: Pull On": "pull on closure",
            "I want a zipper closure": "zipper closure",
            "It needs zippered pockets": "zipper closure",
            "Something with buttons": "button closure",
            "It needs a drawstring waist": "drawstring closure",
            "I prefer a buckle fastening": "buckle closure",
            "Please find one with snap buttons": "snap closure",
        }
        for text, expected in closure_cases.items():
            with self.subTest(text=text):
                found, _ = self.extract(text)
                self.assertIn(expected, _values(found, "feature"))
                self.assertNotIn("style", found)

        style_cases = {
            "a zip-up jacket": "zip up",
            "a button-down shirt": "button down",
            "a pullover sweater": "pullover",
            "slip-on shoes": "slip on",
        }
        closure_values = {
            "pull on closure", "zipper closure", "button closure",
            "drawstring closure", "buckle closure", "snap closure",
        }
        for text, expected in style_cases.items():
            with self.subTest(text=text):
                found, _ = self.extract(text)
                self.assertIn(expected, _values(found, "style"))
                self.assertTrue(_values(found, "feature").isdisjoint(closure_values))

    def test_closure_and_style_surface_forms_are_disjoint(self) -> None:
        closure_values = {
            "pull on closure", "zipper closure", "button closure",
            "drawstring closure", "buckle closure", "snap closure",
        }

        def surfaces(slot: str, values: set[str]) -> set[str]:
            return {
                " ".join(surface.casefold().replace("-", " ").split())
                for canonical, aliases in LEXICON[slot].items()
                if canonical in values
                for surface in [canonical, *aliases]
            }

        closure_surfaces = surfaces("feature", closure_values)
        style_surfaces = surfaces("style", set(LEXICON["style"]))
        self.assertTrue(closure_surfaces.isdisjoint(style_surfaces))

    def test_closure_negation_has_negative_polarity(self) -> None:
        found, _ = self.extract("No zipper or snap closure, please.")
        self.assertEqual(
            _values(found, "feature", False),
            {"zipper closure", "snap closure"},
        )


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


class CatalogPrecisionTest(unittest.TestCase):
    def test_prose_context_and_negation_filter_catalog_values(self) -> None:
        products = [
            {
                "parent_asin": "NEGATED",
                "title": "Rain Shell",
                "features": ["Not waterproof", "A breathable shell"],
                "description": ["Available in blue, not black. Synthetic, not leather."],
                "details": {},
                "categories": ["Clothing, Shoes & Jewelry", "Jackets"],
                "store": "Example",
                "price": None,
            },
            {
                "parent_asin": "RESISTANT",
                "title": "Rain Jacket",
                "features": ["Water-resistant but not fully waterproof"],
                "description": [],
                "details": {},
                "categories": ["Clothing, Shoes & Jewelry", "Jackets"],
                "store": "Example",
                "price": None,
            },
            {
                "parent_asin": "WIRELESS",
                "title": "Wireless Bra",
                "features": ["No underwire construction"],
                "description": [],
                "details": {},
                "categories": ["Clothing, Shoes & Jewelry", "Bras"],
                "store": "Example",
                "price": None,
            },
            {
                "parent_asin": "EXCEPTIONS",
                "title": "Everyday Shirt",
                "features": [
                    "Not only comfortable but also soft",
                    "Breathable without sacrificing comfort",
                ],
                "description": [],
                "details": {},
                "categories": ["Clothing, Shoes & Jewelry", "Shirts"],
                "store": "Example",
                "price": None,
            },
            {
                "parent_asin": "BROAD_FALSE",
                "title": "Metal Jewelry Clasp",
                "features": [],
                "description": [
                    "Clean with a soft cloth. The clasp will work with classic accessories."
                ],
                "details": {},
                "categories": ["Clothing, Shoes & Jewelry", "Jewelry"],
                "store": "Example",
                "price": None,
            },
            {
                "parent_asin": "BROAD_TRUE",
                "title": "Classic Soft Work Boots",
                "features": [],
                "description": [],
                "details": {},
                "categories": ["Clothing, Shoes & Jewelry", "Boots"],
                "store": "Example",
                "price": None,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "catalog.jsonl"
            catalog.write_text(
                "".join(json.dumps(product) + "\n" for product in products),
                encoding="utf-8",
            )
            table = build_attribute_table(catalog, root / "artifacts")

        self.assertNotIn("NEGATED", table.matching("feature", "waterproof"))
        self.assertIn("NEGATED", table.matching("feature", "breathable"))
        self.assertIn("RESISTANT", table.matching("feature", "waterproof"))
        self.assertIn("WIRELESS", table.matching("feature", "wireless"))
        self.assertNotIn("WIRELESS", table.matching("feature", "underwire"))
        self.assertEqual(table.matching("color", "blue"), {"NEGATED"})
        self.assertNotIn("NEGATED", table.matching("color", "black"))
        self.assertNotIn("NEGATED", table.matching("material", "leather"))

        for value in ("comfortable", "soft", "breathable"):
            self.assertIn("EXCEPTIONS", table.matching("feature", value))
        self.assertNotIn("BROAD_FALSE", table.matching("feature", "soft"))
        self.assertNotIn("BROAD_FALSE", table.matching("use_case", "work"))
        self.assertNotIn("BROAD_FALSE", table.matching("style", "classic"))
        self.assertIn("BROAD_TRUE", table.matching("feature", "soft"))
        self.assertIn("BROAD_TRUE", table.matching("use_case", "work"))
        self.assertIn("BROAD_TRUE", table.matching("style", "classic"))

    def test_catalog_closures_do_not_duplicate_style_or_prose_noise(self) -> None:
        products = [
            {
                "parent_asin": "PULL_ON",
                "title": "Pull-On Walking Shoe",
                "features": [],
                "description": [],
                "details": {},
                "categories": ["Clothing, Shoes & Jewelry", "Shoes"],
                "store": "Example",
                "price": None,
            },
            {
                "parent_asin": "STYLE_ONLY",
                "title": "Zip-Up Button-Down Jacket",
                "features": [],
                "description": [],
                "details": {},
                "categories": ["Clothing, Shoes & Jewelry", "Jackets"],
                "store": "Example",
                "price": None,
            },
            {
                "parent_asin": "BUTTON",
                "title": "Casual Shirt",
                "features": ["Closure type: Button"],
                "description": [],
                "details": {},
                "categories": ["Clothing, Shoes & Jewelry", "Shirts"],
                "store": "Example",
                "price": None,
            },
            {
                "parent_asin": "SNAP_DETAILS",
                "title": "Baby Bodysuit",
                "features": [],
                "description": [],
                "details": {"Closure Type": "Snap"},
                "categories": ["Clothing, Shoes & Jewelry", "Baby"],
                "store": "Example",
                "price": None,
            },
            {
                "parent_asin": "MULTI",
                "title": "Utility Pants",
                "features": ["Drawstring waist with an adjustable buckle"],
                "description": [],
                "details": {},
                "categories": ["Clothing, Shoes & Jewelry", "Pants"],
                "store": "Example",
                "price": None,
            },
            {
                "parent_asin": "PROSE_NOISE",
                "title": "Camera Pendant",
                "features": [],
                "description": ["Take a snap photo, then pull on the strap to adjust it."],
                "details": {},
                "categories": ["Clothing, Shoes & Jewelry", "Necklaces"],
                "store": "Example",
                "price": None,
            },
            {
                "parent_asin": "NEGATED",
                "title": "Minimalist Backpack",
                "features": ["No zipper or snap closure"],
                "description": [],
                "details": {},
                "categories": ["Clothing, Shoes & Jewelry", "Backpacks"],
                "store": "Example",
                "price": None,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "catalog.jsonl"
            catalog.write_text(
                "".join(json.dumps(product) + "\n" for product in products),
                encoding="utf-8",
            )
            table = build_attribute_table(catalog, root / "artifacts")

        self.assertEqual(table.matching("feature", "pull on closure"), {"PULL_ON"})
        self.assertEqual(table.matching("feature", "button closure"), {"BUTTON"})
        self.assertEqual(table.matching("feature", "snap closure"), {"SNAP_DETAILS"})
        self.assertEqual(table.matching("feature", "drawstring closure"), {"MULTI"})
        self.assertEqual(table.matching("feature", "buckle closure"), {"MULTI"})
        self.assertEqual(table.matching("style", "zip up"), {"STYLE_ONLY"})
        self.assertEqual(table.matching("style", "button down"), {"STYLE_ONLY"})
        for asin in ("STYLE_ONLY", "PROSE_NOISE", "NEGATED"):
            self.assertFalse({
                value for value in table.values(asin, "feature") if value.endswith("closure")
            })


if __name__ == "__main__":
    unittest.main()
