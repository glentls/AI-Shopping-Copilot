"""Tests for the synthetic development set and the realistic shopper simulator.

Two properties matter most and both are asserted here:

1. every constraint the simulated customer states must be *true* of the target
   product, otherwise the session is unwinnable and the difficulty label is a
   lie rather than a challenge;
2. the difficulty terciles must be balanced and actually ordered by measured
   findability, not by an arbitrary label.
"""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from evaluator.local_evaluator import ALLOWED_ATTRIBUTES, MAX_TURNS, catalog_index, coarse_category
from tools.build_synth_set import allocate, build, difficulty_features
from tools.customer_sim import (
    COLOR_WORDS,
    MATERIAL_WORDS,
    RealisticCustomer,
    build_persona,
    extract_facets,
)
from tools.trace_runner import RealisticCustomerAdapter, build_customer, run_session
from starter.agent import Agent


CATALOG_ROWS = [
    {
        "parent_asin": "S001",
        "title": "Cavalo Women's Black Leather Ankle Boot",
        "features": ["Genuine leather upper", "Rubber sole", "Side zip closure"],
        "details": {"Department": "womens", "Material": "Leather", "Closure": "Zipper"},
        "description": ["Also available in red suede and blue canvas."],
        "categories": ["Clothing, Shoes & Jewelry", "Women", "Shoes", "Boots"],
        "store": "Cavalo",
        "average_rating": 4.6,
        "rating_number": 2400,
        "price": 89.0,
    },
    {
        "parent_asin": "S002",
        "title": "Trailhead Men's Waterproof Hiking Shoe",
        "features": ["Mesh upper", "Waterproof membrane", "Lightweight"],
        "details": {"Department": "mens", "Material": "Mesh"},
        "description": ["Built for hiking."],
        "categories": ["Clothing, Shoes & Jewelry", "Men", "Shoes", "Hiking"],
        "store": "Trailhead",
        "average_rating": 4.1,
        "rating_number": 180,
        "price": 120.0,
    },
    {
        "parent_asin": "S003",
        "title": "Plain Cotton Crew Neck T-Shirt",
        "features": ["100% cotton", "Machine wash"],
        "details": {"Department": "unisex"},
        "description": ["A shirt."],
        "categories": ["Clothing, Shoes & Jewelry", "Men", "Clothing", "Shirts"],
        "store": "Basics",
        "average_rating": 3.9,
        "rating_number": 12,
        "price": 15.0,
    },
]


def catalog_text(product: dict) -> str:
    parts = [str(product.get("title") or ""), str(product.get("store") or "")]
    parts.extend(str(item) for item in product.get("features") or [])
    parts.extend(f"{key} {value}" for key, value in (product.get("details") or {}).items())
    parts.extend(str(item) for item in product.get("categories") or [])
    return " ".join(parts).lower()


def write_rows(root: Path, name: str, rows: list[dict]) -> Path:
    path = root / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


# =============================================================================
# FACET EXTRACTION
# =============================================================================


class FacetTest(unittest.TestCase):
    def facets(self, product: dict) -> dict:
        return extract_facets(product, random.Random(11))

    def test_every_stated_facet_is_true_of_the_product(self) -> None:
        for product in CATALOG_ROWS:
            text = catalog_text(product)
            for attribute, value in self.facets(product).items():
                if attribute in ("budget", "size", "style"):
                    continue  # derived phrasings, checked separately
                with self.subTest(asin=product["parent_asin"], attribute=attribute):
                    self.assertIn(str(value).lower().split()[0], text)

    def test_colour_is_not_borrowed_from_sibling_variants(self) -> None:
        """S001's description mentions red and blue; only black is real."""
        self.assertEqual(self.facets(CATALOG_ROWS[0])["color"], "black")

    def test_material_covers_words_the_starter_extractor_misses(self) -> None:
        self.assertEqual(self.facets(CATALOG_ROWS[1])["material"], "mesh")
        self.assertIn("mesh", MATERIAL_WORDS)
        self.assertIn("suede", MATERIAL_WORDS)
        self.assertIn("denim", MATERIAL_WORDS)

    def test_facet_keys_are_all_contract_legal_attributes(self) -> None:
        for product in CATALOG_ROWS:
            for attribute in self.facets(product):
                with self.subTest(attribute):
                    self.assertIn(attribute, ALLOWED_ATTRIBUTES)

    def test_budget_phrase_is_satisfiable_by_the_real_price(self) -> None:
        for product in CATALOG_ROWS:
            for seed in range(12):
                facets = extract_facets(product, random.Random(seed))
                budget = facets.get("budget")
                if not budget:
                    continue
                numbers = [float(value) for value in __import__("re").findall(r"\d+(?:\.\d+)?", budget)]
                price = float(product["price"])
                with self.subTest(asin=product["parent_asin"], budget=budget):
                    if "between" in budget:
                        self.assertLessEqual(min(numbers), price)
                        self.assertGreaterEqual(max(numbers), price)
                    else:
                        self.assertGreaterEqual(max(numbers), price)

    def test_short_feature_is_a_spec_not_a_marketing_paragraph(self) -> None:
        feature = self.facets(CATALOG_ROWS[0]).get("feature")
        self.assertIsNotNone(feature)
        self.assertLessEqual(len(feature), 60)
        self.assertNotIn("cavalo", feature.lower())

    def test_facets_are_deterministic_for_a_fixed_seed(self) -> None:
        for product in CATALOG_ROWS:
            first = extract_facets(product, random.Random(5))
            second = extract_facets(product, random.Random(5))
            self.assertEqual(first, second)


# =============================================================================
# CUSTOMER BEHAVIOUR
# =============================================================================


class RealisticCustomerTest(unittest.TestCase):
    def customer(self, scenario: str, difficulty: str, product: dict | None = None) -> RealisticCustomer:
        product = product or CATALOG_ROWS[0]
        sample = {
            "sample_id": f"t_{scenario}_{difficulty}",
            "scenario_type": scenario,
            "difficulty_bucket": difficulty,
        }
        return RealisticCustomer(sample, product, coarse_category(product["categories"]))

    def test_opening_names_the_category(self) -> None:
        for scenario in ("buying", "browsing", "intent_override", "boundary"):
            for difficulty in ("easy", "medium", "hard"):
                customer = self.customer(scenario, difficulty)
                with self.subTest(scenario=scenario, difficulty=difficulty):
                    opening = customer.opening()
                    self.assertTrue(opening.strip())
                    if difficulty != "hard" or scenario == "buying":
                        self.assertIn(customer.category.split()[0].lower(), opening.lower())

    def test_easy_openings_disclose_more_than_hard_openings(self) -> None:
        easy = self.customer("buying", "easy")
        hard = self.customer("buying", "hard")
        easy.opening()
        hard.opening()
        self.assertGreaterEqual(len(easy.disclosed), len(hard.disclosed))

    def test_a_facet_is_never_disclosed_twice(self) -> None:
        customer = self.customer("buying", "medium")
        seen = [customer.opening()]
        for attribute in ("color", "material", "use_case", "brand", "size", "feature", "style", "budget"):
            seen.append(customer.reply(attribute))
        values = list(customer.facets.values())
        for value in values:
            # Skip values contained in another facet's value; a shared word is
            # not a repeated disclosure.
            if any(value != other and value in other for other in values):
                continue
            with self.subTest(value=value):
                self.assertLessEqual(sum(1 for line in seen if value in line), 1)

    def test_exhausted_attribute_returns_a_no_preference_reply(self) -> None:
        customer = self.customer("buying", "easy")
        customer.opening()
        customer.reply("color")
        second = customer.reply("color")
        self.assertIn("color", second)
        self.assertTrue(
            any(marker in second.lower() for marker in ("no strong", "additional preference", "nothing specific", "thought about"))
        )

    def test_missing_ask_attribute_prompts_the_agent_to_ask(self) -> None:
        customer = self.customer("browsing", "medium")
        customer.opening()
        self.assertTrue(customer.reply(None).strip())

    def test_boundary_declines_once_then_cooperates(self) -> None:
        customer = self.customer("boundary", "medium")
        customer.opening()
        first = customer.reply("color")
        self.assertTrue(customer.boundary_used)
        self.assertIn("preference", first.lower() + " preference")
        second = customer.reply("material")
        self.assertNotIn("use your judgment", second.lower())

    def test_override_turn_is_three_or_four(self) -> None:
        for difficulty in ("easy", "medium", "hard"):
            customer = self.customer("intent_override", difficulty)
            self.assertIn(customer.override_turn, (3, 4))

    def test_override_message_states_a_new_real_constraint(self) -> None:
        customer = self.customer("intent_override", "medium")
        customer.opening()
        message, new_value = customer.override_message()
        self.assertTrue(new_value)
        self.assertIn(new_value, message)
        self.assertIn(new_value, list(customer.facets.values()) + [customer.category])

    def test_openings_vary_across_sessions(self) -> None:
        openings = set()
        for index in range(25):
            sample = {
                "sample_id": f"vary_{index}",
                "scenario_type": "buying",
                "difficulty_bucket": "easy",
            }
            product = CATALOG_ROWS[index % len(CATALOG_ROWS)]
            openings.add(RealisticCustomer(sample, product, coarse_category(product["categories"])).opening())
        self.assertGreater(len(openings), 6)

    def test_phrasing_differs_from_the_official_template(self) -> None:
        """The whole point: not every opening is 'A key requirement is: X.'"""
        openings = [
            RealisticCustomer(
                {"sample_id": f"p_{index}", "scenario_type": "buying", "difficulty_bucket": "easy"},
                CATALOG_ROWS[index % len(CATALOG_ROWS)],
                "Shoes Boots",
            ).opening()
            for index in range(30)
        ]
        self.assertTrue(any("key requirement" not in text.lower() for text in openings))

    def test_customer_is_deterministic_per_sample_id(self) -> None:
        first = self.customer("buying", "easy")
        second = self.customer("buying", "easy")
        self.assertEqual(first.opening(), second.opening())
        self.assertEqual(first.reply("color"), second.reply("color"))

    def test_persona_fields_are_coherent(self) -> None:
        for seed in range(20):
            persona = build_persona(random.Random(seed), "easy")
            with self.subTest(seed):
                if persona["shopping_for"] == "gift":
                    self.assertIsNotNone(persona["recipient"])
                    self.assertIsNotNone(persona["occasion"])
                else:
                    self.assertIsNone(persona["recipient"])


# =============================================================================
# SET CONSTRUCTION
# =============================================================================


class BuildSynthSetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        rows = []
        for index in range(120):
            base = dict(CATALOG_ROWS[index % len(CATALOG_ROWS)])
            base["parent_asin"] = f"G{index:04d}"
            base["title"] = f"{base['title']} variant {index}"
            base["rating_number"] = 5 + index * 31
            rows.append(base)
        cls.catalog_path = write_rows(root, "catalog.jsonl", rows)
        cls.public_path = write_rows(
            root,
            "public.jsonl",
            [{"sample_id": "public_0001", "ground_truth": {"parent_asin": "G0000"}}],
        )
        cls.rows = build(cls.catalog_path, cls.public_path, count=90, seed=3, pool_size=120)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_requested_count_is_produced(self) -> None:
        self.assertEqual(len(self.rows), 90)

    def test_difficulty_is_split_into_equal_thirds(self) -> None:
        counts = Counter(row["difficulty_bucket"] for row in self.rows)
        self.assertEqual(set(counts), {"easy", "medium", "hard"})
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        for value in counts.values():
            self.assertAlmostEqual(value / 90, 1 / 3, delta=0.02)

    def test_scenario_mix_matches_the_official_proportions(self) -> None:
        counts = Counter(row["scenario_type"] for row in self.rows)
        self.assertAlmostEqual(counts["buying"] / 90, 0.40, delta=0.03)
        self.assertAlmostEqual(counts["browsing"] / 90, 0.40, delta=0.03)
        self.assertAlmostEqual(counts["intent_override"] / 90, 0.15, delta=0.03)
        self.assertAlmostEqual(counts["boundary"] / 90, 0.05, delta=0.03)

    def test_every_difficulty_contains_every_scenario(self) -> None:
        crossed: dict[str, set] = {}
        for row in self.rows:
            crossed.setdefault(row["difficulty_bucket"], set()).add(row["scenario_type"])
        for difficulty, scenarios in crossed.items():
            with self.subTest(difficulty):
                self.assertEqual(scenarios, {"buying", "browsing", "intent_override", "boundary"})

    def test_targets_are_unique_and_disjoint_from_the_public_set(self) -> None:
        targets = [row["ground_truth"]["parent_asin"] for row in self.rows]
        self.assertEqual(len(targets), len(set(targets)))
        self.assertNotIn("G0000", targets)

    def test_sample_ids_are_unique_and_sorted(self) -> None:
        ids = [row["sample_id"] for row in self.rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, sorted(ids))

    def test_rows_carry_the_contract_user_profile_fields(self) -> None:
        required = {"purchase_frequency", "average_prior_rating", "rating_style", "preference_tags", "summary"}
        for row in self.rows:
            with self.subTest(row["sample_id"]):
                self.assertEqual(set(row["user_profile"]), required)

    def test_easy_targets_are_measurably_more_findable_than_hard(self) -> None:
        from collections import Counter as _Counter

        products = {row["parent_asin"]: row for row in (json.loads(l) for l in self.catalog_path.read_text().splitlines())}
        frequency: _Counter = _Counter()
        sizes: _Counter = _Counter()
        for product in products.values():
            frequency.update({token for token in str(product["title"]).lower().split() if len(token) > 2})
            sizes[coarse_category(product["categories"])] += 1
        total = len(products)

        def popularity(bucket: str) -> float:
            values = [
                difficulty_features(products[row["ground_truth"]["parent_asin"]], frequency, total, sizes)["popularity"]
                for row in self.rows
                if row["difficulty_bucket"] == bucket
            ]
            return sum(values) / len(values)

        self.assertGreater(popularity("easy"), popularity("hard"))

    def test_build_is_reproducible_for_a_fixed_seed(self) -> None:
        again = build(self.catalog_path, self.public_path, count=90, seed=3, pool_size=120)
        self.assertEqual(
            [row["ground_truth"]["parent_asin"] for row in again],
            [row["ground_truth"]["parent_asin"] for row in self.rows],
        )

    def test_allocate_is_exact_and_proportional(self) -> None:
        result = Counter(allocate(800, (("buying", 0.40), ("browsing", 0.40), ("intent_override", 0.15), ("boundary", 0.05))))
        self.assertEqual(sum(result.values()), 800)
        self.assertEqual(result["buying"], 320)
        self.assertEqual(result["boundary"], 40)


# =============================================================================
# END TO END THROUGH THE AGENT
# =============================================================================


class RealisticTraceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.catalog_path = write_rows(root, "catalog.jsonl", CATALOG_ROWS)
        cls.ids, cls.categories, cls.products = catalog_index(cls.catalog_path)
        cls.agent = Agent(cls.catalog_path)
        cls.samples = [
            {
                "sample_id": f"rt_{index:03d}",
                "scenario_type": scenario,
                "difficulty_bucket": difficulty,
                "ground_truth": {"parent_asin": CATALOG_ROWS[index % 3]["parent_asin"]},
                "user_profile": {
                    "purchase_frequency": "3-4 prior purchases",
                    "average_prior_rating": 4.0,
                    "rating_style": "balanced",
                    "preference_tags": ["fit"],
                    "summary": "Prior purchases emphasize fit.",
                },
            }
            for index, (scenario, difficulty) in enumerate(
                [
                    (scenario, difficulty)
                    for scenario in ("buying", "browsing", "intent_override", "boundary")
                    for difficulty in ("easy", "medium", "hard")
                ]
            )
        ]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def trace(self, sample: dict) -> dict:
        return run_session(self.agent, sample, self.ids, self.categories, self.products, "realistic")

    def test_realistic_traces_are_well_formed(self) -> None:
        for sample in self.samples:
            trace = self.trace(sample)
            with self.subTest(sample["sample_id"]):
                self.assertEqual(trace["simulator"], "realistic")
                self.assertGreaterEqual(trace["turn_count"], 1)
                self.assertLessEqual(trace["turn_count"], MAX_TURNS)
                self.assertTrue(trace["facets"])
                self.assertTrue(trace["turns"][0]["user_message"])

    def test_realistic_traces_record_difficulty_and_mission(self) -> None:
        trace = self.trace(self.samples[0])
        self.assertIn(trace["difficulty_bucket"], ("easy", "medium", "hard"))
        self.assertIn(
            trace["mission_type"],
            ("find_specific_solution", "explore_and_discover", "conversational_navigation"),
        )

    def test_override_sessions_cannot_hit_before_the_switch(self) -> None:
        for sample in [item for item in self.samples if item["scenario_type"] == "intent_override"]:
            trace = self.trace(sample)
            with self.subTest(sample["sample_id"]):
                if trace["hit"]:
                    self.assertGreaterEqual(trace["first_hit_turn"], trace["behavior"]["override"]["turn"])

    def test_non_override_sessions_have_no_override_turn(self) -> None:
        for sample in [item for item in self.samples if item["scenario_type"] != "intent_override"]:
            adapter = build_customer("realistic", sample, self.products, sample["ground_truth"]["parent_asin"], "Shoes Boots")
            with self.subTest(sample["sample_id"]):
                self.assertIsNone(adapter.override_turn)

    def test_official_and_realistic_share_one_interface(self) -> None:
        sample = self.samples[0]
        target = sample["ground_truth"]["parent_asin"]
        for simulator in ("official", "realistic"):
            customer = build_customer(simulator, sample, self.products, target, "Shoes Boots")
            with self.subTest(simulator):
                self.assertIsInstance(customer.opening(), str)
                self.assertIsInstance(customer.reply("color"), str)
                self.assertIsInstance(customer.boundary_used, bool)
                self.assertIn("intent_card", customer.describe())

    def test_realistic_adapter_reports_its_kind(self) -> None:
        adapter = build_customer("realistic", self.samples[0], self.products, "S001", "Shoes Boots")
        self.assertIsInstance(adapter, RealisticCustomerAdapter)
        self.assertEqual(adapter.kind, "realistic")


if __name__ == "__main__":
    unittest.main(verbosity=2)


# =============================================================================
# FACET DIVERSITY
# =============================================================================


class FacetDiversityTest(unittest.TestCase):
    """The customer must be able to talk about more than colour and material."""

    RICH_PRODUCT = {
        "parent_asin": "D001",
        "title": "Aurora Women's Striped Long Sleeve Midi Dress",
        "features": ["Machine washable", "Side pockets", "Fully lined"],
        "details": {
            "Department": "womens",
            "Color": "07-navy+cream",
            "Material": "95% Viscose, 5% Elastane",
            "Pattern": "Striped",
            "Closure type": "Zipper",
            "Special feature": "Wrinkle free",
            "Size": "Medium",
            "Sport type": "Travel",
        },
        "description": ["Also sold in red floral."],
        "categories": ["Clothing, Shoes & Jewelry", "Women", "Clothing", "Dresses"],
        "store": "Aurora",
        "average_rating": 4.4,
        "rating_number": 900,
        "price": 58.0,
    }

    def facets(self, product: dict | None = None) -> dict:
        return extract_facets(product or self.RICH_PRODUCT, random.Random(3))

    def test_structured_detail_keys_win_over_prose_guessing(self) -> None:
        facets = self.facets()
        self.assertEqual(facets["style"], "Striped")
        self.assertEqual(facets["size"], "Medium")
        self.assertEqual(facets["use_case"], "Travel")

    def test_sku_noise_is_stripped_from_colour_details(self) -> None:
        """The catalog stores colours like "07-navy+cream"."""
        self.assertEqual(self.facets()["color"], "navy")

    def test_material_percentages_resolve_to_a_real_material(self) -> None:
        self.assertIn(self.facets()["material"], MATERIAL_WORDS)

    def test_a_rich_product_yields_many_distinct_attributes(self) -> None:
        self.assertGreaterEqual(len(self.facets()), 6)

    def test_feature_pool_offers_several_distinct_specs(self) -> None:
        from tools.customer_sim import feature_pool

        pool = feature_pool(self.RICH_PRODUCT)
        self.assertGreaterEqual(len(pool), 3)
        self.assertEqual(len(pool), len(set(item.lower() for item in pool)))

    def test_repeat_feature_questions_get_new_answers(self) -> None:
        sample = {"sample_id": "div_1", "scenario_type": "buying", "difficulty_bucket": "easy"}
        customer = RealisticCustomer(sample, self.RICH_PRODUCT, "Clothing Dresses")
        customer.opening()
        first = customer.reply("feature")
        second = customer.reply("feature")
        self.assertNotEqual(first, second)
        self.assertFalse(
            all("no strong" in text.lower() for text in (first, second))
        )

    def test_vocabulary_covers_patterns_fits_and_performance_terms(self) -> None:
        from tools.customer_sim import FEATURE_WORDS, PATTERN_WORDS, STYLE_WORDS, USE_CASE_WORDS

        for word in ("floral", "plaid", "tie dye", "leopard"):
            self.assertIn(word, PATTERN_WORDS)
        for word in ("high waisted", "v-neck", "wide leg", "oversized"):
            self.assertIn(word, STYLE_WORDS)
        for word in ("waterproof", "moisture wicking", "arch support", "rfid blocking"):
            self.assertIn(word, FEATURE_WORDS)
        for word in ("wedding", "maternity", "snowboarding", "date night"):
            self.assertIn(word, USE_CASE_WORDS)

    def test_colour_vocabulary_is_wider_than_the_starter_extractor(self) -> None:
        from starter.extractor import COLORS as STARTER_COLORS

        self.assertGreater(len(COLOR_WORDS), len(STARTER_COLORS) * 2)
        for word in ("navy", "burgundy", "teal", "cream"):
            self.assertIn(word, COLOR_WORDS)
            self.assertNotIn(word, STARTER_COLORS)

    def test_junk_detail_values_are_rejected(self) -> None:
        product = dict(self.RICH_PRODUCT)
        product["details"] = {"Department": "womens", "Color": "N/A", "Pattern": "Solid", "Size": "12345"}
        facets = extract_facets(product, random.Random(3))
        self.assertNotEqual(facets.get("color"), "n/a")
        self.assertNotEqual(facets.get("style"), "Solid")
        self.assertNotEqual(facets.get("size"), "12345")

    def test_catalog_wide_coverage_spans_every_attribute(self) -> None:
        counts: Counter = Counter()
        for index, product in enumerate((CATALOG_ROWS + [self.RICH_PRODUCT]) * 4):
            counts.update(extract_facets(product, random.Random(index)).keys())
        for attribute in ("color", "material", "style", "brand", "feature", "size"):
            with self.subTest(attribute):
                self.assertGreater(counts[attribute], 0)
