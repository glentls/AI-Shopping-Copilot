from __future__ import annotations

import unittest

from starter.personalization import (
    compute_profile_boost,
    parse_profile,
    profile_adjusted_priority,
)
from starter.retrieval.catalog_store import ProductMeta


def _product(**overrides: object) -> ProductMeta:
    base = {
        "parent_asin": "A",
        "title": "Comfort fit running shoe",
        "categories": "Shoes",
        "features": "mesh upper comfort cushioning",
        "details": "department womens",
        "store": "RunCo",
        "description": "durable daily trainer",
        "price": 49.0,
        "average_rating": 4.6,
        "rating_number": 120,
        "searchable_text": "comfort fit running shoe mesh upper comfort cushioning durable daily trainer",
    }
    base.update(overrides)
    return ProductMeta(**base)  # type: ignore[arg-type]


class ProfileSignalsTest(unittest.TestCase):
    def test_parse_maps_material_and_comfort_tags(self) -> None:
        signals = parse_profile(
            {
                "preference_tags": ["material", "comfort", "fit"],
                "rating_style": "usually positive",
                "average_prior_rating": 5.0,
                "summary": "Prior purchases emphasize material, comfort, fit.",
            }
        )
        self.assertEqual(signals.tags, ("material", "comfort", "fit"))
        self.assertEqual(signals.preferred_attributes, ("material", "feature"))

    def test_parse_handles_empty_profile(self) -> None:
        signals = parse_profile({})
        self.assertEqual(signals.tags, ())
        self.assertEqual(signals.preferred_attributes, ())


class RerankBoostTest(unittest.TestCase):
    def test_tag_match_increases_score(self) -> None:
        matching = {"preference_tags": ["comfort", "fit"], "rating_style": "mixed"}
        unrelated = {"preference_tags": ["waterproof-xyz"], "rating_style": "mixed"}
        self.assertGreater(
            compute_profile_boost(_product(), matching),
            compute_profile_boost(_product(), unrelated),
        )

    def test_profile_boost_is_capped(self) -> None:
        profile = {
            "preference_tags": ["comfort", "fit", "material", "style", "durability"],
            "rating_style": "usually positive",
            "summary": "comfort fit material style durability performance warmth",
        }
        boost = compute_profile_boost(_product(), profile)
        self.assertLessEqual(boost, 8.0)

    def test_critical_shoppers_prefer_high_rated_products(self) -> None:
        profile = {"preference_tags": [], "rating_style": "critical"}
        high = compute_profile_boost(_product(average_rating=4.6), profile)
        low = compute_profile_boost(_product(average_rating=3.0), profile)
        self.assertGreater(high, low)


class QuestionPriorityTest(unittest.TestCase):
    def test_profile_attributes_move_to_front(self) -> None:
        profile = {"preference_tags": ["material", "comfort"]}
        priority = profile_adjusted_priority(
            ("category", "budget", "material", "feature"),
            profile,
        )
        self.assertEqual(priority[0], "material")
        self.assertEqual(priority[1], "feature")
        self.assertIn("category", priority)


if __name__ == "__main__":
    unittest.main()
