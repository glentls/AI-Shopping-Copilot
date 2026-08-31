from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate

from scripts.harness import run_session


CATALOG_ROWS = [
    {
        "parent_asin": "A",
        "title": "Blue running shoe",
        "features": ["cotton"],
        "details": {"department": "womens"},
        "description": ["walking shoe"],
        "categories": ["Clothing", "Shoes"],
        "store": "Example",
        "average_rating": 4.2,
        "rating_number": 10,
        "price": 49.0,
    },
    {
        "parent_asin": "B",
        "title": "Black winter boot",
        "features": ["leather"],
        "details": {"department": "womens"},
        "description": ["winter boot"],
        "categories": ["Clothing", "Boots"],
        "store": "Example",
        "average_rating": 4.4,
        "rating_number": 12,
        "price": 89.0,
    },
] + [
    {
        "parent_asin": f"FILLER{i}",
        "title": f"Filler product {i}",
        "features": [],
        "details": {},
        "description": [],
        "categories": ["Clothing"],
        "store": "Example",
        "average_rating": 4.0,
        "rating_number": 1,
        "price": 10.0,
    }
    for i in range(14)
]

SAMPLE = {
    "sample_id": "public_v2_0001",
    "scenario_type": "buying",
    "user_profile": {"summary": "x"},
    "ground_truth": {"parent_asin": "A"},
}


class EchoTargetAgent:
    """Always returns the target within the scored top-10 -- mirrors
    tests/test_evaluator.py's fixture agent so run_session's top_k=10 path can be
    cross-checked against evaluate() directly."""

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.session_id = session_id

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {"message": "ok", "ask_attribute": None, "recommendations": [{"parent_asin": "A"}]}


class BuriedTargetAgent:
    """Returns the target only at rank 15 -- outside the scored top-10, but inside a
    top_k=20 probe. Simulates a ranking failure: retrievable, not surfaced."""

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.session_id = session_id

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        padding = [{"parent_asin": f"FILLER{i}"} for i in range(14)]
        recs = (padding + [{"parent_asin": "A"}])[:top_k]
        return {"message": "ok", "ask_attribute": None, "recommendations": recs}


class NeverTargetAgent:
    """Never returns the target at any depth. Simulates a retrieval failure."""

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.session_id = session_id

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {"message": "ok", "ask_attribute": None, "recommendations": [{"parent_asin": "B"}]}


class HarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        catalog_path = root / "catalog.jsonl"
        catalog_path.write_text("".join(json.dumps(row) + "\n" for row in CATALOG_ROWS), encoding="utf-8")
        self.catalog_ids, self.categories, self.products = catalog_index(catalog_path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_top_k_10_matches_official_evaluate(self) -> None:
        official = evaluate(EchoTargetAgent(), [SAMPLE], self.catalog_ids, self.categories, self.products)
        session = run_session(EchoTargetAgent(), SAMPLE, self.catalog_ids, self.categories, self.products, top_k=10)

        self.assertEqual(session["hit"], official["sessions"][0]["hit"])
        self.assertEqual(session["first_hit_turn"], official["sessions"][0]["first_hit_turn"])
        self.assertEqual(session["best_rank"], official["sessions"][0]["best_rank"])
        self.assertEqual(session["reciprocal_rank"], official["sessions"][0]["reciprocal_rank"])

    def test_deterministic_across_repeated_runs(self) -> None:
        first = run_session(EchoTargetAgent(), SAMPLE, self.catalog_ids, self.categories, self.products, top_k=10)
        second = run_session(EchoTargetAgent(), SAMPLE, self.catalog_ids, self.categories, self.products, top_k=10)
        first.pop("sample_id"), second.pop("sample_id")  # session_id/uuid differ by design; strip nothing else
        self.assertEqual(first, second)

    def test_buried_target_is_a_ranking_failure_not_a_retrieval_failure(self) -> None:
        session = run_session(BuriedTargetAgent(), SAMPLE, self.catalog_ids, self.categories, self.products, top_k=20)
        self.assertFalse(session["hit"])
        self.assertTrue(session["recall_hit"])

    def test_never_shown_target_is_a_retrieval_failure(self) -> None:
        session = run_session(NeverTargetAgent(), SAMPLE, self.catalog_ids, self.categories, self.products, top_k=20)
        self.assertFalse(session["hit"])
        self.assertFalse(session["recall_hit"])


if __name__ == "__main__":
    unittest.main()
