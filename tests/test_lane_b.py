from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.contracts import Candidate, ConversationState, SlotValue
from src.retrieval import Retriever


class StubTable:
    def __init__(self, values: dict[tuple[str, str], list[str]] | None = None) -> None:
        self._values = values or {}

    def values(self, asin: str, slot: str) -> list[str]:
        return self._values.get((asin, slot), [])

    def matching(self, slot: str, value: str) -> set[str]:
        return {
            asin for (asin, held_slot), values in self._values.items()
            if held_slot == slot and value in values
        }

    def coverage(self, slot: str) -> float:
        return 1.0


def _row(index: int) -> dict:
    material = "leather" if index == 0 else "cotton"
    return {
        "parent_asin": f"P{index:02d}",
        "title": f"Travel walking shoe number {index}",
        "features": [material, "comfortable cushioned support"],
        "description": ["A lightweight option for long trips"],
        "categories": ["Clothing", "Shoes", "Walking"],
        "details": {"Department": "unisex"},
        "store": f"Brand {index}",
        "price": 40.0 + index,
        "rating_number": 100 - index,
    }


class RetrieverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.catalog = root / "catalog.jsonl"
        self.artifacts = root / "artifacts"
        self.catalog.write_text(
            "".join(json.dumps(_row(index)) + "\n" for index in range(12)),
            encoding="utf-8",
        )
        values = {
            (f"P{index:02d}", "material"): ["leather" if index == 0 else "cotton"]
            for index in range(12)
        }
        self.environment = patch.dict(os.environ, {"TJ_RETRIEVAL_MODE": "bm25"})
        self.environment.start()
        self.retriever = Retriever(self.catalog, self.artifacts, StubTable(values))

    def tearDown(self) -> None:
        self.environment.stop()
        self.temporary.cleanup()

    def test_search_never_returns_fewer_than_ten(self) -> None:
        state = ConversationState("session", {})
        state.history.append(("customer", "I don't have an additional preference for brand."))
        candidates = self.retriever.search(state, top_n=10)
        self.assertEqual(len(candidates), 10)
        self.assertEqual(len({candidate.parent_asin for candidate in candidates}), 10)

    def test_retracted_slot_never_permanently_excludes_a_product(self) -> None:
        state = ConversationState("session", {})
        state.add("material", SlotValue("leather", 0.9, 1, polarity=False))
        candidates = [Candidate(f"P{index:02d}", -float(index)) for index in range(12)]
        ranked = self.retriever.rerank(candidates, state)
        self.assertEqual(len(ranked), 12)
        self.assertIn("P00", {candidate.parent_asin for candidate in ranked})
        leather = next(candidate for candidate in ranked if candidate.parent_asin == "P00")
        self.assertLess(leather.components["slot"], 0.0)

    def test_query_hygiene_drops_filler_and_old_override_text(self) -> None:
        self.assertFalse(Retriever._informative("I don't have an additional preference for brand."))
        self.assertFalse(Retriever._informative("Those options are not quite right yet."))
        state = ConversationState("session", {})
        state.history.extend([
            ("customer", "I'm looking for shoes. I prefer leather."),
            ("customer", "Actually, ignore my earlier preference. What I need is: cotton."),
        ])
        state.add("material", SlotValue("leather", 0.9, 1, polarity=False))
        state.add("material", SlotValue("cotton", 0.9, 2))
        query = self.retriever._query_text(state).lower()
        self.assertNotIn("leather", query)
        self.assertIn("cotton", query)

    def test_search_stays_within_turn_budget(self) -> None:
        state = ConversationState("session", {"preference_tags": ["comfort"]})
        state.history.append(("customer", "comfortable walking shoes for a long trip"))
        self.retriever.search(state, top_n=10)  # warm SQLite page cache
        timings = []
        for _ in range(10):
            started = time.perf_counter()
            self.retriever.search(state, top_n=10)
            timings.append(time.perf_counter() - started)
        self.assertLess(max(timings), 0.050, timings)


if __name__ == "__main__":
    unittest.main()
