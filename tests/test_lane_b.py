from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.contracts import Candidate, ConversationState, SlotValue
from src.retrieval import Retriever
from src.retrieval.bm25 import BM25Index, build_bm25_index
from src.retrieval.blend import rerank_candidates


class StubTable:
    def __init__(
        self,
        values: dict[tuple[str, str], list[str]] | None = None,
        confidence: dict[tuple[str, str, str], float] | None = None,
    ) -> None:
        self._values = values or {}
        self._confidence = confidence or {}

    def values(self, asin: str, slot: str) -> list[str]:
        return self._values.get((asin, slot), [])

    def matching(self, slot: str, value: str) -> set[str]:
        return {
            asin for (asin, held_slot), values in self._values.items()
            if held_slot == slot and value in values
        }

    def coverage(self, slot: str) -> float:
        return 1.0

    def confidence(self, asin: str, slot: str, value: str) -> float:
        return self._confidence.get((asin, slot, value), 1.0)


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

    def test_slot_bonus_is_weighted_by_confidence(self) -> None:
        state = ConversationState("session", {})
        state.add("material", SlotValue("leather", 0.25, 1))
        candidate = Candidate("P00", 0.0)
        with patch.dict(os.environ, {"TJ_SLOT_WEIGHT": "12"}):
            self.retriever.rerank([candidate], state)
        self.assertAlmostEqual(candidate.components["slot"], 3.0)

    def test_slot_bonus_uses_catalog_source_confidence(self) -> None:
        state = ConversationState("session", {})
        state.add("material", SlotValue("leather", 0.5, 1))
        candidate = Candidate("P00", 0.0)
        table = StubTable(
            {("P00", "material"): ["leather"]},
            {("P00", "material", "leather"): 0.5},
        )
        with patch.dict(
            os.environ,
            {"TJ_SLOT_WEIGHT": "12", "TJ_CATALOG_CONFIDENCE_POWER": "1"},
        ):
            reranked = rerank_candidates(
                [candidate], state, table, self.retriever.metadata
            )
        self.assertAlmostEqual(reranked[0].components["slot"], 3.0)

    def test_missing_catalog_confidence_preserves_the_slot_match(self) -> None:
        state = ConversationState("session", {})
        state.add("material", SlotValue("leather", 0.5, 1))
        candidate = Candidate("P00", 0.0)
        table = StubTable(
            {("P00", "material"): ["leather"]},
            {("P00", "material", "leather"): 0.0},
        )
        with patch.dict(os.environ, {"TJ_SLOT_WEIGHT": "12"}):
            reranked = rerank_candidates(
                [candidate], state, table, self.retriever.metadata
            )
        self.assertAlmostEqual(reranked[0].components["slot"], 6.0)

    def test_query_hygiene_drops_replaced_value_but_keeps_later_constraints(self) -> None:
        self.assertFalse(Retriever._informative("I don't have an additional preference for brand."))
        self.assertFalse(Retriever._informative("Those options are not quite right yet."))
        state = ConversationState("session", {})
        state.history.extend([
            ("customer", "I'm looking for shoes. I prefer leather."),
            ("customer", "For that, what matters is: waterproof."),
            ("customer", "Actually, ignore my earlier preference. What I need is: cotton."),
        ])
        state.add("material", SlotValue("leather", 0.9, 1, polarity=False))
        state.add("feature", SlotValue("waterproof", 0.9, 2))
        state.add("material", SlotValue("cotton", 0.9, 3))

        for query in (
            self.retriever._query_text(state),
            self.retriever._semantic_query_text(state),
            " ".join(self.retriever._exact_phrases(state)),
        ):
            with self.subTest(query=query):
                lowered = query.lower()
                self.assertNotIn("leather", lowered)
                self.assertIn("waterproof", lowered)
                self.assertIn("cotton", lowered)

    def test_query_hygiene_drops_only_the_no_requirement_clause(self) -> None:
        state = ConversationState("session", {})
        state.history.append((
            "customer",
            "No brand requirement, but I would like a zipper on the dress.",
        ))

        query = self.retriever._query_text(state).lower()

        self.assertNotIn("brand", query)
        self.assertNotIn("requirement", query)
        self.assertIn("zipper", query)
        self.assertIn("dress", query)

    def test_porter_route_matches_plural_to_singular(self) -> None:
        root = Path(self.temporary.name) / "porter-case"
        root.mkdir()
        catalog = root / "catalog.jsonl"
        artifacts = root / "artifacts"
        rows = [
            {
                **_row(0),
                "parent_asin": "TARGET",
                "title": "Rayon Strap Celebrity Midi Dress",
                "features": [],
                "description": [],
                "categories": ["Clothing", "Dresses"],
                "details": {},
            },
            {
                **_row(1),
                "parent_asin": "OTHER",
                "title": "Long Sleeve Evening Gown",
                "features": [],
                "description": [],
                "categories": ["Clothing", "Dresses"],
                "details": {},
            },
        ]
        catalog.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        index = BM25Index(build_bm25_index(catalog, artifacts))

        self.assertNotIn("TARGET", {hit.parent_asin for hit in index.search("straps", 10)})
        self.assertIn(
            "TARGET",
            {hit.parent_asin for hit in index.stemmed_search("straps", 10)},
        )

    def test_reasserted_value_is_not_scrubbed_from_queries(self) -> None:
        state = ConversationState("session", {})
        state.history.extend([
            ("customer", "I need leather shoes."),
            ("customer", "Actually, leather is still what I need."),
        ])
        state.add("material", SlotValue("leather", 0.9, 1, polarity=False))
        state.add("material", SlotValue("leather", 0.95, 2))

        self.assertIn("leather", self.retriever._query_text(state).lower())

    def test_profile_context_only_enriches_browsing_semantics(self) -> None:
        browsing = ConversationState(
            "browse", {"preference_tags": ["comfort", "fit"]}
        )
        browsing.history.append(("customer", "I'm still exploring."))
        buying = ConversationState(
            "buy", {"preference_tags": ["comfort", "fit"]}
        )
        buying.history.append(("customer", "I need leather shoes."))
        buying.add("material", SlotValue("leather", 0.95, 1))

        self.assertIn(
            "preferences: comfort fit",
            self.retriever._semantic_query_text(browsing),
        )
        self.assertNotIn(
            "preferences:", self.retriever._semantic_query_text(buying)
        )

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

    def test_agent_import_does_not_require_numpy(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-S", "-c", "from starter.agent import Agent"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_agent_falls_back_to_bm25_without_numpy(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                (
                    "import sys; from starter.agent import Agent; "
                    "agent = Agent(sys.argv[1], sys.argv[2]); "
                    "assert agent.retriever.dense is None"
                ),
                str(self.catalog),
                str(self.artifacts),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_skip_dense_build_does_not_require_numpy(self) -> None:
        target = Path(self.temporary.name) / "skip-dense-artifacts"
        completed = subprocess.run(
            [
                sys.executable,
                "-S",
                "-m",
                "tools.build_index",
                "--skip-dense",
                "--catalog",
                str(self.catalog),
                "--artifacts",
                str(target),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue((target / "bm25.sqlite3").exists())


if __name__ == "__main__":
    unittest.main()
