from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config import load_config
from src.contracts import RetrievalRequest
from src.retrieval.bm25 import build_index, search


class RetrievalSoftPreferenceTest(unittest.TestCase):
    def test_matching_soft_preference_changes_candidate_order(self) -> None:
        rows = [
            {"parent_asin": "A", "title": "classic shoe", "categories": ["Shoes"], "features": [], "description": [], "store": "Store", "details": {}, "average_rating": 4.0, "rating_number": 10},
            {"parent_asin": "B", "title": "classic shoe", "categories": ["Shoes"], "features": ["red color"], "description": [], "store": "Store", "details": {}, "average_rating": 4.0, "rating_number": 9},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            config = load_config()
            config = {**config, "retrieval": {**config["retrieval"], "soft_pref_match_weight": 10.0}}
            result = search(
                build_index(path, config),
                RetrievalRequest("classic shoe", "buy", {}, {"color": {"red": 1.0}}, 2),
                config,
            )
        self.assertEqual([candidate.parent_asin for candidate in result], ["B", "A"])


if __name__ == "__main__":
    unittest.main()
