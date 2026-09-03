from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llm.cache import LLMCache, cache_key
from llm.reranker import LLMReranker


class CacheTest(unittest.TestCase):
    def test_miss_then_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LLMCache(tmp)
            key = cache_key("model", "query", ["A", "B"])
            self.assertIsNone(cache.get(key))
            cache.put(key, {"ranked_asins": ["B", "A"], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
            self.assertEqual(cache.get(key)["ranked_asins"], ["B", "A"])

    def test_key_is_sensitive_to_candidate_set_and_query(self) -> None:
        k1 = cache_key("m", "blue shoes", ["A", "B"])
        k2 = cache_key("m", "blue shoes", ["A", "C"])
        k3 = cache_key("m", "red shoes", ["A", "B"])
        self.assertNotEqual(k1, k2)
        self.assertNotEqual(k1, k3)


class RerankerFallbackTest(unittest.TestCase):
    """No real API calls: these tests only exercise the fallback paths and the
    cache-hit path, which is all that's reachable without ANTHROPIC_API_KEY set."""

    def test_no_api_key_falls_back_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {}, clear=True):
            reranker = LLMReranker(cache_dir=tmp)
            candidates = [{"parent_asin": "A", "title": "x"}, {"parent_asin": "B", "title": "y"}]
            ranked, usage = reranker.rerank("blue shoes", candidates)
            self.assertIsNone(ranked)
            self.assertEqual(usage, {"prompt_tokens": 0, "completion_tokens": 0})

    def test_empty_candidates_falls_back_without_touching_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reranker = LLMReranker(cache_dir=tmp)
            ranked, usage = reranker.rerank("blue shoes", [])
            self.assertIsNone(ranked)
            self.assertEqual(usage, {"prompt_tokens": 0, "completion_tokens": 0})

    def test_cache_hit_short_circuits_before_client_construction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reranker = LLMReranker(cache_dir=tmp)
            candidates = [{"parent_asin": "A", "title": "x"}, {"parent_asin": "B", "title": "y"}]
            key = cache_key(reranker.model, "blue shoes", ["A", "B"])
            reranker.cache.put(key, {"ranked_asins": ["B", "A"], "usage": {"prompt_tokens": 42, "completion_tokens": 7}})

            # No ANTHROPIC_API_KEY needed -- the cache hit must return before any
            # client/network path is touched.
            with mock.patch.dict("os.environ", {}, clear=True):
                ranked, usage = reranker.rerank("blue shoes", candidates)
            self.assertEqual(ranked, ["B", "A"])
            self.assertEqual(usage, {"prompt_tokens": 42, "completion_tokens": 7})

    def test_client_construction_failure_is_swallowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake"}):
            reranker = LLMReranker(cache_dir=tmp)
            # Simulate the anthropic import/construction itself failing (e.g. package
            # not installed in a judge's minimal environment) -- must not raise.
            with mock.patch("anthropic.Anthropic", side_effect=RuntimeError("boom")):
                candidates = [{"parent_asin": "A", "title": "x"}]
                ranked, usage = reranker.rerank("blue shoes", candidates)
            self.assertIsNone(ranked)
            self.assertEqual(usage, {"prompt_tokens": 0, "completion_tokens": 0})


if __name__ == "__main__":
    unittest.main()
