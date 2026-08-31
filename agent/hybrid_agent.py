"""Phase 2 retrieval agent: fuses lexical/dense/structured routes with config-driven RRF
(retrieval/fusion.py). Deliberately retrieval-only -- ask_attribute is always null and
recommendations are plain RRF order, no reranking or portfolio logic -- so this phase's
contribution to the ablation table isn't entangled with Phase 3's dialog policy or
Phase 4's LLM reranking.

Constructor matches the required Agent(catalog_path) contract (evaluator/local_evaluator.py:306
constructs it that way); config_path is an optional second argument so our own ablation
scripts can point at different route configs (configs/retrieval_*.json) while the
official evaluator's single-arg construction still resolves to our best config.
"""

from __future__ import annotations

import json
from pathlib import Path

from retrieval.catalog import lexical_text, load_catalog
from retrieval.dense import DenseRetriever
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.lexical import LexicalRetriever
from retrieval.query import DialogState
from retrieval.structured import StructuredRetriever

DEFAULT_CONFIG_PATH = "configs/retrieval.json"


def _load_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Agent:
    def __init__(self, catalog_path: str = "data/catalog.jsonl", config_path: str = DEFAULT_CONFIG_PATH):
        self.config = _load_config(config_path)
        self.products = load_catalog(catalog_path)
        ids = list(self.products.keys())

        routes_config = self.config["routes"]
        self._route_names: list[str] = []
        self._route_weights: list[float] = []
        self._lexical: LexicalRetriever | None = None
        self._dense: DenseRetriever | None = None
        self._structured: StructuredRetriever | None = None

        if routes_config.get("lexical", {}).get("enabled"):
            texts = [lexical_text(self.products[asin]) for asin in ids]
            self._lexical = LexicalRetriever(ids, texts)
            self._route_names.append("lexical")
            self._route_weights.append(routes_config["lexical"].get("weight", 1.0))

        if routes_config.get("dense", {}).get("enabled"):
            self._dense = DenseRetriever(
                ids,
                cache_dir=self.config.get("dense_cache_dir", "data/dense_index"),
                model_name=self.config.get("dense_model", "BAAI/bge-small-en-v1.5"),
                model_cache_dir=self.config.get("model_cache_dir", "data/model_cache"),
            )
            self._route_names.append("dense")
            self._route_weights.append(routes_config["dense"].get("weight", 1.0))

        if routes_config.get("structured", {}).get("enabled"):
            self._structured = StructuredRetriever(self.products)
            self._route_names.append("structured")
            self._route_weights.append(routes_config["structured"].get("weight", 1.0))

        if not self._route_names:
            raise ValueError("At least one retrieval route must be enabled in the config")

        self._candidate_k = int(self.config.get("candidate_k", 100))
        self._rrf_k = int(self.config.get("rrf_k", 60))
        self._states: dict[str, DialogState] = {}
        self._profiles: dict[str, dict] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._states[session_id] = DialogState()
        self._profiles[session_id] = user_profile

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._states[session_id]
        state.add_turn(user_message)
        query = state.build_query(self._profiles.get(session_id))

        # Route depth scales with the requested top_k so a diagnostic probe (top_k > 10)
        # actually deepens retrieval instead of being capped at the fusion candidate depth.
        route_k = max(self._candidate_k, top_k)
        ranked_lists: list[list[str]] = []
        for name in self._route_names:
            if name == "lexical":
                ranked_lists.append([asin for asin, _ in self._lexical.search(query, route_k)])
            elif name == "dense":
                ranked_lists.append([asin for asin, _ in self._dense.search(query, route_k)])
            elif name == "structured":
                ranked_lists.append([asin for asin, _ in self._structured.search(query, route_k)])

        fused = reciprocal_rank_fusion(ranked_lists, self._route_weights, self._rrf_k)
        recommendations = [{"parent_asin": asin} for asin in fused[:top_k]]

        return {
            "message": "Here are some options based on what you've told me so far.",
            "ask_attribute": None,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
