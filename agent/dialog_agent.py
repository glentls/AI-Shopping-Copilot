"""Phase 3 agent: Phase 2's retrieval stack (lexical/dense/structured + RRF) plus a
dialog policy layer -- proven-negative exclusion, posterior downweighting, portfolio
(coverage) ranking with an explore/exploit schedule, and EIG question selection with
contradiction-based slot override. Every dialog feature is config-toggleable so each is
independently ablatable (docs/ablations.md).

Constructor matches the required Agent(catalog_path) contract; config_path is an
optional second arg for our own ablation scripts, exactly like agent/hybrid_agent.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from dialog.portfolio import portfolio_rerank
from dialog.posterior import RejectionTracker
from dialog.question_policy import choose_attribute
from dialog.slots import ASK_ATTRIBUTE_BLOCKLIST, SlotState
from llm.reranker import DEFAULT_MODEL as DEFAULT_LLM_MODEL
from llm.reranker import LLMReranker
from retrieval.catalog import lexical_text, load_catalog
from retrieval.dense import DenseRetriever
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.lexical import LexicalRetriever
from retrieval.query import DialogState
from retrieval.structured import StructuredRetriever

DEFAULT_CONFIG_PATH = "configs/dialog.json"

FIXED_QUESTION_ORDER = ("material", "color", "size", "budget", "style", "use_case", "feature")


def _load_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class _SessionState:
    def __init__(self) -> None:
        self.dialog = DialogState()
        self.slots = SlotState()
        self.shown: set[str] = set()
        self.last_shown: list[str] = []
        self.rejection_tracker = RejectionTracker()
        self.profile: dict | None = None


class Agent:
    def __init__(self, catalog_path: str = "data/catalog.jsonl", config_path: str = DEFAULT_CONFIG_PATH):
        self.config = _load_config(config_path)
        self.products = load_catalog(catalog_path)
        ids = list(self.products.keys())

        routes_config = self.config["routes"]
        self._route_names: list[str] = []
        self._route_weights: list[float] = []
        self._lexical = self._dense = self._structured = None

        if routes_config.get("lexical", {}).get("enabled"):
            texts = [lexical_text(self.products[asin]) for asin in ids]
            self._lexical = LexicalRetriever(ids, texts)
            self._route_names.append("lexical")
            self._route_weights.append(routes_config["lexical"].get("weight", 1.0))

        if routes_config.get("dense", {}).get("enabled"):
            self._dense = DenseRetriever(
                ids,
                cache_dir=self.config.get("dense_cache_dir", "data/dense_index"),
                model_name=self.config.get("dense_model", "sentence-transformers/all-MiniLM-L6-v2"),
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

        self._candidate_k = int(self.config.get("candidate_k", 300))
        self._rrf_k = int(self.config.get("rrf_k", 60))

        dialog_cfg = self.config.get("dialog", {})
        self._proven_negative_filter: bool = dialog_cfg.get("proven_negative_filter", False)
        self._posterior_downweight: bool = dialog_cfg.get("posterior_downweight", False)
        self._portfolio_ranking: bool = dialog_cfg.get("portfolio_ranking", False)
        self._explore_exploit: bool = dialog_cfg.get("explore_exploit", False)
        self._question_policy: str = dialog_cfg.get("question_policy", "none")  # "none" | "fixed" | "eig"

        llm_cfg = self.config.get("llm_rerank", {})
        self._llm_rerank_enabled: bool = llm_cfg.get("enabled", False)
        self._llm_candidate_depth: int = int(llm_cfg.get("candidate_depth", 50))
        self._reranker: LLMReranker | None = None
        if self._llm_rerank_enabled:
            self._reranker = LLMReranker(
                model=llm_cfg.get("model", DEFAULT_LLM_MODEL),
                timeout_seconds=float(llm_cfg.get("timeout_seconds", 20.0)),
                cache_dir=llm_cfg.get("cache_dir", "data/llm_cache"),
            )

        self._sessions: dict[str, _SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = _SessionState()
        self._sessions[session_id].profile = user_profile

    def _retrieve_fused(self, query: str, route_k: int) -> list[str]:
        ranked_lists: list[list[str]] = []
        for name in self._route_names:
            if name == "lexical":
                ranked_lists.append([asin for asin, _ in self._lexical.search(query, route_k)])
            elif name == "dense":
                ranked_lists.append([asin for asin, _ in self._dense.search(query, route_k)])
            elif name == "structured":
                ranked_lists.append([asin for asin, _ in self._structured.search(query, route_k)])
        return reciprocal_rank_fusion(ranked_lists, self._route_weights, self._rrf_k)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions[session_id]

        if self._posterior_downweight and state.last_shown:
            state.rejection_tracker.record_rejected_batch(state.last_shown, self.products)

        state.dialog.add_turn(user_message)
        just_overridden = state.slots.update(user_message)
        if just_overridden:
            # A contradicted slot means the old intent's rejections don't transfer:
            # items excluded (or downweighted) because they didn't match the *old*
            # value shouldn't stay excluded once the target attribute has changed.
            #
            # Known limitation, disclosed rather than papered over: this only fires when
            # the override re-uses an attribute we already track (material/color/size/
            # budget) AND conflicts with an already-filled value for it. When the
            # override instead introduces a *new* attribute (common -- see
            # docs/ablations.md), no contradiction is flagged and stale exclusions can
            # persist across the override boundary. A turn-to-turn embedding-similarity
            # pivot detector was tried and rejected here: calibration against 8 real
            # override transcripts showed no threshold cleanly separates override from
            # non-override turns (override similarities 0.165-0.463 vs. non-override
            # -0.01-0.577 -- heavy overlap), so it would have added noise, not signal.
            state.shown.clear()
            state.rejection_tracker.counts.clear()
        query = state.dialog.build_query(state.profile)

        route_k = max(self._candidate_k, top_k)
        fused = self._retrieve_fused(query, route_k)

        if self._proven_negative_filter:
            fused = [asin for asin in fused if asin not in state.shown]

        if self._portfolio_ranking:
            counts = state.rejection_tracker.counts if self._posterior_downweight else None
            final = portfolio_rerank(
                fused, self.products, top_k, turn, rejected_value_counts=counts, use_schedule=self._explore_exploit
            )
        else:
            final = fused[:top_k]

        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        if self._llm_rerank_enabled and self._reranker is not None:
            # Reranks the *wider* fused pool, not the already-narrowed `final` list --
            # CLAUDE.md Phase 4: "Rerank the fused top-50 down to the final 10." Falls
            # back to whatever Phase 3 already produced (`final`) on any failure: no
            # key, network error, timeout, or a malformed tool response.
            pool = [self.products[asin] for asin in fused[: self._llm_candidate_depth] if asin in self.products]
            llm_ranked, usage = self._reranker.rerank(query, pool)
            if llm_ranked is not None:
                final = llm_ranked[:top_k]

        state.shown.update(final)
        state.last_shown = final

        ask_attribute = self._choose_ask_attribute(state, fused)

        message = (
            f"Do you have a {ask_attribute} preference?"
            if ask_attribute
            else "Here are some options based on what you've told me so far."
        )
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [{"parent_asin": asin} for asin in final],
            "usage": usage,
        }

    def _choose_ask_attribute(self, state: _SessionState, fused: list[str]) -> str | None:
        if self._question_policy == "none":
            return None
        filled = set(state.slots.values.keys())
        if self._question_policy == "fixed":
            for attribute in FIXED_QUESTION_ORDER:
                if attribute not in filled and attribute not in ASK_ATTRIBUTE_BLOCKLIST:
                    return attribute
            return None
        if self._question_policy == "eig":
            pool = [self.products[asin] for asin in fused[:50] if asin in self.products]
            return choose_attribute(pool, filled)
        raise ValueError(f"Unknown question_policy: {self._question_policy!r}")
