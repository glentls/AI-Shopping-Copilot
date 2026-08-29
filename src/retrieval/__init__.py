"""Conversation state to hybrid-retrieval candidates."""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.attributes import AttributeTable
from src.contracts import SLOTS, Candidate, ConversationState
from src.lexicons import NO_PREFERENCE_RE, OVERRIDE_CUES

from .blend import reciprocal_rank_fusion, rerank_candidates
from .bm25 import ARTIFACT_NAME as BM25_ARTIFACT, BM25Index, build_bm25_index
from .dense import DenseIndex


_EXACT_CLAUSE_RE = re.compile(
    r"(?:key requirement is|what matters is|what i need is)\s*:\s*(.+)", re.IGNORECASE
)
_GENERIC_ONLY_RE = re.compile(
    r"^(?:those options are not quite right yet|ask me about (?:one |a )?specific attribute|"
    r"i(?:'m| am) still exploring|nothing else|no idea)[.! ]*$",
    re.IGNORECASE,
)
_OVERRIDE_RE = re.compile("|".join(re.escape(cue) for cue in OVERRIDE_CUES), re.IGNORECASE)


class Retriever:
    def __init__(
        self,
        catalog_path: str | Path,
        artifacts_dir: str | Path,
        table: AttributeTable,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.table = table
        bm25_path = self.artifacts_dir / BM25_ARTIFACT
        if not bm25_path.exists():
            # Useful for unit tests and tiny custom catalogs. A real 50k
            # deployment should always run tools.build_index ahead of time.
            build_bm25_index(self.catalog_path, self.artifacts_dir)
        self.bm25 = BM25Index(bm25_path)
        self.metadata, self.fallback = self.bm25.metadata()
        self._profile_cache: dict[tuple[str, ...], dict[str, int]] = {}
        self.mode = os.environ.get("TJ_RETRIEVAL_MODE", "fused").strip().lower()
        if self.mode not in {"bm25", "dense", "fused"}:
            self.mode = "fused"
        try:
            self.dense = None if self.mode == "bm25" else DenseIndex(self.artifacts_dir)
        except (FileNotFoundError, ImportError, ValueError):
            self.dense = None
        self._route_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="dense-retrieval")
            if self.dense is not None and self.mode in {"dense", "fused"}
            else None
        )

    @staticmethod
    def _informative(text: str) -> bool:
        """Reject replies that communicate no product preference."""
        cleaned = " ".join((text or "").strip().split())
        return (
            bool(cleaned)
            and not NO_PREFERENCE_RE.search(cleaned)
            and not _GENERIC_ONLY_RE.match(cleaned)
        )

    @staticmethod
    def _without_excluded(text: str, state: ConversationState) -> str:
        for slot in SLOTS:
            for value in state.slots.get(slot, []):
                if value.polarity or not value.value:
                    continue
                text = re.sub(rf"\b{re.escape(value.value)}\b", " ", text, flags=re.IGNORECASE)
        return text

    def _semantic_query_text(self, state: ConversationState) -> str:
        """Current intent only, suitable for semantic retrieval."""
        messages = [
            text for role, text in state.history
            if role == "customer" and self._informative(text)
        ]
        # Once the customer says to ignore an earlier preference, raw history
        # before that point is unsafe. Preserve the initial category clause,
        # then use the override and everything learned after it.
        override_at = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if _OVERRIDE_RE.search(messages[index])
            ),
            None,
        )
        if override_at is not None and override_at > 0:
            category_clause = messages[0].split(".", 1)[0]
            messages = [category_clause, *messages[override_at:]]
        said = " ".join(messages)
        said = self._without_excluded(said, state)
        live_values = " ".join(
            value.value
            for slot in SLOTS
            for value in state.slots.get(slot, [])
            if value.polarity and slot != "budget"
        )
        return " ".join(f"{said} {live_values}".split())

    def _query_text(self, state: ConversationState) -> str:
        """All informative lexical evidence plus the current live slots.

        Exact catalog words disclosed earlier remain useful BM25 evidence even
        after an override. Contradicted values affect the soft slot reranker;
        they never hard-delete an otherwise strong lexical candidate.
        """
        said = " ".join(
            text
            for role, text in state.history
            if role == "customer" and self._informative(text)
        )
        live_values = " ".join(
            value.value
            for slot in SLOTS
            for value in state.slots.get(slot, [])
            if value.polarity and slot != "budget"
        )
        return " ".join(f"{said} {live_values}".split())

    @staticmethod
    def _exact_phrases(state: ConversationState) -> list[str]:
        phrases: list[str] = []
        messages = [
            text for role, text in state.history
            if role == "customer" and Retriever._informative(text)
        ]
        override_at = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if _OVERRIDE_RE.search(messages[index])
            ),
            None,
        )
        if override_at is not None:
            messages = messages[override_at:]
        for text in messages:
            match = _EXACT_CLAUSE_RE.search(text)
            if not match:
                continue
            phrases.extend(
                part.strip(" .")
                for part in match.group(1).split(";")
                if part.strip(" .")
            )
        return phrases

    def search(self, state: ConversationState, top_n: int = 300) -> list[Candidate]:
        top_n = max(10, min(int(top_n), len(self.metadata))) if self.metadata else 0
        if top_n <= 0:
            return []
        lexical_query = self._query_text(state)
        semantic_query = self._semantic_query_text(state)
        route_limit = max(300, top_n)
        exact_phrases = self._exact_phrases(state)
        informative_messages = [
            text.lower() for role, text in state.history
            if role == "customer" and self._informative(text)
        ]
        generic_browsing = bool(informative_messages) and all(
            "still exploring" in text for text in informative_messages
        )

        dense_future = (
            self._route_executor.submit(self.dense.search, semantic_query, route_limit)
            if self._route_executor is not None
            else None
        )
        bm25_hits = (
            self.bm25.search(lexical_query, route_limit)
            if self.mode in {"bm25", "fused"}
            else []
        )
        dense_hits = dense_future.result() if dense_future is not None else []
        exact_hits = (
            self.bm25.exact_search(exact_phrases, route_limit)
            if self.mode == "fused"
            else []
        )
        tags = state.user_profile.get("preference_tags", []) if state.user_profile else []
        if not isinstance(tags, list):
            tags = []
        tag_key = tuple(sorted({str(tag).strip().lower() for tag in tags if str(tag).strip()}))
        if tag_key not in self._profile_cache:
            self._profile_cache[tag_key] = self.bm25.profile_ranks(tag_key)
        profile_ranks = self._profile_cache[tag_key]
        return reciprocal_rank_fusion(
            bm25_hits,
            dense_hits,
            exact_hits,
            profile_ranks,
            self.metadata,
            self.fallback,
            top_n,
            # Natural requests need dense-only candidates to clear the BM25
            # pool. Verbatim catalog clauses are already high-confidence
            # lexical evidence, so semantic retrieval becomes a light hedge.
            dense_weight=0.10 if exact_phrases or generic_browsing else 1.0,
        )

    def rerank(self, cands: list[Candidate], state: ConversationState) -> list[Candidate]:
        return rerank_candidates(cands, state, self.table, self.metadata)


__all__ = ["Retriever"]
