"""Conversation state to hybrid-retrieval candidates."""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.attributes import AttributeTable
from src.contracts import SLOTS, Candidate, ConversationState
from src.extract import replaces_earlier_preference
from src.lexicons import NO_PREFERENCE_RE
from src.orchestration import compile_context_program

from .blend import lock_hard_constraints, reciprocal_rank_fusion, rerank_candidates
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

_NO_REQUIREMENT_RE = re.compile(
    r"\b(?:"
    r"(?:i\s+)?(?:have\s+)?no\s+(?:specific\s+|particular\s+|strong\s+)?"
    r"(?:(?:brand|color|colour|material|size|style|feature|category)\s+)?"
    r"(?:requirement|preference)"
    r"|(?:without|skip)\s+(?:a\s+)?(?:brand|color|colour|material|size|style)\s+"
    r"(?:requirement|preference)"
    r")\b",
    re.IGNORECASE,
)


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
        try:
            self.bm25 = BM25Index(bm25_path)
        except ValueError:
            # Schema upgrades (for example adding the Porter table) should not
            # leave a previously built checkout unable to start.
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
    def _clean_message(text: str) -> str:
        """Remove no-constraint clauses without discarding useful siblings."""
        source = " ".join((text or "").strip().split())
        if not source:
            return ""
        clauses = re.split(r"(?<=[,;.!?])\s*", source)
        kept = [
            clause.strip()
            for clause in clauses
            if clause.strip()
            and not NO_PREFERENCE_RE.search(clause)
            and not _NO_REQUIREMENT_RE.search(clause)
        ]
        return " ".join(kept)

    @staticmethod
    def _informative(text: str) -> bool:
        """Reject replies that communicate no product preference."""
        cleaned = Retriever._clean_message(text)
        return (
            bool(cleaned)
            and not _GENERIC_ONLY_RE.match(cleaned)
        )

    @staticmethod
    def _without_excluded(text: str, state: ConversationState) -> str:
        for slot in SLOTS:
            for value in state.excluded(slot):
                if not value:
                    continue
                text = re.sub(rf"\b{re.escape(value)}\b", " ", text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _intent_messages(state: ConversationState) -> list[str]:
        """Informative customer text with a replaced opener preference retired.

        A broad override replaces the preference after the opener's category
        clause, not useful constraints disclosed on turns two and three. Slot
        exclusions scrub extracted stale values below; trimming this one raw
        clause also handles catalog phrases the deterministic extractor does
        not understand.
        """
        messages = []
        for role, text in state.history:
            if role != "customer":
                continue
            cleaned = Retriever._clean_message(text)
            if Retriever._informative(cleaned):
                messages.append(cleaned)
        if messages and any(
            replaces_earlier_preference(text) for text in messages[1:]
        ):
            messages[0] = messages[0].split(".", 1)[0]
        return messages

    def _semantic_query_text(self, state: ConversationState) -> str:
        """Current intent only, suitable for semantic retrieval."""
        program = compile_context_program(state)
        messages = self._intent_messages(state)
        said = " ".join(messages)
        said = self._without_excluded(said, state)
        live_values = " ".join(
            value.value
            for slot in SLOTS
            for value in state.slots.get(slot, [])
            if value.polarity and slot != "budget"
        )
        profile_context = (
            "preferences: " + " ".join(program.profile_terms)
            if program.route == "browsing" and program.profile_terms
            else ""
        )
        return " ".join(f"{said} {live_values} {profile_context}".split())

    def _query_text(self, state: ConversationState) -> str:
        """Current lexical evidence plus live slots, with stale values scrubbed."""
        said = " ".join(self._intent_messages(state))
        said = self._without_excluded(said, state)
        live_values = " ".join(
            value.value
            for slot in SLOTS
            for value in state.slots.get(slot, [])
            if value.polarity and slot != "budget"
        )
        return " ".join(f"{said} {live_values}".split())

    @classmethod
    def _exact_phrases(cls, state: ConversationState) -> list[str]:
        phrases: list[str] = []
        for text in cls._intent_messages(state):
            match = _EXACT_CLAUSE_RE.search(text)
            if not match:
                continue
            for part in match.group(1).split(";"):
                cleaned = cls._without_excluded(part, state).strip(" .")
                if cleaned:
                    phrases.append(cleaned)
        return phrases

    def search(self, state: ConversationState, top_n: int = 300) -> list[Candidate]:
        program = compile_context_program(state)
        top_n = max(10, min(int(top_n), len(self.metadata))) if self.metadata else 0
        if top_n <= 0:
            return []
        lexical_query = self._query_text(state)
        semantic_query = self._semantic_query_text(state)
        route_limit = max(300, top_n)
        exact_phrases = self._exact_phrases(state)
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
        stemmed_hits = (
            self.bm25.stemmed_search(lexical_query, route_limit)
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
        fusion_limit = min(top_n, program.candidate_cutoff)
        return reciprocal_rank_fusion(
            bm25_hits,
            stemmed_hits,
            dense_hits,
            exact_hits,
            profile_ranks,
            self.metadata,
            self.fallback,
            fusion_limit,
            # Natural requests need dense-only candidates to clear the BM25
            # pool. Verbatim catalog clauses are already high-confidence
            # lexical evidence, so they do not need a semantic contribution.
            # Generic browsing remains a light semantic hedge.
            dense_weight=0.0 if exact_phrases else program.dense_weight,
            # Verbatim requirement clauses already provide strong exact
            # lexical evidence, so morphology is a smaller hedge there.
            exact_phrase_mode=bool(exact_phrases),
            profile_weight=program.profile_weight,
        )

    def rerank(self, cands: list[Candidate], state: ConversationState) -> list[Candidate]:
        ranked = rerank_candidates(cands, state, self.table, self.metadata)
        program = compile_context_program(state)
        if program.lock_hard_constraints and program.hard_constraints:
            ranked = lock_hard_constraints(
                ranked, program.hard_constraints, self.table
            )
        return ranked


__all__ = ["Retriever"]
