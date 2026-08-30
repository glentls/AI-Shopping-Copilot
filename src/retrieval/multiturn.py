"""Phase 7 -- multi-turn retrieval context.

The evaluator never replays prior turns to the agent, and once a real dialog component is in
place each turn-t user message is only the *newly disclosed* constraint ("For that, what
matters is: cotton; relaxed fit.") with no category and no earlier context. Retrieving on that
alone throws away everything the customer already said. So retrieval keeps its own per-session
memory (contract: `RetrievalRequest.session_id` "keys its own per-session state") and rebuilds
an effective query from the whole conversation each turn.

Everything here is behind `config.retrieval.multiturn.enabled` AND gated on
`RetrievalRequest.turn >= 1` and a non-empty `session_id`. Under today's wiring agent.py sends
`turn=0` / `session_id=""`, so `build_effective_query()` returns `request.canonical_query`
untouched and `search()` is byte-for-byte its pre-Phase-7 self. Turning it on is R4's
integration step (see docs/plan/r1_contract_change.md); the deltas are measured offline in
`eval/recall_probe.py --multiturn`.

Nothing here imports another component.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from src.contracts import ProductMeta, RetrievalRequest

from .bm25 import STOPWORDS, _terms


@dataclass
class SessionMemory:
    """`session_id -> [(turn, message)]`, oldest first. Single-writer: agent.py calls search()
    from a one-worker executor and the evaluator runs sessions serially, so a plain dict is
    safe. Bounded by LRU eviction so a long private run can't grow it without limit."""

    _turns: dict[str, list[tuple[int, str]]] = field(default_factory=dict)
    _lru: list[str] = field(default_factory=list)
    max_sessions: int = 4000

    def observe(self, session_id: str, turn: int, message: str, intent_changed: bool = False) -> None:
        if not session_id or turn < 1:
            return
        if turn <= 1 or intent_changed:
            self._turns[session_id] = []  # new session, or an override wipes the stale context
        history = self._turns.setdefault(session_id, [])
        history[:] = [(t, m) for t, m in history if t != turn]
        history.append((turn, message))
        if session_id in self._lru:
            self._lru.remove(session_id)
        self._lru.append(session_id)
        while len(self._lru) > self.max_sessions:
            self._turns.pop(self._lru.pop(0), None)

    def turns(self, session_id: str) -> list[tuple[int, str]]:
        return list(self._turns.get(session_id, []))


def accumulate_query(
    turn_texts: list[tuple[int, str]],
    config: dict,
    stopwords: frozenset[str] = STOPWORDS,
) -> str:
    """De-duplicated bag of terms across every turn, **most-recent turn first** so that when the
    `max_query_terms` cap bites it is the oldest terms that fall off (recency weighting, as far
    as a flat FTS5 OR-query can express it). A turn that carries only conversational scaffolding
    -- NullDialog's "Those options are not quite right yet" -- contributes no terms and drops
    out on its own."""
    retrieval_cfg = config["retrieval"]
    max_terms = int(retrieval_cfg.get("max_query_terms", 40))
    window = int(retrieval_cfg.get("multiturn", {}).get("window_turns", 0))

    ordered = sorted(turn_texts, key=lambda pair: pair[0], reverse=True)
    if window > 0:
        ordered = ordered[:window]

    seen: set[str] = set()
    out: list[str] = []
    for _, message in ordered:
        for term in _terms(message, stopwords):
            if term not in seen:
                seen.add(term)
                out.append(term)
                if len(out) >= max_terms:
                    return " ".join(out)
    return " ".join(out)


def blend_profile(query: str, profile: dict, config: dict) -> str:
    """Append `user_profile.preference_tags` to the query. Phase 1 measured this in isolation:
    +5pt overall Recall@100, driven entirely by buying (+12pt), with browsing slightly down --
    a real trade-off, hence its own flag."""
    if not config["retrieval"].get("multiturn", {}).get("profile_blend", False):
        return query
    if not isinstance(profile, dict):
        return query
    tags = [str(t).strip() for t in (profile.get("preference_tags") or []) if str(t).strip()]
    if not tags:
        return query
    return f"{query} {' '.join(tags)}".strip()


def rocchio_terms(
    accepted: list[ProductMeta],
    negatives: list[ProductMeta],
    config: dict,
    stopwords: frozenset[str] = STOPWORDS,
) -> tuple[list[str], list[str]]:
    """Lexical pseudo-relevance feedback (Rocchio, text form). Returns `(add_terms, avoid_terms)`:

      add_terms   -- most frequent non-stopword terms in the title+features of items the user
                     reacted to positively (Rocchio beta). Appended to the query.
      avoid_terms -- terms frequent in rejected items and absent from accepted ones (gamma).
                     Returned for a caller that wants to down-weight; not wired into scoring
                     here (see the module note -- no feedback signal exists in this eval).

    Unmeasurable on the public/dev sets (the evaluator provides no accept/reject feedback), so
    this is unit-tested on fabricated metas and reported only at the oracle ceiling.
    """
    mt = config["retrieval"].get("multiturn", {})
    top_add = int(mt.get("rocchio_add_terms", 6))
    top_avoid = int(mt.get("rocchio_avoid_terms", 6))

    def bag(metas: list[ProductMeta]) -> Counter:
        counter: Counter = Counter()
        for meta in metas:
            text = " ".join([meta.title or "", *(meta.features or [])])
            counter.update(_terms(text, stopwords))
        return counter

    pos, neg = bag(accepted), bag(negatives)
    add_terms = [term for term, _ in pos.most_common(top_add)]
    avoid_terms = [term for term, _ in neg.most_common() if term not in pos][:top_avoid]
    return add_terms, avoid_terms


def build_effective_query(
    request: RetrievalRequest,
    memory: SessionMemory,
    config: dict,
    stopwords: frozenset[str] = STOPWORDS,
    meta_of: dict[str, ProductMeta] | None = None,
) -> str:
    """The Phase-7 orchestrator, called at the top of `retriever.search()`. Records this turn,
    then rebuilds the query from the whole session + profile + positive feedback. Returns
    `request.canonical_query` unchanged whenever the feature is off or the request carries no
    turn / session id -- the inert path."""
    mt = config["retrieval"].get("multiturn", {})
    if not mt.get("enabled", False) or request.turn < 1 or not request.session_id:
        return request.canonical_query

    memory.observe(request.session_id, request.turn, request.canonical_query, request.intent_changed)
    history = memory.turns(request.session_id)

    query = request.canonical_query if len(history) <= 1 else accumulate_query(history, config, stopwords)
    query = blend_profile(query, request.profile, config)

    if mt.get("rocchio", False) and request.accepted and meta_of:
        accepted_metas = [meta_of[a] for a in request.accepted if a in meta_of]
        negative_metas = [meta_of[a] for a in request.negatives if a in meta_of]
        add_terms, _ = rocchio_terms(accepted_metas, negative_metas, config, stopwords)
        if add_terms:
            query = f"{query} {' '.join(add_terms)}".strip()

    return query or request.canonical_query
