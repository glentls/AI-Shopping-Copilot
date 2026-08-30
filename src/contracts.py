"""Every cross-component dataclass lives here and nowhere else.

Frozen per docs/plan/architecture.md's rule: components are leaves, only agent.py imports the
whole graph, and every component is a pure function over these dataclasses. Changes to this file
require unanimous team agreement (see CLAUDE.md).

Field names are drawn from the real catalog schema and the real agent_api_contract.json, not from
guesses -- see docs/plan/RECON.md (D1/D2/D3) and docs/plan/FEASIBILITY.md for the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The exact ask_attribute enum the harness accepts (docs/agent_api_contract.json:42).
# None is also a valid value ("don't ask this turn").
ASK_ATTRIBUTES: tuple[str, ...] = (
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case", "other",
)


@dataclass(frozen=True)
class ProductMeta:
    """Mirrors data/catalog.jsonl. There is no top-level `brand` field in the real catalog
    (docs/plan/RECON.md D1) -- `store` is the closest near-universal proxy (99.4% populated);
    `details_brand` is a sparse (4.7%) enrichment and must never be relied on as a filter key."""

    title: str
    price: float | None
    categories: list[str]
    features: list[str]
    description: list[str]
    store: str | None
    details_brand: str | None
    average_rating: float
    rating_number: int


@dataclass(frozen=True)
class Candidate:
    parent_asin: str  # the only field the evaluator scores (docs/competition_specification.md:17)
    score: float
    route: str  # e.g. "bm25" | "dense" | "category" | "fallback"
    meta: ProductMeta


class RetrievalResult(list):
    """Retrieval's return type: a ``list[Candidate]`` that also carries the relaxation-ladder
    diagnostics R3's over-generality clarification trigger consumes.

    It is a ``list`` subclass rather than a frozen dataclass on purpose. ``search()`` used to
    return a bare ``list[Candidate]``; keeping that true means every existing consumer
    (``ranking``, ``agent._ensure_top_k``, the fallback path) works unchanged and the skeleton
    keeps scoring identically with no edit to ``agent.py``. Consumers that want the diagnostics
    read ``.pool_size`` / ``.dropped_constraints``; consumers that don't are unaffected.

    ``agent.py`` forwards these onto ``SessionState.retrieval_pool_size`` /
    ``SessionState.dropped_constraints`` when it integrates the richer retrieval path, which is
    how R3 receives the signal without importing retrieval. See docs/plan/r1_contract_change.md.
    """

    pool_size: int
    dropped_constraints: list

    def __init__(self, candidates=(), *, pool_size: int = 0, dropped_constraints=None) -> None:
        super().__init__(candidates)
        self.pool_size = int(pool_size)
        self.dropped_constraints = list(dropped_constraints or [])


@dataclass(frozen=True)
class RetrievalRequest:
    canonical_query: str
    intent: str  # "buy" | "browse" -- never sent by the harness, always inferred
    hard_filters: dict
    soft_prefs: dict
    top_k: int
    # --- Multi-turn retrieval inputs (all optional; default values reproduce the old
    # single-shot behaviour exactly, so existing callers are unaffected). agent.py fills
    # these from SessionState when it wires the conversational retrieval path; R1's
    # dense/fusion/feedback code reads them. See docs/plan/r1_contract_change.md. ---
    session_id: str = ""          # stable per session -> retrieval can keep its own per-session state
    turn: int = 0                 # 1-based turn index; 0 means "not supplied"
    negatives: list = field(default_factory=list)  # parent_asins ruled out so far (accumulates across turns)
    accepted: list = field(default_factory=list)    # parent_asins reacted to positively (usually empty in this eval)
    intent_changed: bool = False  # R3 signalled an intent override this turn -> retrieval hard-resets its query vector
    profile: dict = field(default_factory=dict)     # user_profile, for profile blending from turn 1


@dataclass
class SessionState:
    """Owned entirely by the Agent instance, keyed by session_id. The harness never replays
    prior turns back to respond() (docs/plan/RECON.md E5), so everything the agent needs to
    remember about a session lives here."""

    session_id: str
    turn: int
    intent: str
    slots: dict
    slot_turn_added: dict
    asked_attributes: list[str] = field(default_factory=list)
    negatives: list = field(default_factory=list)
    canonical_query: str = ""
    history: list = field(default_factory=list)
    profile: dict = field(default_factory=dict)
    # Populated by agent.py from the last RetrievalResult, so R3's dialog component can read
    # the current candidate-pool size and which constraints the relaxation ladder had to drop
    # (its over-generality "ask a narrowing question" trigger) without importing retrieval.
    retrieval_pool_size: int = 0
    dropped_constraints: list = field(default_factory=list)


@dataclass(frozen=True)
class DialogResult:
    canonical_query: str
    ask_attribute: str | None
    slots: dict
    message: str
    # R3 fills these; NullDialog leaves the defaults. agent.py uses `intent` to set
    # SessionState.intent and passes `intent_override` through to
    # RetrievalRequest.intent_changed so retrieval can hard-reset its accumulated query
    # vector on an override turn. "" / False mean "no signal", identical to today's behaviour.
    intent: str = ""              # "buy" | "browse" | "" (unknown)
    intent_override: bool = False  # this turn's message overrides the earlier stated intent


@dataclass(frozen=True)
class MemoryProfile:
    boosts: dict  # e.g. {"store": {name: weight}} -- empty for NullMemory
    summary: str


@dataclass(frozen=True)
class AgentResponse:
    recommendations: list[str]  # ordered parent_asin values
    message: str
    ask_attribute: str | None  # a literal ASK_ATTRIBUTES value or None -- never a bool
    usage: dict
