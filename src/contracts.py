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


@dataclass(frozen=True)
class RetrievalRequest:
    canonical_query: str
    intent: str  # "buy" | "browse" -- never sent by the harness, always inferred
    hard_filters: dict
    soft_prefs: dict
    top_k: int


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


@dataclass(frozen=True)
class DialogResult:
    canonical_query: str
    ask_attribute: str | None
    slots: dict
    message: str


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
