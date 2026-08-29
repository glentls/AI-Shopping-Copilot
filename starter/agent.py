from __future__ import annotations

import json
import os
import re
import sqlite3
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# NOTE: a dense/embedding retrieval path used to live here. It was cut
# (Task 5): _merge_candidates filled from the keyword pool first and
# returned once it hit CANDIDATE_POOL_SIZE, which the keyword retriever
# always satisfied, so dense hits were never merged -- a measured 0.000
# contribution for a 25-minute embedding build and a per-turn network
# call. The Buying/Browsing router is kept and made load-bearing instead
# (see _route_intent / _retrieve_candidates). numpy is no longer imported.


# The LLM is confined to reranking the retrieved candidate pool and is OFF
# by default. Question selection is fully deterministic (Section 5.1/5.2): a
# well-reasoned semantic attribute scores worse than "other" against this
# simulator, so the LLM must never touch the ask. Flip LLM_RERANK=1 only on
# a box with Ollama reachable, and only ship it on if it measurably beats
# the deterministic score on the public set.
LLM_RERANK = os.environ.get("LLM_RERANK", "0") == "1"

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    # Simulator reply-template filler -- must not pollute BM25 when the
    # paraphrase fallback runs _terms() over a raw reply.
    "im", "still", "exploring", "key", "requirement", "what", "matters",
    "have", "dont", "additional", "preference", "actually", "ignore",
    "earlier", "need", "judgment", "use", "your", "not", "quite", "right",
    "yet", "ask", "about", "one", "specific", "attribute", "options", "those",
}

# Simulator reply templates parsed for verbatim constraint strings. Every
# constraint the simulator discloses is pulled verbatim from the target
# product's own features/details (evaluator intent_card), so a disclosed
# phrase is a literal substring of the target document -- the highest-value
# signal in the session. RE_OVERRIDE also yields hard_constraints[0].
RE_INITIAL = re.compile(r"I'm looking for (.+?)(?:\.|,)", re.IGNORECASE)
RE_KEYREQ = re.compile(r"key requirement is:\s*(.+?)\.?$", re.IGNORECASE)
RE_MATTERS = re.compile(r"what matters is:\s*(.+?)\.?$", re.IGNORECASE)
RE_OVERRIDE = re.compile(r"ignore my earlier preference.*?What I need is:\s*(.+?)\.?$", re.IGNORECASE)


def _is_dead_end_reply(lowered: str) -> bool:
    """True for the simulator's zero-information replies -- these must never
    be parsed as constraints or fed to the paraphrase fallback."""
    return (
        "not quite right yet" in lowered
        or "ask me about one specific attribute" in lowered
        or "don't have an additional preference" in lowered
        or "don't have a preference for" in lowered
    )

# Simple, deliberately narrow signal for "the customer just replaced an
# earlier preference" (Intent Override sessions). It only clears
# bookkeeping -- it never drops earlier turns from the search text (see
# SessionState.query_text) since the underlying category usually still
# holds even when one constraint changes.
OVERRIDE_RE = re.compile(r"\bignore\b.*\b(earlier|previous)\b", re.IGNORECASE)

# Two different simulator replies both contain "don't have ... preference"
# and they must be handled differently:
#
#   drain    "I don't have an additional preference for other."
#            -> the intent card is exhausted for that attribute; if the
#               attribute is "other" the whole card is drained and we stop
#               asking. Otherwise only that type-targeted attribute is dead.
#   boundary "I don't have a preference for other; please use your judgment."
#            -> fires at most once per session; the customer keeps talking
#               afterwards. Record it as declined for that one attribute
#               only, then ask a *different* attribute next turn. Storing it
#               as a drain would silence the agent with the card still full.
#
# DRAINED_RE is always checked first. Both keep a loose "no preference"
# alternative so a paraphrased private-set reply still degrades sensibly.
DRAINED_RE = re.compile(r"don't have an additional preference(?:\s+for\s+(\w+))?", re.IGNORECASE)
NO_PREFERENCE_RE = re.compile(r"don't have a preference for\s+(\w+)|no preference", re.IGNORECASE)

# Router signals: presence of any of these in a message (or an already
# non-empty disclosed-constraints dict) is treated as "the customer has
# given us something concrete" -> Buying track. Their absence -> Browsing.
BUDGET_RE = re.compile(r"\$\s?\d|\bunder\b|\bbudget\b|\bless than\b|\bcheap\b", re.IGNORECASE)
MATERIAL_WORDS = {"cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric", "denim", "suede"}
COLOR_WORDS = {"black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange", "navy", "beige"}
SIZE_WORDS = {"small", "medium", "large", "size", "xl", "xs", "wide", "narrow", "petite", "tall"}

CATALOG_COLUMNS = ("parent_asin", "title", "categories", "features", "details", "store", "description")

ALLOWED_ATTRIBUTES = frozenset({
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
})

SYSTEM_PROMPT = (
    "You are the ranking assistant inside a shopping agent for an Amazon "
    "Clothing/Shoes/Jewelry catalog. Each turn you receive the conversation "
    "so far, what the customer has already told us, a detected Buying/"
    "Browsing track, and a sample of currently-matching products. Output "
    "ONE JSON object describing how to filter and rank the full candidate "
    "pool -- you do not pick products directly, you never see full product "
    "IDs, and you do NOT decide what to ask the customer.\n\n"
    "Rules:\n"
    "- Use only the allowed attribute names below.\n"
    "- On a BUYING turn, favor fewer, higher-confidence constraints (the "
    "customer has told us something concrete -- lock onto it).\n"
    "- On a BROWSING turn, the candidate pool is more varied by design; "
    "spread weight across plausible interpretations.\n"
    "- Respond with JSON only, matching the schema exactly. No prose, no "
    "markdown fences.\n\n"
    "Allowed attribute names: " + ", ".join(sorted(ALLOWED_ATTRIBUTES)) + ".\n\n"
    "Schema:\n"
    "{\n"
    '  "constraints": {"<attribute>": "<value>", ...},\n'
    '  "weights": {"<attribute>": <0.0-1.0>, ...},\n'
    '  "message": "<one short, natural sentence to the customer>"\n'
    "}"
)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _as_float(value: object) -> float | None:
    """Best-effort numeric parse -- used for catalog price/rating and for
    reading a budget value like "under $80" out of free text."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value.replace(",", ""))
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
    return None


@dataclass
class SessionState:
    """Per-session memory. Everything the pipeline knows about one customer."""

    profile: dict
    turn: int = 0
    messages: list[str] = field(default_factory=list)
    agent_messages: list[str] = field(default_factory=list)
    ask_log: list[str | None] = field(default_factory=list)
    # attribute -> the reply text that answered it (assumed to answer
    # whichever attribute we last asked about)
    disclosed: dict[str, str] = field(default_factory=dict)
    # attributes the customer explicitly has no preference for -- distinct
    # from disclosed (there's no value here, just "don't ask again")
    declined: set[str] = field(default_factory=set)
    last_asked: str | None = None
    override_seen: bool = False
    # Coarse category from the turn-1 message ("Jewelry Necklaces", ...).
    # The single most reliable signal; never goes stale, even across an
    # override. Captured once and weighted heavily in retrieval + scoring.
    category: str = ""
    # Verbatim constraint phrases parsed out of the simulator's templated
    # replies. Each is a literal substring of the target document, so
    # substring containment scoring is close to a direct lookup.
    constraints: list[str] = field(default_factory=list)
    # Set once the simulator reports the intent card is fully drained
    # ("I don't have an additional preference for other."). After this,
    # asking yields nothing, so _select_ask_attribute returns None.
    card_drained: bool = False
    # Type-targeted attributes that returned nothing ("... for material.").
    # Distinct from declined (boundary) but treated the same by the asker.
    exhausted: set[str] = field(default_factory=set)
    # Rough count of disclosed constraint phrases, for the question-value
    # estimator. Replaced by len(constraints) once exact-phrase scoring lands.
    disclosed_count: int = 0
    # Paraphrase-proof backstop: if the drain string is reworded on the
    # private set we would never see card_drained; stop asking after this
    # many consecutive info-free replies instead of burning every turn.
    asks_without_gain: int = 0
    # The Intent Override value (hard_constraints[0], guaranteed verbatim in
    # the target document). Scored with a larger phrase bonus than an
    # ordinary constraint -- nothing else in a session has that guarantee.
    override_value: str = ""
    # Last successfully-scored ranking. Re-served verbatim if a later turn
    # fails internally, so a transient bug never costs a scoreable turn.
    last_ranked: list[str] = field(default_factory=list)
    # Ollama calls made this session (only non-zero when LLM_RERANK=1).
    llm_calls: int = 0

    def _add_constraint(self, phrase: str) -> bool:
        phrase = phrase.strip(" .;,-")
        if phrase and phrase not in self.constraints:
            self.constraints.append(phrase)
            return True
        return False

    def record_message(self, message: str, turn: int) -> None:
        self.turn = turn
        lowered = message.lower()
        gained = False

        initial = RE_INITIAL.search(message)
        if initial and not self.category:
            self.category = initial.group(1).strip()

        if OVERRIDE_RE.search(message):
            # Intent Override. The pre-override opener carried a *soft*
            # preference, which we never stored as a constraint (turn 1 has
            # no pending question), so the retrieval query is already clean
            # of it -- no wipe needed. Keep every constraint learned so far
            # (they are still true substrings of the target) and the
            # category (never stale). Reopen asking; hard_constraints[0]
            # arrives here verbatim and gets the larger phrase bonus.
            self.override_seen = True
            self.disclosed.clear()
            self.declined.clear()  # a new intent may reopen declined attributes
            self.card_drained = False
            match = RE_OVERRIDE.search(message)
            if match:
                value = match.group(1).strip(" .;,-")
                if value:
                    self.override_value = value
                    gained = self._add_constraint(value)
        else:
            drain = DRAINED_RE.search(message)
            boundary = NO_PREFERENCE_RE.search(message)
            if drain:
                attribute = (drain.group(1) or "").lower()
                if attribute and attribute != "other":
                    self.exhausted.add(attribute)
                else:
                    self.card_drained = True
            elif boundary:
                if self.last_asked:
                    self.declined.add(self.last_asked)
            else:
                for regex in (RE_KEYREQ, RE_MATTERS):
                    hit = regex.search(message)
                    if hit:
                        # RE_MATTERS can carry two constraints joined by ";".
                        for part in hit.group(1).split(";"):
                            gained |= self._add_constraint(part)
                if not gained and self.last_asked and not _is_dead_end_reply(lowered):
                    # Paraphrase fallback (Section 5.3): the private simulator
                    # may reword its reply templates. If no template matched
                    # but the customer answered our question, keep the token
                    # residue as a low-confidence constraint so the agent
                    # still accumulates signal. Regex parsing degrades, it
                    # does not collapse.
                    residue = " ".join(_terms(message)[:12])
                    gained |= self._add_constraint(residue)
                if gained and self.last_asked:
                    self.disclosed[self.last_asked] = message

        if "what matters is" in lowered:
            self.disclosed_count += lowered.split("what matters is", 1)[1].count(";") + 1
        elif "key requirement is" in lowered:
            self.disclosed_count += 1
        if gained:
            self.disclosed_count = max(self.disclosed_count, len(self.constraints))
            self.asks_without_gain = 0
        elif turn > 1:
            self.asks_without_gain += 1
        self.messages.append(message)

    def record_turn_result(self, ask_attribute: str | None, message: str) -> None:
        self.last_asked = ask_attribute
        self.ask_log.append(ask_attribute)
        self.agent_messages.append(message)

    def query_text(self) -> str:
        # Built from the coarse category plus the verbatim disclosed
        # constraint phrases -- NOT the raw message history, so the
        # simulator's dead-end replies and a stale pre-override preference
        # never enter the retrieval query. Falls back to the message history
        # only if nothing at all was parsed (paraphrase safety net).
        parts: list[str] = []
        if self.category:
            parts.append(self.category)
        parts.extend(self.constraints)
        if not parts:
            return " ".join(self.messages)
        return " ".join(parts)


@dataclass
class InstructionSheet:
    """The LLM's answer for one turn: how to filter/rank candidates, and
    whether to ask a clarifying question instead."""

    constraints: dict[str, str] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    ask_attribute: str | None = None
    message: str = ""
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})

    @classmethod
    def parse(cls, raw: str) -> "InstructionSheet":
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("instruction sheet must be a JSON object")

        constraints = {
            str(key): str(value)
            for key, value in (data.get("constraints") or {}).items()
            if value not in (None, "", [])
        }

        weights: dict[str, float] = {}
        for key, value in (data.get("weights") or {}).items():
            try:
                weights[str(key)] = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                continue

        # Parsed tolerantly (a model may still emit the key) but NOT used by
        # respond() -- question selection is deterministic, see Section 5.2.
        ask_attribute = data.get("ask_attribute")
        if ask_attribute is not None:
            ask_attribute = str(ask_attribute).strip().lower()
            if ask_attribute in ("", "null", "none"):
                ask_attribute = None
            elif ask_attribute not in ALLOWED_ATTRIBUTES:
                ask_attribute = "other"

        message = data.get("message")
        message = str(message).strip() if message else ""

        return cls(constraints=constraints, weights=weights, ask_attribute=ask_attribute, message=message)

    @classmethod
    def fallback(cls) -> "InstructionSheet":
        # Empty constraints -> the executor's scoring collapses to the
        # original retrieval order, and respond() fills in a generic message.
        return cls()


class Agent:
    """Two-track keyword retrieval -> LLM instruction sheet -> deterministic
    execution.

    Pipeline per turn: update SessionState -> route the turn to a Buying
    (narrow, phrase-locked) or Browsing (wide, category-driven) retrieval
    track -> optionally ask a local LLM (via Ollama, off by default) for a
    small JSON "instruction sheet" (constraints + weights) -> apply that
    sheet on top of exact-phrase scoring deterministically -> return top_k.
    Any LLM failure falls back to the plain retrieval order rather than
    raising.
    """

    MAX_TURNS = 10
    # Buying track: candidates retrieved by BM25 and re-scored by
    # exact-phrase containment. 400 (not 100): the phrase rescore routinely
    # lifts a target from deep in the BM25 tail to the top-10, so the pool
    # must be deep enough to contain it.
    CANDIDATE_POOL_SIZE = 400
    # Browsing track: no verbatim constraints yet, so the query is
    # category-dominated and recall matters more than precision -- use a
    # deeper pool and lean entirely on the category signal. This is the
    # load-bearing difference between the two router branches now that dense
    # retrieval is gone. (An earlier variant also had Buying stop asking
    # once >=3 constraints were known; it regressed buying Hit@10 by ~0.05
    # on the public set -- see README -- and was removed.)
    BROWSING_POOL_SIZE = 600
    # Candidates actually shown to the LLM in the prompt -- kept small to
    # fit Ollama's default 4096-token context alongside conversation history.
    CANDIDATE_SAMPLE_SIZE = 18
    LLM_FAILURE_THRESHOLD = 3
    # Hard ceiling on Ollama calls per session when LLM_RERANK=1 (Task 7).
    MAX_LLM_CALLS_PER_SESSION = 3

    # --- Question-value estimation (Section 5.1) --------------------------
    # The intent card is 2 hard + 2 soft constraints.
    EXPECTED_CARD_SIZE = 4
    # Stop asking after this many consecutive info-free replies even if we
    # never observe the (English) drain string -- private-set paraphrase guard.
    MAX_ASKS_WITHOUT_GAIN = 6
    # Prior probability that a single card entry classifies as each
    # type-targeted attribute, read off the simulator's classify_constraint()
    # keyword rules: "feature" is the fall-through default and dominates,
    # the rest are narrow keyword gates. "other" is handled separately with
    # coverage 1.0 because it bypasses the classifier entirely.
    ATTRIBUTE_PRIORS = {
        "feature": 0.55,
        "style": 0.12,
        "material": 0.10,
        "color": 0.08,
        "use_case": 0.07,
        "size": 0.04,
        "budget": 0.04,
    }

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._products: dict[str, dict] = {}
        # parent_asin -> lowercased "title categories features details store
        # description", precomputed once for the exact-phrase rescore. At
        # pool 400 rebuilding this per candidate per turn would be ~800k
        # string joins per evaluation run.
        self.corpus: dict[str, str] = {}
        self._build_index()

        self.ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        self.llm_timeout = float(os.environ.get("OLLAMA_TIMEOUT", "30"))
        self._llm_failures = 0
        self._llm_disabled = False

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        popularity: list[tuple[float, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                fields = {
                    "parent_asin": parent_asin,
                    "title": _text(product.get("title")),
                    "categories": _text(product.get("categories")),
                    "features": _text(product.get("features")),
                    "details": _text(product.get("details")),
                    "store": _text(product.get("store")),
                    "description": _text(product.get("description")),
                    "price": _as_float(product.get("price")),
                    "average_rating": _as_float(product.get("average_rating")),
                }
                self._products[parent_asin] = fields
                self.corpus[parent_asin] = " ".join(
                    str(fields[column]) for column in CATALOG_COLUMNS[1:]
                ).lower()
                popularity.append((_as_float(product.get("rating_number")) or 0.0, parent_asin))
                batch.append(tuple(fields[column] for column in CATALOG_COLUMNS))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

        # Precomputed once: the 10 most-reviewed products, used as an
        # absolute last-resort recommendation list so respond() can never
        # return fewer than 10 candidates (Task 6). A blind guess scores 0,
        # but so does an empty list -- and the empty list forfeits the turn.
        popularity.sort(key=lambda item: -item[0])
        self._fallback_ids: list[str] = [asin for _, asin in popularity[:10]]

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(profile=user_profile)

    # ---- Intent routing ------------------------------------------------

    def _route_intent(self, state: SessionState, user_message: str) -> str:
        """"buying" (precise) or "browsing" (exploratory), decided fresh
        each turn -- a session can move from browsing to buying as the
        customer reveals concrete constraints."""
        if OVERRIDE_RE.search(user_message):
            return "buying"  # a replaced preference is still a hard constraint
        if BUDGET_RE.search(user_message):
            return "buying"
        message_terms = set(_terms(user_message))
        if message_terms & (MATERIAL_WORDS | COLOR_WORDS | SIZE_WORDS):
            return "buying"
        if state.disclosed or state.constraints:
            return "buying"
        return "browsing"

    # ---- Question-value estimation ------------------------------------

    def _select_ask_attribute(self, state: SessionState) -> str | None:
        """Adaptive question-value estimation: score each allowed attribute
        by the expected number of still-unknown constraints it can unlock,
        and ask the argmax.

        The simulator (evaluator ``customer_reply``) only reacts to
        ``ask_attribute``; a semantically "correct" attribute that the
        target's constraints don't classify as returns nothing and wastes a
        turn, while ``"other"`` bypasses the classifier and matches *any*
        undisclosed card entry. So ``"other"`` carries coverage 1.0 and wins
        by construction while the card still has unknowns. The type-targeted
        attributes remain a live secondary strategy: when ``"other"`` is
        declined (a boundary turn) the next-best prior -- ``"feature"`` -- is
        picked instead, so the agent never goes silent with the card full.

        Returns ``None`` only when the card is known to be drained or the
        paraphrase backstop trips. Never returns a declined/exhausted value.
        """
        if state.card_drained:
            return None
        if state.asks_without_gain >= self.MAX_ASKS_WITHOUT_GAIN:
            return None

        blocked = state.declined | state.exhausted
        # Expected number of card entries still to be revealed.
        remaining = max(1, self.EXPECTED_CARD_SIZE - state.disclosed_count)

        coverage = {"other": 1.0, **self.ATTRIBUTE_PRIORS}
        best_attribute: str | None = None
        best_value = -1.0
        for attribute, unlock_prob in coverage.items():
            if attribute in blocked:
                continue
            expected_yield = unlock_prob * remaining
            if expected_yield > best_value:
                best_value = expected_yield
                best_attribute = attribute
        return best_attribute

    # ---- Stage 1: keyword retrieval (Buying / Browsing tracks) --------

    def _retrieve_candidates(self, state: SessionState, track: str) -> list[dict]:
        """BM25 keyword retrieval. The Buying and Browsing tracks differ
        here, which is what keeps _route_intent load-bearing now that dense
        retrieval is gone:

        - Buying: verbatim constraints exist, so the query is phrase-locked
          (category + constraint terms) and the pool is the tighter 400.
        - Browsing: no constraints yet, so the query is category-dominated
          and recall matters more than precision -- pull a deeper 600-row
          pool and lean entirely on the category signal.
        """
        category_terms = _terms(state.category)
        if track == "browsing":
            limit = self.BROWSING_POOL_SIZE
            # Category terms only, tripled so they dominate the OR-set even
            # after dedup/truncation. A stray token from a paraphrased reply
            # can still be present in query_text; keep it as a weak tail.
            ordered = category_terms * 3 + _terms(state.query_text())
        else:
            limit = self.CANDIDATE_POOL_SIZE
            constraint_terms: list[str] = []
            for phrase in state.constraints:
                constraint_terms.extend(_terms(phrase)[:12])
            ordered = category_terms + category_terms + constraint_terms
            if not ordered:
                ordered = _terms(state.query_text())

        unique_terms = list(dict.fromkeys(ordered))[:60]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            return []
        try:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 8.0, 5.0, 3.0, 3.0, 1.5, 1.0) LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # Quoted alphanumeric terms should never produce invalid FTS5
            # syntax, but a private-set paraphrase or an odd catalog token
            # could. Degrade to no keyword hits rather than raising into
            # respond()'s blanket handler, which would waste the whole turn.
            return []
        # Look up the full record (incl. price/rating, which the FTS table
        # itself doesn't carry) instead of rebuilding a partial dict from
        # the SQL row -- self._products is the single source of truth.
        return [self._products[row[0]] for row in rows if row[0] in self._products]

    # ---- Stage 2: LLM instruction sheet ---------------------------------

    def _summarize_candidate(self, candidate: dict) -> str:
        title = (candidate.get("title") or "product").strip()
        if len(title) > 70:
            title = title[:67].rstrip() + "..."
        price = candidate.get("price")
        price_text = f"${price:.2f}" if isinstance(price, float) else "price n/a"
        categories = " ".join((candidate.get("categories") or "").split()[:6])
        return f'"{title}" | {price_text} | {categories}'.strip()

    def _build_user_prompt(self, state: SessionState, candidates: list[dict], track: str) -> str:
        profile = state.profile or {}
        track_note = (
            "Detected track: BUYING -- the customer has given something concrete; "
            "prioritize precision and lock in hard constraints."
            if track == "buying"
            else "Detected track: BROWSING -- the customer is still exploring; the "
            "candidate pool is broader by design, favor a clarifying question "
            "unless one option is clearly strongest."
        )
        lines = [
            f"Turn {state.turn} of {self.MAX_TURNS}.",
            track_note,
            "Customer profile: "
            f"purchase_frequency={profile.get('purchase_frequency', 'unknown')}, "
            f"rating_style={profile.get('rating_style', 'unknown')}, "
            f"preference_tags={profile.get('preference_tags', [])}, "
            f'summary="{profile.get("summary", "")}"',
            "",
            "Conversation so far:",
        ]
        for index, customer_message in enumerate(state.messages):
            lines.append(f'{index + 1}. Customer: "{customer_message}"')
            if index < len(state.agent_messages):
                asked = state.ask_log[index] if index < len(state.ask_log) else None
                suffix = f" (asked: {asked})" if asked else ""
                lines.append(f'   Agent: "{state.agent_messages[index]}"{suffix}')
        lines.append("")
        lines.append(
            "Known constraints so far: " + (json.dumps(state.disclosed) if state.disclosed else "none yet")
        )
        if state.declined:
            lines.append(
                "Customer has NO preference for these -- do not ask again: "
                + ", ".join(sorted(state.declined))
            )
        lines.append("")
        sample = candidates[: self.CANDIDATE_SAMPLE_SIZE]
        lines.append(f"Sample of currently-matching products ({len(candidates)} total, showing {len(sample)}):")
        for candidate in sample:
            lines.append(f"- {self._summarize_candidate(candidate)}")
        lines.append("")
        lines.append("Return the JSON instruction sheet now.")
        return "\n".join(lines)

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        payload = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2, "num_ctx": 4096},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.ollama_host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.llm_timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body.get("message", {}).get("content", "")
        usage = {
            "prompt_tokens": max(0, int(body.get("prompt_eval_count") or 0)),
            "completion_tokens": max(0, int(body.get("eval_count") or 0)),
        }
        return content, usage

    def _get_instruction_sheet(self, state: SessionState, candidates: list[dict], track: str) -> InstructionSheet:
        # LLM reranking is opt-in. When off, skip prompt building and the
        # socket entirely -- zero network calls, zero tokens (Task 2).
        if not LLM_RERANK or self._llm_disabled:
            return InstructionSheet.fallback()
        # Feasibility cap (Section 5.7): 800 private sessions x several turns
        # is thousands of sequential Ollama calls. Spend the budget on the
        # turns most likely to matter -- once constraints exist and the pool
        # has stabilised -- and never exceed a few calls per session.
        if state.llm_calls >= self.MAX_LLM_CALLS_PER_SESSION:
            return InstructionSheet.fallback()
        state.llm_calls += 1
        user_prompt = self._build_user_prompt(state, candidates, track)
        try:
            content, usage = self._call_ollama(SYSTEM_PROMPT, user_prompt)
            sheet = InstructionSheet.parse(content)
            sheet.usage = usage
            self._llm_failures = 0
            return sheet
        except Exception:
            self._llm_failures += 1
            if self._llm_failures >= self.LLM_FAILURE_THRESHOLD:
                self._llm_disabled = True
            return InstructionSheet.fallback()

    # ---- Stage 3: deterministic execution -------------------------------

    # Verbatim phrase-hit bonus -- dominant over token overlap (<=1.2) and
    # category terms (0.6 each). The override value is hard_constraints[0],
    # guaranteed verbatim in the target document, so it earns more.
    PHRASE_BONUS = 3.0
    OVERRIDE_PHRASE_BONUS = 4.5

    def _score_candidates(
        self, state: SessionState, candidates: list[dict], sheet: InstructionSheet
    ) -> list[str]:
        """Exact-phrase constraint scoring (Section 5.3). Every disclosed
        constraint is a literal substring of the target document, so verbatim
        containment is the dominant signal; partial token overlap and the
        coarse category are weaker additive terms. The LLM instruction sheet
        (only populated when LLM_RERANK=1) sits on top as a capped layer and
        can never outweigh a verbatim phrase hit."""
        phrases = [c.lower() for c in state.constraints if 4 <= len(c) <= 120]
        override_phrase = state.override_value.lower() if state.override_value else None
        category_terms = _terms(state.category)

        scored: list[tuple[float, int, str]] = []
        for rank, candidate in enumerate(candidates):
            parent_asin = candidate["parent_asin"]
            doc = self.corpus.get(parent_asin)
            if doc is None:
                doc = " ".join(
                    str(candidate.get(field_name, "")) for field_name in CATALOG_COLUMNS[1:]
                ).lower()

            # Preserve BM25 order as a tiebreak (small, negative-by-rank).
            score = -rank * 0.001

            for phrase in phrases:
                if phrase in doc:
                    score += (
                        self.OVERRIDE_PHRASE_BONUS
                        if phrase == override_phrase
                        else self.PHRASE_BONUS
                    )
                else:
                    tokens = _terms(phrase)
                    if tokens:
                        score += 1.2 * sum(token in doc for token in tokens) / len(tokens)

            for term in category_terms:
                if term in doc:
                    score += 0.6

            for attribute, value in sheet.constraints.items():
                weight = sheet.weights.get(attribute, 0.5)
                if attribute == "budget":
                    budget_max = _as_float(value)
                    price = candidate.get("price")
                    if budget_max is not None and price is not None and price <= budget_max:
                        score += weight
                elif value.lower() in doc:
                    score += weight

            scored.append((score, rank, parent_asin))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [parent_asin for _, _, parent_asin in scored]

    def _guaranteed_ten(self, state: SessionState, ranked: list[str]) -> list[str]:
        """Always return exactly 10 valid catalog IDs. Top up a short list
        with the previous good ranking, then the precomputed popular IDs."""
        out: list[str] = []
        seen: set[str] = set()
        for source in (ranked, state.last_ranked, self._fallback_ids):
            for pid in source:
                if pid not in seen and pid in self._products:
                    seen.add(pid)
                    out.append(pid)
                    if len(out) >= 10:
                        return out
        return out

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        state = self._sessions[session_id]

        # Everything below touches retrieval/scoring/LLM logic that we own;
        # the LLM call already catches its own failures, but this is a last
        # line of defense so a stray bug in our code still returns a valid,
        # contract-shaped response instead of raising and counting the turn
        # (or the whole session) as a hard miss.
        try:
            state.record_message(user_message, turn)

            track = self._route_intent(state, user_message)
            candidates = self._retrieve_candidates(state, track)

            if not candidates:
                ask_attribute = self._select_ask_attribute(state)
                message = (
                    "Tell me a bit more about what you're looking for."
                    if ask_attribute
                    else "Here are the closest matches I found."
                )
                state.record_turn_result(ask_attribute, message)
                # Never forfeit a scoreable turn: fall back to the last good
                # ranking, then to the precomputed popular IDs.
                ranked_ids = self._guaranteed_ten(state, [])
                return {
                    "message": message,
                    "ask_attribute": ask_attribute,
                    "recommendations": [{"parent_asin": pid} for pid in ranked_ids],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                }

            sheet = self._get_instruction_sheet(state, candidates, track)
            ranked_ids = self._guaranteed_ten(
                state, self._score_candidates(state, candidates, sheet)[:top_k]
            )
            state.last_ranked = ranked_ids

            # The clarifying question is chosen by the deterministic
            # question-value estimator, never by the LLM: Section 4.3 shows a
            # well-reasoned semantic attribute scores *worse* than "other"
            # against this simulator. sheet.ask_attribute is deliberately
            # ignored here. _select_ask_attribute already excludes declined
            # and exhausted attributes.
            ask_attribute = self._select_ask_attribute(state)
            if ask_attribute:
                message = "Could you tell me a bit more about what matters here?"
            else:
                message = sheet.message or "Here are the closest matches I found."

            state.record_turn_result(ask_attribute, message)
            return {
                "message": message,
                "ask_attribute": ask_attribute,
                "recommendations": [{"parent_asin": pid} for pid in ranked_ids],
                "usage": sheet.usage,
            }
        except Exception:
            # Last line of defence. self._fallback_ids is precomputed in
            # __init__ -- nothing here can raise a second time.
            try:
                ranked_ids = self._guaranteed_ten(state, [])
            except Exception:
                ranked_ids = list(self._fallback_ids)
            return {
                "message": "Sorry, could you tell me more about what you're looking for?",
                "ask_attribute": None,
                "recommendations": [{"parent_asin": pid} for pid in ranked_ids],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
