# Phase 1 — Feasibility Verdict

Read `docs/plan/strategy.md` and `docs/plan/architecture.md` after finishing Phase 0 recon
(`docs/plan/RECON.md`), as instructed, so the plan wasn't read first. Verdicts below are judged
against what Phase 0 actually found, not against the plan's own assumptions.

**Bottom line: the plan is broadly workable.** The core architecture (leaf components, `agent.py`
as sole orchestrator, dataclass contracts, Null fallbacks, config-driven tunables) survives recon
intact — nothing in the evaluator or data contradicts it. Seven concrete facts change *emphasis
and detail*, not the shape of the system: no `brand` field, mostly-null `price`, no LLM keys, an
asymmetric failure contract, a simulator that only reads `ask_attribute` (never prose), no
cross-session user identity, and a `AgentResponse` contract that needs the literal `ask_attribute`
string, not a boolean.

---

## Component verdicts

| Component | Verdict | Recon finding that justifies it |
|---|---|---|
| Dual-track intent router (Buy vs Browse) | **VIABLE** | Nothing in the data blocks a Buy/Browse classifier over the utterance. `scenario_type` (the ground truth label) is never sent to the agent (`docs/agent_api_contract.json:4-23` — `reset_request` has no such field), so this has to be inferred either way, exactly as the plan assumes. Caveat: the plan's constraint-density signal ("brand, size, color, price bound...") should not lean on brand being a catalog-native slot — see Retrieval row below. |
| RRF multi-route retrieval (BM25 + dense + structured) | **MODIFIED** | BM25 (Route A) and dense+RRF (Route B, fusion) are fully viable: BM25 is already implemented and reproducible (`starter/agent.py`, `RECON.md` V4), and `sentence-transformers`/`faiss-cpu` have current wheels for this exact interpreter and a live Hugging Face download succeeded (`RECON.md` V2). **Route C (structured attribute lookup) needs to change what it filters on.** `categories` (breadcrumb list) is present and non-empty in 100% of 50,000 rows — the one fully reliable structured axis. `price` is `null` in 78.9% of rows and `brand` doesn't exist as a field at all (nested `details.Brand` covers only 4.7%). A hard filter on brand or price will empty most candidate pools; these must be soft-scored/best-effort, never hard `WHERE` filters, and the relaxation ladder the plan already specifies (strategy.md:22) is not optional polish here — it's load-bearing from day one. |
| Cross-encoder rerank | **VIABLE** (install unverified, availability verified) | PyPI has `torch-2.13.0-cp313-cp313-win_amd64.whl` and current `sentence-transformers`; a real Hugging Face Hub file download succeeded from this machine (`RECON.md` V2). The actual multi-hundred-MB install + a live cross-encoder forward pass has **not** been run yet — that's the first thing R2 should do in Phase 2, not an assumption to carry forward. |
| LLM listwise rerank | **MODIFIED — optional, config-gated, off by default** | `RECON.md` V3: zero API keys anywhere in this environment (checked env vars and the whole repo tree for `.env`/config files — none), and the spec confirms the organizer issues none and final judging network access "may" be disabled (`docs/submission_rules.md:59`). This is exactly the failure mode `strategy.md:225`'s own "Hackathon shortcut" already warns about — the difference is recon shows it's not a hedge, it's the current, literal state of this machine. Cross-encoder must be the unconditional primary path; LLM rerank must be reachable only behind a config flag with a real API key check at startup, defaulting to the Null/cross-encoder-only path. |
| Dialogue state machine + override detection | **VIABLE, and easier than planned** | `docs/competition_specification.md:36-37` and `evaluator/local_evaluator.py:256-264` confirm override sessions announce the switch as a literal structured turn, not something the agent has to detect from ambiguous prose: `"Actually, ignore my earlier preference. What I need is: {new_value}."` (verbatim, regenerated live in `RECON.md` D3 for three real samples). The plan's regex marker list (`actually|instead|no wait|never mind|forget|change of plans|different`, `strategy.md:241`) will catch this directly. One caution: `docs/competition_specification.md:40` notes the organizer may add paraphrasing later, so don't over-fit the regex to the exact public-set string — keep the general marker list as already planned. |
| Clarification policy | **MODIFIED — real lever, but only through `ask_attribute`, never through `message` text** | `evaluator/local_evaluator.py:166-185`: `customer_reply` takes `ask_attribute` as its only input from the agent's response and never reads `response["message"]`. This is *not* the plan's worst-case ("fixed script, clarification buys nothing," `strategy.md:5-7`) — clarification genuinely surfaces new information, because asking for the right attribute reveals a real, previously-undisclosed constraint. But it also isn't the plan's implicit best case either: no amount of well-crafted natural-language phrasing in `message` changes what happens next, because the simulator never parses it. All effort belongs in (a) choosing the right `ask_attribute` value (max information gain over the fixed 10-value enum) and (b) the turn-budget/stop-asking guard, not in question wording. Since recommendations score every turn (E2, confirmed), the plan's own "always attach your current best 10 alongside any clarification" shortcut (`strategy.md:255`) is validated and should be a hard rule, not a nice-to-have. |
| Memory distillation (session + long-term profile) | **MODIFIED — drop cross-session persistence, keep intra-session** | `strategy.md:79` proposes "persist to a dict keyed by user ID." **There is no user ID anywhere in the contract.** `reset_request` (`docs/agent_api_contract.json:4-23`) only carries `session_id` (a fresh random UUID minted by the harness every session, `local_evaluator.py:227`) and the anonymized aggregate `user_profile` — there is no field to key a persistent profile store by, and nothing in the protocol links two sessions to the same simulated shopper. The "long-term memory across sessions" half of Pillar III has no data plumbing to attach to and should be dropped. Intra-session distillation (compressing accumulated slots/rejections across one session's turns) is fully supported by the state the agent already owns privately and should be kept. Also worth noting for R5: of `user_profile`'s five fields, `purchase_frequency` is a constant (`"3-4 prior purchases"`) across all 200 public sessions — it carries zero discriminating signal on the dev set; `preference_tags` and `rating_style` are the fields with real variation to build a soft prior from. |

---

## Where the plan contradicts what recon found

Stated bluntly, as instructed:

1. **`brand` does not exist as a catalog field.** `strategy.md:15,19,150` and the `Candidate.meta`
   example (`strategy.md:117`) all assume brand is a queryable/filterable slot. It isn't — there
   is no top-level `brand` key in any of the 50,000 catalog rows. The nearest analogues are
   `details.Brand` (4.7% of rows) and `store` (99.4% of rows, but that's the seller/storefront
   name, not necessarily the manufacturer brand). Any document template, retrieval filter, or
   prompt field that says "brand" needs to read `store` (+ optionally `details.Brand` when
   present) instead, and hard-filtering on it is a bad idea given the sparsity.

2. **`price` as a hard filter will wipe out most candidate pools.** `price` is `null` in 78.9% of
   rows. The constraint-relaxation ladder the plan already specifies (`strategy.md:22`) has to be
   active for price essentially by default, not as a rare edge case.

3. **There's no field named `category_path`.** The catalog's structured category signal is
   `categories`, a breadcrumb *list* (e.g. `["Clothing, Shoes & Jewelry", "Women", "Jewelry",
   "Earrings", "Hoop"]`), never null or empty. `architecture.md:22` and `strategy.md:117` both
   write `category_path` as if it's a source field; it has to be built by joining `categories`,
   not read off the catalog directly. Minor, but worth fixing in the contract now rather than at
   3am.

4. **`AgentResponse.asked_clarification: bool` (`strategy.md:139-143`) loses information the real
   contract requires.** The actual `turn_response` schema (`docs/agent_api_contract.json:35-68`)
   needs the literal `ask_attribute` string — one of `category, material, color, size, style,
   brand, budget, feature, use_case, other`, or `null` — not a boolean. A bool can't be turned back
   into the value `customer_reply` needs to match against. This must be fixed in `contracts.py`
   before anyone starts building against it (see below).

5. **"Persist to a dict keyed by user ID" (`strategy.md:79`) is not buildable against this
   contract.** There is no user ID field anywhere, ever — see the Memory row above.

6. **The plan's opening framing (`strategy.md:5-7`) poses E1 as a binary: fixed script vs. "a
   user simulator that responds to your prompts."** The actual answer is a third case the framing
   doesn't name: a simulator that responds only to the *structured* `ask_attribute` field and is
   completely blind to the free-text `message` — i.e., prompts don't matter, only enum choices do.
   This isn't a contradiction of the plan's conclusion (clarification is real, Pillar II matters,
   as the plan hoped) but it does redirect *where* Pillar II effort should go — natural-language
   question quality is provably worth zero points; `ask_attribute` selection is worth everything.

7. **The failure contract is more asymmetric than `architecture.md:61` describes.** The plan says
   "a degraded answer scores; an exception scores zero and can poison the rest of the session."
   Empirically (`RECON.md` E4, live-tested): a `respond()` exception costs exactly one turn — the
   session keeps going with an empty-recommendation fallback for that turn only. But a `reset()`
   exception is **not caught anywhere in the harness** and crashes the *entire* evaluation run —
   every session in the batch, not just the current one. `reset()` needs to be treated as the
   single most safety-critical function in the whole codebase, more so than the plan currently
   emphasizes.

8. **The "session read-along" task (`architecture.md:35-37`, `strategy.md:206`) will find nothing
   to read.** Both docs assume slot vocabulary and override phrasing can be hand-extracted by
   reading `data/public_set.jsonl` sessions. The shipped session file contains **no conversation
   turns, no intent card, no override text, and no opening utterance** — verified: 0/200 rows have
   an `intent_card` or `behavior` key. All of that is generated at evaluation time from the
   *target catalog product's own fields*, deterministically seeded by `sample_id` +
   `scenario_type` (`local_evaluator.py:52-87, 204-213`). The only real per-session signal in the
   static file is `user_profile.preference_tags`. The task needs to be redirected: either read
   `local_evaluator.py`'s generation logic directly (what Phase 0 did), or actually run sessions
   through the evaluator and log what the simulator says, rather than hand-reading the JSONL file.

None of E2 (scored every turn) or E3 (the composite formula) contradict the plan — both are
exactly as `strategy.md`/`architecture.md` hoped, and both are now confirmed with real numbers
rather than assumed.

---

## Recommended `contracts.py`

Key names below are drawn from D1 (real catalog fields) and D2/D3 (real session/user_profile
fields and the fixed `ask_attribute` enum from the contract), not from the plan's examples.

```python
from __future__ import annotations
from dataclasses import dataclass, field

ASK_ATTRIBUTES = (
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case", "other",
)  # docs/agent_api_contract.json:42; None is also valid ("don't ask this turn")


@dataclass
class ProductMeta:
    """Mirrors the real catalog schema (data/catalog.jsonl). No `brand` field exists upstream —
    `store` is the closest near-universal proxy (99.4% populated); `details_brand` is a sparse
    (4.7%) enrichment, not a reliable filter key."""
    title: str
    price: float | None          # null in 78.9% of rows — never hard-filter without a fallback
    categories: list[str]        # breadcrumb path, e.g. ["Clothing, Shoes & Jewelry", "Women", ...]; always present
    features: list[str]          # empty in 10.4% of rows
    description: list[str]       # empty in 47.8% of rows
    store: str | None            # brand-proxy; null in 0.6% of rows
    details_brand: str | None    # from details.Brand; present in only 4.7% of rows
    average_rating: float
    rating_number: int


@dataclass
class Candidate:
    parent_asin: str             # the only scored ID field (docs/competition_specification.md:17)
    score: float
    route: str                   # "bm25" | "dense" | "category"
    meta: ProductMeta


@dataclass
class RetrievalRequest:
    canonical_query: str
    intent: str                  # "buy" | "browse" — inferred, never sent by the harness
    hard_filters: dict           # apply only to fields that are reliably populated (categories); price/brand as soft
    soft_prefs: dict
    top_k: int


@dataclass
class SessionState:
    turn: int
    intent: str
    slots: dict                  # keyed by the ASK_ATTRIBUTES vocabulary, not free-form names
    slot_turn_added: dict
    asked_attributes: list[str]  # ask_attribute values already used this session — avoid re-asking
    negatives: list
    canonical_query: str
    history: list
    profile: dict                 # derived once from user_profile at reset(); no cross-session persistence (no user ID exists)


@dataclass
class AgentResponse:
    recommendations: list[str]        # ordered parent_asin values, always attached even when asking a question
    message: str                      # customer-facing text; NOT read by the simulator's scoring logic
    ask_attribute: str | None         # must be a literal ASK_ATTRIBUTES value or None — not a bool (see contradiction #4)
```

---

## Revised Phase-0 / Phase-1 task list

The plan's own Phase 0/1 task tables (`strategy.md:198-227`) stay structurally intact. Changes:

**Phase 0 (Contract Freeze):**
- R4: catalog isn't in this fork's GitHub releases — it's on the **upstream** repo
  (`TechJam2026/techjam-conversational-search`, tag `participant-kit`). Verify SHA256 against the
  release's own `SHA256SUMS` asset before unpacking (already done once this session — script it).
  Use `py`, not `python3` — the latter isn't on PATH here.
- R4: add a `reset()`-hardening smoke test to the very first commit, not later — inject a raising
  `reset()` and confirm the harness crashes today so the whole team feels why this matters before
  building `agent.py`'s try/except wrapper.
- R3: redirect the "hand-read 30 dev sessions" task — read `local_evaluator.py`'s
  `intent_card`/`behavior_for`/`customer_reply` functions (or run 15-20 samples through the
  evaluator with a logging agent) instead of `data/public_set.jsonl` directly, which contains no
  conversational content at all.
- R1: document template should read `store` where the plan says `brand`, and build
  `category_path` by joining the `categories` list rather than expecting a source field of that
  name.
- R2: budget time in this phase specifically to *install* (not just confirm availability of)
  `sentence-transformers`/`faiss-cpu`/a cross-encoder and run one real forward pass — Phase 0
  recon confirmed downloadability, not a working install.
- All: freeze `contracts.py` using the version above — in particular, `ask_attribute: str | None`
  on `AgentResponse`, not a boolean.

**Phase 1 (Vertical Slice):**
- R1: Route C (structured) filters on `categories` only as a hard constraint; treat `price` and
  brand-like signals as soft-scored, with the relaxation ladder active from the first commit, not
  bolted on later.
- R2: since no API key exists anywhere on this machine right now, build and ship the cross-encoder
  path as the complete, working primary ranker before writing a single line of LLM-reranker code;
  gate the LLM path behind a config flag plus a startup key-presence check that falls back to
  Null/cross-encoder-only when absent.
- R3: clarification-policy work in this phase should optimize the `ask_attribute` selector, not
  question wording — the simulator never reads `message`, confirmed live in Phase 0.
- R4: always attach the current best 10 recommendations alongside any clarification question,
  as a hard rule from the first working version of `agent.py`, since every turn is scored (E2).

---

## Gate

Not re-litigated here — Phase 1 is analysis only, no code. Proceeding to Phase 2 requires explicit
approval, as instructed.
