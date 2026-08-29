# Strategy: Concepts, Team Breakdown, and 72-Hour Execution Plan

> **Before anything else — one instruction that outranks everything below.**
>
> In your first 3 hours, read the evaluator's session-replay loop and answer this question: **are the user's turns a fixed script, or are they generated in response to your agent's questions?**
>
> If the 200 dev sessions contain a pre-recorded list of user utterances that get fed to your agent regardless of what it says, then clarification questions buy you nothing and cost you MTTC directly. The optimal policy becomes "extract everything from turn 1, rank, and never ask." If instead there's a user simulator that responds to your prompts, clarification is a real lever and Pillar II matters enormously. These two worlds imply completely different systems. Most teams will build the second and get scored in the first. **Find out on day one.**

---

## PART 1: Concept Decoding & Research Grounding

### 1. Dual-Track Intent Routing (Buying vs. Browsing)

**Plain English:** A cheap classifier sits in front of your retrieval stack and decides whether the user knows what they want. If they do ("black leather Chelsea boots, size 10, under $150"), you run a precision pipeline: parse hard constraints into structured filters, apply them as a strict `WHERE` clause, and return a narrow set. If they don't ("something for my sister's beach wedding"), you run a recall pipeline: no filters, pure dense retrieval over a scenario-level embedding, deliberately diversified across categories so a dress, a clutch, and sandals can all surface.

**Mechanically:**

- Score the utterance on **constraint density** — count of extractable hard slots (brand, size, color, price bound, specific product noun) vs. scenario markers ("for a", "ideas", "something", "gift", "I'm going to").
- Threshold or classify → `BUY | BROWSE`.
- Route selects three things: which retrievers fire, whether filters are hard (drop non-matching) or soft (score boost), and how many candidates go to the ranker (buy: 20 tight; browse: 100 diverse).
- **The failure mode to guard:** a hard filter with zero survivors. Always implement constraint relaxation — drop the least-confident slot and re-run rather than returning an empty list.

**Papers:**

- Rose & Levinson, *Understanding User Goals in Web Search* (WWW 2004) — the original taxonomy that every intent router is a descendant of. Skim the taxonomy table, ignore the rest.
- Reddy et al., *Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search* (2022) — Amazon's own framing of query→product relevance grades. Directly relevant to your dataset's domain.

---

### 2. Multi-Route Retrieval → LLM Semantic Ranking

**Plain English:** Two-stage funnel. Stage one is dumb, fast, and high-recall: several independent retrievers each return a ranked list, and you merge them. Stage two is smart, slow, and high-precision: a model reads the merged shortlist and reorders it. You never let the smart model touch 50k products — only the top ~30.

**Mechanically:**

- **Route A (lexical):** BM25 over title + brand + category. Catches exact model names and brands that embeddings smear.
- **Route B (dense):** Embed the rewritten, state-aware query, cosine-search a FAISS index of product embeddings. Catches "something cozy for autumn."
- **Route C (structured):** Category-tree and attribute lookup from parsed slots.
- **Fusion:** Reciprocal Rank Fusion — `score(d) = Σ_routes 1/(k + rank_r(d))`, `k=60`. RRF works on ranks, not scores, so you skip the entire score-normalization problem. This is the single biggest time-saver in your retrieval stack.
- **Rerank:** Feed the top 30 candidate titles + attributes to a reranker in one prompt; ask for an ordered list of the top 10 IDs. Listwise beats pointwise here because the model can compare candidates against each other.

**Papers:**

- Cormack, Clarke & Buettcher, *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods* (SIGIR 2009) — four pages, one formula, implement it directly.
- Sun et al., *Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents* (EMNLP 2023) — the RankGPT paper. Steal the sliding-window listwise prompt format verbatim.
- *(Optional background)* Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering* (2020) for why dual-encoder retrieval works at all.

---

### 3. Dynamic State Machine & Intent Override

**Plain English:** A dictionary of slots that survives across turns, plus rules for when to add to it versus when to burn it down. "Also make it waterproof" accumulates. "Actually, forget boots — I need a raincoat" overrides, and if you keep the old slots you'll return waterproof boots forever.

**Mechanically:**

- **State object:** `{intent, slots{}, slot_turn_added{}, negatives[], canonical_query, candidates_prev[]}`.
- Each turn, extract slots from the new utterance.
- **Override detection** — this is the hard part. Trigger a reset when you see (a) explicit reversal markers ("actually", "no", "instead", "never mind", "change of plans"), (b) a category-level slot conflict (new head noun maps to a different branch of the category tree), or (c) semantic distance between the new utterance and the accumulated canonical query above a threshold.
- **Reset is selective, not total:** wipe category-dependent slots (size, style, material) but keep category-independent ones (budget, recipient, occasion). A user who says "forget the boots, I need a raincoat" still has a $150 budget.
- **Slot decay:** attach a turn index; down-weight slots older than N turns in soft-scoring mode so stale preferences fade rather than calcify.
- **Proactive clarification:** compute a dispersion signal over the candidate pool — e.g., entropy over top-level categories in the top-50, or count of surviving items after filters. If the pool is over-general, pick the slot whose values are most evenly split (max information gain) and ask about that one, offering 3–4 concrete options rather than an open question.

**Papers:**

- Zhang et al., *Towards Conversational Search and Recommendation: System Ask, User Respond* (CIKM 2018) — the canonical framing of asking about attributes to converge.
- Aliannejadi et al., *Asking Clarifying Questions in Open-Domain Information-Seeking Conversations* (SIGIR 2019) — the Qulac work; useful for question selection strategy.
- *(For query rewriting across turns)* Elgohary et al., *Can You Unpack That? Learning to Rewrite Questions-in-Context* (EMNLP 2019).

---

### 4. Personalized Context Distillation

**Plain English:** Don't stuff the whole dialogue into the prompt. Periodically compress it into a small, structured artifact — one short-term session summary and one long-term user profile — and pass those to the ranker instead. *Distillation* because you're boiling the transcript down to its active ingredients.

**Mechanically:**

- **Short-term (session):** After every turn, merge the new utterance into a fixed-schema JSON blob capped at ~200 tokens: active constraints, rejected items and why, inferred style vector.
- **Long-term (profile):** Across sessions, aggregate stable signals — typical price band, preferred brands, size, aversions ("never suggests heels"). Persist to a dict keyed by user ID.
- **Injection:** The profile becomes a soft-scoring term (boost candidates matching profile brands/price band) and a prompt preamble for the reranker. Keep it a soft prior — hard-filtering on the profile makes the agent unable to accommodate a gift purchase.
- **The "self-evolution" framing they want:** log which policy branch fired per turn (asked / ranked / relaxed constraints) alongside the outcome, and use that to shift thresholds within the session. If clarification questions in this session haven't moved the candidate pool, stop asking and just rank. That's genuine runtime re-orchestration and it's cheap to implement as a small policy table.

**Papers:**

- Park et al., *Generative Agents: Interactive Simulacra of Human Behavior* (2023) — the memory-stream + reflection architecture. Section on retrieval/reflection is the part to read.
- Packer et al., *MemGPT: Towards LLMs as Operating Systems* (2023) — tiered memory with explicit paging between working context and long-term store.
- Liu et al., *Lost in the Middle: How Language Models Use Long Contexts* (2023) — your justification for why distillation beats stuffing, and where to place the profile in your prompt.

> **A note on vocabulary:** "Dynamic Context Programming" is the organizer's coinage, not an established term with a literature behind it. Don't waste hours searching for it. It maps onto what the field calls *context engineering* / *memory management*, and the three papers above are the closest real anchors. In your Devpost, define the term yourself and show your implementation of it — judges reward a team that owns the definition.

---

## PART 2: 5-Person Team Work Breakdown

The merge-conflict problem is solved by **file ownership**, not by coordination. One person owns each directory. Nobody edits another's directory. All cross-boundary communication happens through dataclasses in a single `contracts.py` that is frozen at T-66h and can only change by unanimous agreement.

```
src/
  contracts.py        ← FROZEN at T-66h. Owned by R4, changed only by group consent.
  retrieval/          ← R1 only
  ranking/            ← R2 only
  dialog/             ← R3 only
  agent.py            ← R4 only (the glue)
  memory/             ← R5 only
eval/                 ← R4 only
docs/, demo/          ← R5 only
```

### The contract, concretely

```python
@dataclass
class Candidate:
    asin: str
    score: float
    route: str              # "bm25" | "dense" | "category"
    meta: dict              # title, brand, price, category_path, features

@dataclass
class RetrievalRequest:
    canonical_query: str
    intent: str             # "buy" | "browse"
    hard_filters: dict
    soft_prefs: dict
    top_k: int

@dataclass
class SessionState:
    turn: int
    intent: str
    slots: dict
    slot_turn_added: dict
    negatives: list
    canonical_query: str
    history: list
    profile: dict

@dataclass
class AgentResponse:
    recommendations: list[str]   # ordered ASINs
    message: str
    asked_clarification: bool
```

### R1 — Data & Retrieval Lead

| | |
|---|---|
| **Owns** | `retrieval/`, the FAISS index, the embedding cache |
| **Responsibilities** | Catalog normalization and the product "document" text template (title + brand + category path + top features — this template is worth 3–5 points of Hit Rate on its own). Embedding the 50k catalog and caching to `.npy`. FAISS `IndexFlatIP`. BM25 tuning over the starter kit. Structured filter + relaxation ladder. RRF fusion. Recall diagnostics: is the gold ASIN even in my top-100? |
| **Hands off** | `search(RetrievalRequest) -> list[Candidate]` to R4. Guarantees ASINs are valid and `meta` is populated for R2's prompts. |
| **Needs from others** | The `hard_filters` / `soft_prefs` schema from R3 (agree the exact slot key names in hour one — `color` vs `colour` will cost you an hour at 3am). |

### R2 — Ranking & LLM Engineer

| | |
|---|---|
| **Owns** | `ranking/`, all prompt templates |
| **Responsibilities** | Local cross-encoder reranker as the primary path (see the shortcut note below). Listwise LLM reranker as the optional boost. Robust JSON parsing with repair and fallback-to-input-order. Response caching keyed by `hash(query, candidate_set)`. Prompt versioning so you can A/B. Deduplication and diversity injection for the Browse track. |
| **Hands off** | `rerank(query, state, list[Candidate]) -> list[Candidate]`. Contract guarantee: never returns fewer than 10 items, never returns a hallucinated ASIN, never raises. On any failure it returns the input order. |
| **Needs from others** | `Candidate.meta` fields from R1 (fix the exact keys used in the prompt). Distilled profile string from R5. |

### R3 — Dialogue State Architect

| | |
|---|---|
| **Owns** | `dialog/` |
| **Responsibilities** | Intent classifier. Slot extraction (regex + gazetteer from the actual catalog vocabulary first, LLM second). Override detection and selective slot reset. Query rewriting into `canonical_query`. Clarification policy and question generation. The turn-budget guard — the hard stop that guarantees you never hit turn 11 and score zero. |
| **Hands off** | `update(state, utterance) -> SessionState` and `should_clarify(state, candidates) -> Optional[str]`. |
| **Needs from others** | Candidate pool stats from R1 to compute the over-generality signal. |

### R4 — Evaluation & Integration Ops

| | |
|---|---|
| **Owns** | `agent.py`, `eval/`, CI, the leaderboard |
| **Responsibilities** | The single integration point. Reproduces the baseline in hour one. Splits the 200 public sessions into dev-150 / holdout-50 and guards the holdout jealously — it's your only defense against overfitting to the public set before the private 800 hits you. Builds a fast eval loop (50 sessions, under 90 seconds) plus a full nightly run. Runs every ablation. Maintains a scoreboard in the team channel. Error analysis: which sessions fail, and at which stage. Latency budgets and caching. Submission packaging and the SHA256 verification. |
| **Hands off** | Score deltas within 10 minutes of any merge. This role is what stops five people from arguing about whose idea is better. |
| **Needs from others** | Nothing. This person is unblocked from hour zero, which is why they answer the fixed-script-vs-simulator question first. |

### R5 — Memory & Deliverables Strategy Lead

| | |
|---|---|
| **Owns** | `memory/`, `docs/`, `demo/`, Devpost |
| **Responsibilities** | Pillar III: session distillation, long-term profile store, the adaptive-orchestration policy table. Then: architecture diagrams, README, reproduction instructions, the limitations reflection, the demo video, Devpost copy, and mapping every judging criterion to a concrete artifact. |
| **Hands off** | `distill(state) -> str` (a ≤200-token profile block) and `apply_profile(profile, candidates) -> candidates` (soft boost). |
| **Why this pairing** | Pillar III is the most demonstrable differentiator and the least entangled with the hot path, so it can be built and shown without blocking R1–R3. And a pure-documentation role idles for 60 hours; this one doesn't. |

> **Note the weighting:** Technical Execution is 35%, but Innovation + Impact + Feasibility together are 55%. R5's work is not overhead. A team with a 0.42 MRR and a crisp story beats a team with 0.45 and a README stub.

---

## PART 3: 72-Hour Execution Plan

> **Sleep is scheduled, not optional.** Two shifts: R1/R2/R4 sleep T-58→T-50; R3/R5 sleep T-52→T-44. Everyone gets a second block around T-30. A team that doesn't sleep ships a broken submission at T-1h.

### Phase 0 — T-72 to T-66 · Contract Freeze & Ground Truth

**Goal:** Baseline reproduced and scored. `contracts.py` frozen. Every role can work without talking to anyone for the next 12 hours.

| Role | Task |
|---|---|
| **R4** | Clone kit, verify SHA256, run BM25 baseline, record the number. Read the evaluator loop and answer the fixed-script question. Post the answer to the team channel. Set up repo, branch protection, dev/holdout split. |
| **R1** | Inspect catalog schema. Draft the product document template. Kick off embedding of all 50k products immediately — this runs in the background while you do everything else. |
| **R3** | Hand-read 30 dev sessions. Build the slot vocabulary from the data, not from imagination. Draft the slot schema and post it. |
| **R2** | Download the cross-encoder. Decide the LLM situation (whose key, what budget, what's the offline fallback). |
| **R5** | Set up Devpost skeleton. Draft the architecture diagram from the four pillars — this doubles as the team's shared mental model. |
| **All** | 30-minute meeting at T-70: agree slot key names, `Candidate.meta` keys, and freeze `contracts.py`. |

> **Hackathon shortcut:** Don't design the perfect schema. Take the 15 slots that actually appear in the dev sessions, name them, and move on. A schema you can change in an hour is worth more than a schema you argued about for three.

### Phase 1 — T-66 to T-50 · The Vertical Slice

**Goal:** End-to-end hybrid pipeline beating BM25 baseline on dev-150. Ugly is fine. Working is mandatory.

| Role | Task |
|---|---|
| **R1** | FAISS `IndexFlatIP` over cached embeddings. RRF fusion of BM25 + dense. Report Recall@100 — this is your ceiling and every downstream point depends on it. |
| **R2** | Cross-encoder rerank of top-30. Wire the never-fails contract (try/except → input order). |
| **R3** | Regex + gazetteer slot extraction. Naive intent heuristic. Multi-turn slot accumulation, no override logic yet. |
| **R4** | Wire `agent.py`. Get the fast eval loop under 90 seconds. Post first hybrid score. |
| **R5** | Slot-merge-based distillation (no LLM). Start recording B-roll of the pipeline running. |

> **Hackathon shortcut — this is the big one:** Make the local cross-encoder your **primary** ranker, not the LLM. The organizer provides no API keys. A `ms-marco-MiniLM-L-6-v2` cross-encoder over 30 candidates is deterministic, runs in ~200ms on CPU, costs nothing, never rate-limits, and gets you most of the way there. Add the LLM listwise reranker as a conditional layer on top (e.g., only for the Browse track, or only when cross-encoder scores are closely bunched). This decouples your score from a fragile external dependency, and "graceful degradation with no external API" is a direct Feasibility & Practicality point.

> **Second shortcut:** At 50k × 384 dims, a raw numpy matmul is exact and takes milliseconds. FAISS `IndexFlatIP` is the same thing with a nicer API. Do not build IVF, do not build HNSW, do not tune `nprobe`. Approximate search at this scale is pure downside.

### Phase 2 — T-50 to T-36 · Depth on the Hard Pillars

**Goal:** Override handling, dual-track differentiation, and clarification policy all live and measurably contributing.

| Role | Task |
|---|---|
| **R1** | Constraint relaxation ladder (drop lowest-confidence slot on zero results). Browse-track diversification — cap items per category in the fused list. Route weight sanity check. |
| **R2** | Listwise LLM reranker with the RankGPT-style prompt. Caching. Prompt v2 with negative constraints ("user rejected X because Y"). |
| **R3** | Override detection: reversal markers + category-conflict + semantic distance. Selective slot reset. Over-generality signal and the max-information-gain question selector. Ship the hard turn-budget guard. |
| **R4** | Ablation table: BM25 / +dense / +RRF / +cross-encoder / +LLM. Post the marginal contribution of each. Kill anything with a negative delta. |
| **R5** | Long-term profile store, soft-boost scoring. First README pass. |

> **Hackathon shortcut:** For override detection, ship the regex marker list first (`actually|instead|no wait|never mind|forget|change of plans|different`) plus a category-head-noun mismatch check. That combination catches the large majority of real overrides at near-zero cost and zero latency. Only add the embedding-distance check if R4's error analysis shows it's actually missing cases. Don't build an LLM-based override classifier — it adds a full round-trip to every single turn for a rare event.

### Phase 3 — T-36 to T-24 · The MTTC Trade-off

**Goal:** Empirically resolve how aggressively to clarify. Self-evolution layer demonstrable.

| Role | Task |
|---|---|
| **R4** | The key experiment: sweep clarification aggressiveness across {never, once, twice, adaptive} and plot MRR against MTTC. Find the combined-score optimum. This single sweep is worth more than any modeling work done in this window. |
| **R3** | Implement whatever the sweep says. Tune the entropy threshold to match. |
| **R2** | Prompt iteration against R4's failure cases only. Stop guessing; read the losses. |
| **R1** | Recall failure analysis: for every session where gold is outside top-100, find out why. Usually it's the document template or a bad hard filter. |
| **R5** | Adaptive orchestration: in-session policy table that stops asking when questions stop narrowing the pool. Record the demo segment that shows this happening. |

> **Hackathon shortcut:** Whatever the sweep says, **always return your current best 10 recommendations alongside any clarification question.** You get a free shot at converting on that turn while still gathering information. There is no scenario where returning a question with an empty list is correct. If the eval scores each turn's list, this is close to free MRR.

### Phase 4 — T-24 to T-10 · Freeze, Harden, Verify

**Goal:** Feature freeze at T-20. From there, only bug fixes and documentation.

| Role | Task |
|---|---|
| **All** | Feature freeze at T-20h. Enforce it. |
| **R4** | Clean-clone reproduction test: fresh machine, follow the README literally, confirm the score. Run the holdout-50 for the first time — if the gap to dev-150 is large, you overfit; revert to the simplest config that holds up. Determinism check: same seed, same score, twice. |
| **R1/R2/R3** | Bug fixes only. Timeout and exception hardening on every external call. Verify the turn guard by force-feeding an 11-turn session. |
| **R5** | Record and edit the demo video. Full Devpost writeup. Limitations section — be specific and honest here; judges reward a team that knows its own weaknesses. |

> **Hackathon shortcut:** Your README's reproduction section should be three commands or fewer. If setup requires a paragraph of prose, judges will not reproduce it, and "the demo runs reliably" is explicitly in the 35% Technical Execution criterion. Ship a `make eval` target.

### Phase 5 — T-10 to T-0 · Submit Early

**Goal:** Everything submitted by T-4h. The last four hours are buffer, not work.

| Role | Task |
|---|---|
| **R5** | Upload video to YouTube (public — verify in an incognito window). Devpost complete with all required fields. Repo public — verify in incognito. |
| **R4** | Final tagged release. Checksum verify. One last clean-clone run. |
| **R1/R2/R3** | Prep pitch answers. Each person must be able to defend their component's design decision and name its limitation. |
| **All** | Dry-run the pitch twice. |

> **Hackathon shortcut:** Submit a complete, working version at T-12h with whatever you have. Then improve and resubmit. A submitted mediocre system scores; an unsubmitted excellent one does not. This is the single most common way strong hackathon teams lose.

---

## Three failure modes I'd bet on

1. **Over-investing in the LLM ranker before verifying retrieval recall.** If the gold ASIN isn't in your top-100, no ranker on earth recovers it. R1's Recall@100 number is the ceiling on everything; check it in Phase 1 and re-check it after every retrieval change.
2. **Building elaborate clarification logic that MTTC punishes.** Hence the Phase 3 sweep, and hence the question at the top of this document.
3. **Overfitting to the 200 public sessions.** The private 800 use different users and different target products. Every hyperparameter you tune on 200 sessions is a small bet against generalization. Prefer the simpler configuration when scores are within noise, and treat the holdout-50 as sacred.
