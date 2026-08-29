# Tuning Log — 0.733 → 0.919

Every number below is a full 200-session run of `evaluator/local_evaluator.py` on
`data/public_set.jsonl`. Reproduce a row with `python3 scripts/score.py <label>`.

## Diagnosis that drove the work

Two histograms located almost all of the recoverable score:

- **Rank histogram.** 91 of 171 hits sat at rank 1. If every hit were rank 1, MRR would be
  0.855 against an actual 0.567 — roughly 0.086 of TechnicalScore locked in ranking alone.
- **First-hit-turn histogram.** Zero hits after turn 8, which pointed straight at the
  clarification policy rather than at retrieval.

## What worked

| # | Change | Score | Δ |
|---|--------|-------|---|
| 0 | Pillar III baseline | 0.7334 | — |
| 1 | Question policy: never waste an actionable turn | 0.8436 | +0.110 |
| 2 | Reranker rebuild: colon normalization + popularity prior | 0.8568 | +0.013 |
| 3 | Weight sweep: profile boost 8.0 → 2.0, popularity up | 0.8972 | +0.040 |
| 4 | Category-path match boost | 0.9066 | +0.009 |
| 5 | Candidate pool 50 → 80 | 0.9115 | +0.005 |
| 6 | BM25 features/details 2.5 → 4.0 | 0.9124 | +0.001 |
| 7 | Drop whole utterances from slot phrases | 0.9165 | +0.004 |
| 8 | Candidate pool 80 → 60 | 0.9187 | +0.002 |

### 1. Question policy (the dominant fix)

`_choose_question` had two bugs. It returned `None` for every turn from 8 onward, and it
also returned `None` whenever retrieval did not look overloaded. A null `ask_attribute`
makes the simulated customer answer with pure boilerplate, so those turns produced no new
constraints at all.

The deeper point is in `customer_reply`: asking a **specific** attribute only yields
information when the customer happens to hold a constraint that `classify_constraint` maps
to that attribute, otherwise the reply is "I don't have an additional preference for X".
Asking `other` returns *any* two undisclosed requirements. Preferring the open question and
never skipping an actionable turn took boundary sessions from 0.50 to 1.00 hit rate.

### 2. Colon normalization

`details` are indexed as `"key value"` (see `_text` in `catalog_store.py`) but constraints
arrive as `"key: value"`. Exact-phrase matching therefore never fired on any
details-derived constraint. Normalizing colons to spaces on both sides fixed it.

### 3. Popularity prior

`POPULARITY_WEIGHT * log10(1 + rating_number)` is the strongest single ranking feature:
zeroing it costs ~0.043. Ground-truth targets are real purchases, which skew popular.
The same sweep showed `MAX_PROFILE_BOOST = 8.0` was over-promoting — a profile match could
outrank a full constraint match — so it dropped to 2.0 for ~0.026.

### 4. Category-path match

The product type in the opening message is produced by `coarse_category()` from the
target's own category path, so it is **guaranteed true for the target**. Scoring the
fraction of those tokens present in a candidate's `categories` field prunes wrong-category
distractors and moved 12 more hits to rank 1.

### 7. Slot phrase hygiene

`_extract_updates` assigns the entire user message to every attribute it matches. Those
whole utterances flowed into `_slot_values_from_filters`, consumed the 6-phrase budget in
`build_fts_expression`, and injected boilerplate tokens into the reranker's term list.
Dropping non-concise values in both modes lifted browsing to a perfect hit rate.

## What did not work

Recorded because these are the intuitive moves, and the negative results are the reason the
final design looks simple.

| Change | Score | Why it failed |
|--------|-------|---------------|
| Candidate pool 150 / 300 | 0.830 / 0.816 | Hit rate rises but MRR falls further; the reranker promotes tail distractors |
| BM25 rank as a linear prior in the rerank score | 0.846 | Dilutes the phrase signal |
| IDF-weighted phrase matches | 0.894–0.896 | Neutral to slightly negative even after rescaling phrase weights |
| Multi-route retrieval + reciprocal rank fusion | 0.865 | Generic phrases ("machine wash") retrieve noise; also broke intent_override 1.00 → 0.87 |
| Buying-mode constraint relaxation ladder | 0.892 | Narrow Boolean pools exclude the target |
| Same ladder with a pool-breadth guard | 0.874 | Its loosest level drops phrase clauses, unlike the browsing fallback |
| `average_rating` quality prior | 0.910–0.912 | No signal beyond review volume |
| Step bonus for the BM25 top-25 | 0.919 peak | A knife-edge spike; neighbours fell to 0.903, so it was rejected as overfitting |

**The consistent lesson: recall-first retrieval plus a strong reranker beats precision-first
Boolean retrieval on this task.** Every attempt to narrow the candidate set with hard
constraints lost more targets than it gained rank on.

## Defects found during review

- `CONSTRAINT_PATTERNS` used `(.+?)(?:\.|$)`, stopping at the first period, so
  `"heel height: 3.5 inches"` was captured as `"heel height: 3"`. Turns are now newline
  scoped so captures survive internal periods.
- Dead `budget_values` local in `constraint_phrases`.
- The buying AND expression **never matched anything** — measured 79/79 buying turns falling
  through to relaxed search. The recall-first fallback is what actually runs; the code now
  says so instead of implying a working precision route.
- The price/budget signal is inert: `local_evaluator.py` appends the budget candidate last,
  past its 4-constraint cutoff, so **0 of 200 sessions ever disclose a budget**. The logic
  is retained only as a guard in case the private set differs.

## Locked configuration

| Constant | Value | File |
|----------|-------|------|
| `CANDIDATE_POOL_SIZE` | 60 | `retrieval/search.py` |
| `BM25_WEIGHTS` | `(0, 6, 4, 4, 4, 1.5, 1)` | `retrieval/catalog_store.py` |
| `EXACT_PHRASE_SCORE` | 10.0 | `retrieval/reranker.py` |
| `TERM_MATCH_SCORE` | 1.5 | `retrieval/reranker.py` |
| `CATEGORY_MATCH_SCORE` | 6.0 | `retrieval/reranker.py` |
| `POPULARITY_WEIGHT` | 2.0 | `retrieval/reranker.py` |
| `MAX_PROFILE_BOOST` | 2.0 | `personalization/rerank_boost.py` |
| `LAST_ACTIONABLE_TURN` | 10 | `agent.py` |

## Overfitting note

These weights were tuned on the public set. The reassuring signal is that each sweep
optimum was a broad plateau rather than a spike — the final category/popularity sweep
spanned 0.9064–0.9067 across very different values — so the configuration does not sit on a
knife edge. The one genuine spike found (the BM25 top-tier step bonus) was rejected for
exactly that reason.

## Remaining headroom

At hit 0.995 and MTTC 2.12, a perfect MRR would score ~0.996, so ~0.077 remains and about
0.056 of it is MRR. The open question is how to separate the target from 55 near-miss
candidates that match the same disclosed phrases. Untried ideas: contradiction penalties
(disclosed "blue" versus a candidate stating another colour), and per-attribute weighting
so hard constraints outrank soft preferences.
