# R1 Retrieval — phase log

Append-only. One row per phase gate. Scores from `py -m eval.run_eval --mode fast` (fast-50)
unless noted; Recall@100 from `py -m eval.recall_probe` (turn-1 messages, dev-150) unless noted.

| date | phase | what changed | Recall@100 | Hit@10 | MRR | TechScore | wall-clock |
|---|---|---|---|---|---|---|---|
| 2026-08-30 | 0 | recon only, no code | — | — | — | — | — |
| 2026-08-30 | contract | `contracts.py`: +6 optional `RetrievalRequest` fields, `RetrievalResult(list)` return type, +2 `SessionState` / +2 `DialogResult` fields. No behaviour change. | 0.5733 | 0.08 | 0.035 | 0.0653 | test 2.7s / eval-fast 8.9s |
| 2026-08-30 | 1 | `eval/recall_probe.py` + `tests/test_recall_probe.py`. Offline recall harness; no retrieval logic touched. | **0.5733** | 0.08 | 0.035 | 0.0653 | probe 10.3s / test 3.5s / eval-fast 8.9s |
| 2026-08-30 | 2 | BM25 fix: expanded query-side stopwords (conversational scaffolding) + `features` weight 2.5→4.5. Both config-only. Porter stemming and a `details` key allowlist were tested and rejected (both lowered recall). | **0.6400** | 0.10 (fast) / 0.1467 (full) | 0.0494 / 0.0865 | 0.0848 / **0.1278** | probe 11s / test 3.8s / eval-fast 9s / eval-full 22s |
| 2026-08-30 | 3 | dense embedding index builder (`embed_index.py`), `enabled:false` | 0.640 | — | — | 0.1278 | build 28min |
| 2026-08-30 | 4 | `dense_search` / `dense_search_batch` | 0.640 / dense 0.427 / union 0.733 | — | — | 0.1278 | — |
| 2026-08-30 | 5 | BM25+dense fusion (`fusion.py`, `retriever.py`); z-score, bm25:3/dense:1. `enabled:false` (torch + 77 MB cache cost) | 0.647 (fusion, ~flat) | 0.180 (fusion on) | 0.094 | 0.1278 committed / **0.1535 fusion-on** | test 3.6s |
| 2026-08-30 | 6 | pool post-processing (`postprocess.py`): **popularity prior `enabled:true` w=0.2**, category filter + relaxation ladder (inert R3 hook), soft prefs (inert R5 hook) | 0.773 (probe) | 0.147→**0.193** | 0.087→**0.116** | 0.1278→**0.1693** | test 4.6s |
| 2026-08-30 | 7 | multi-turn (`multiturn.py`): query accumulation `enabled:true` (inert until R4 wires turn/session), profile-blend rejected, Rocchio unmeasurable | — | 0.193 | 0.116 | **0.1693** (not wired live) | test 4.6s |

## R1 final summary — ablation, negatives, cross-role notes

### Ablation (dev-150). Recall@100 from `recall_probe` (turn-1 queries); TechScore from `run_eval --mode full`.

| step | Recall@100 | TechScore | Δ TechScore | committed? |
|---|---|---|---|---|
| 0. shipped baseline (BM25, starter weights) | 0.573 | 0.1278* | — | — |
| 1. + query-side stopwords + `features` weight 4.5 (Phase 2) | 0.640 | 0.1278 | +0.000† | **yes** |
| 2. + dense route, alone (Phase 4) | 0.427 | — | *worse alone* | no (fusion only) |
| 3. + RRF fusion of BM25 ∪ dense (Phase 5) | 0.607 | 0.1281 | +0.000 | **no — a wash** |
| 4. + z-score fusion, bm25:3/dense:1 (Phase 5) | 0.647 | 0.1535 | +0.026 | no (torch + cache cost) |
| 5. + popularity prior, w=0.2 (Phase 6) | 0.773 | **0.1693** | **+0.042** | **yes** |
| 6. + category hard filter, oracle (Phase 6) | 0.787 | ~0.169 | +0.000 | ships inert (R3 hook) |
| 7. + query accumulation, oracle dialog (Phase 7) | 0.987 @t4 | — | large, contingent on R3 | ships inert (`enabled:true`, R4 wires) |
| — fusion + popularity 0.3 together (not committed) | — | 0.2977 | +0.170 | no |

\* Phase 1 fast-50 logged 0.0653; the 0.1278 column is `--mode full`, the honest per-phase comparison base.
† Phase 2 moved full-150 TechScore 0.1141 → 0.1278; the "+0.000" entries are deltas *within* this table's full-150 measurement, rounded.

**Committed pipeline: BM25 (Phase-2 weights) + popularity prior 0.2 + inert multi-turn/filter hooks → TechScore 0.1278 → 0.1693 (+32%), Recall@100 0.573 → 0.773.** Fusion (+0.026 alone, +0.17 stacked with a higher popularity weight) is one config line away, gated only on the team accepting the `torch` dependency + shipping the 77 MB embedding cache.

### What did NOT work (or barely did)

- **Dense retrieval alone** — Recall@100 0.427 vs BM25's 0.640 on the full 50k. The 8k-distractor A/B over-flattered it (0.77); at full scale the near-duplicate clothing crowd buries the gold. Only earns its place inside fusion.
- **BLAIR (`hyp1231/blair-roberta-base`)** — lost to bge-small by ~12pt R@100 and builds 5× slower. Its (metadata↔review) contrastive pretraining doesn't transfer to short synthetic queries.
- **RRF fusion** — a wash (TechScore +0.0003). Gains hit@10, loses MRR by the same amount: its flat `1/(k+rank)` demotes BM25's rank-1/2 golds whenever dense disagrees. On a catalog this lexically homogeneous, discarding score magnitude discards the signal. z-score fusion (magnitude-aware) is the one that works.
- **Porter stemming** — lowered Recall@100 0.593 → 0.533. Over-stemming pulls false matches into the top-100.
- **`details` key allowlist** (index only "signal" keys) — lowered Recall@100 0.593 → 0.553. BM25's IDF already discounts the date/dimension noise, and the allowlist hurt `intent_override` whose queries carry long detail sentences.
- **Category hard filter** — works, relaxation ladder fires correctly, but +1-2pt at the *oracle* ceiling and ~0 once the popularity prior is in. The `categories` field is already BM25-weighted 4.0. Ships because it's R3's contracted hook, not because it moves the number.
- **Profile blend** (`user_profile.preference_tags` into the query) — Phase 1's +5pt didn't survive. Net-negative against the accumulated query on the full pipeline; the tags are too generic and add noise, worst on browsing (t1 0.333 → 0.083).
- **Rocchio relevance feedback** — implemented + unit-tested but unmeasurable: the evaluator provides no accept/reject signal, so `request.accepted` / `.negatives` are always empty.

### Cross-role notes (informational — R1 acted on none of these)

- **Popularity is the dominant signal and it is a property of the benchmark's target selection** (148/150 dev targets above catalog-median rating count; median at the 99.5th percentile). The submission spec says the private set is built the same way. R1 ships this at a deliberately conservative `weight: 0.2`; the dev-150 curve keeps climbing to TechScore 0.28 at 0.5. **R4: sweep `retrieval.popularity.weight` on the holdout split with the real reranker before raising it** — it's measured with NullReranker, and a high weight makes the agent lean on "popular + loosely matching" and ignore the conversation.
- **Retrieval's turn-1 ceiling for browsing is ~0.48 Recall@100 / 0.05 hit@10** — its query is 2-3 content tokens ("I'm looking for <category>, but I'm still exploring"). Nothing R1 does to scoring fixes a query that thin. Browsing hit@10 needs R3 to extract constraints; the Phase 7 accumulation numbers show it then climbs to 0.92 by turn 4.
- **Multi-turn retrieval is wired but dormant.** R4's ~8-line glue (`docs/plan/r1_contract_change.md`) populates `RetrievalRequest.{session_id,turn,negatives,profile,intent_changed}` and copies `RetrievalResult.{pool_size,dropped_constraints}` onto `SessionState`. Query accumulation (`multiturn.enabled: true`) then activates and is the correctness path for turn ≥ 2 — without it, retrieval on turn 2+ sees only the newly disclosed constraint.
- **`intent_override` is retrieval's strongest scenario** (turn-1 R@100 0.909) — its query carries a full distinctive feature sentence. The `intent_changed` flag wipes `SessionMemory` for that session so accumulation doesn't drag the old preference forward.
- **`RetrievalResult.dropped_constraints`** now carries real data when a category filter relaxes (and always lists `color`/`size`/etc. as unenforceable). R3's "ask a narrowing question" trigger can read it straight off `SessionState` once R4 wires the copy.

## Phase 2 detail — BM25 Recall@100: 0.573 → **0.640** (dev-150, turn-1 queries)

Ablation, each lever measured alone against `eval/recall_probe.py` (config-only, no code paths added):

| change | Recall@100 | note |
|---|---|---|
| baseline (end of Phase 1) | 0.573 | title 6 / cats 4 / feat 2.5 / details 2.5 / store 1.5 / desc 1 |
| + Porter stemming (`porter unicode61 …`) | 0.533 | **rejected** — over-stemming pulls false matches into the top-100 |
| + expanded stopwords | 0.593 | **kept** — kills "still / exploring / key / requirement / actually / …"; browsing +1.6pt, buying +3.4pt |
| + `details` key allowlist (index only signal keys) | 0.553 | **rejected** — hurts intent_override (its queries carry long detail sentences); BM25 IDF already discounts date/dimension noise |
| + `features` weight 2.5 → 4.5 | 0.627 → 0.640 | **kept** — material/fabric/closure strings live in `features` and match customer constraints directly. 4.5 and 5.0 tie; 4.5 chosen |
| raising title / categories / store, lowering details / description | ≤ baseline | **rejected** — all neutral-to-negative |

Final Phase-2 weights: `title 6 / categories 4 / features 4.5 / details 2.5 / store 1.5 / description 1`, tokenizer `unicode61 remove_diacritics 2`.

Per-scenario Recall@100: buying 0.633 → **0.733**, browsing 0.417 → **0.483**, intent_override 0.909 (flat), boundary 0.375 (flat, n=8). Oracle stays 1.000. Median rank when found 64.5 → 49.

Scored impact (NullDialog + NullReranker, so retrieval order == final order): fast-50 TechnicalScore 0.0653 → 0.0848; full-150 0.1141 → **0.1278**. This is the cheapest recall in the project — config-only, no new dependency, no new code path.

---

| date | phase | what changed | Recall@100 | Hit@10 | MRR | TechScore | wall-clock |
|---|---|---|---|---|---|---|---|
| 2026-08-30 | 3 | `src/retrieval/embed_index.py` + `tests/test_embed_index.py` (6). Dense embedding index builder: pluggable encoder, SHA-keyed `.npy` cache. `embedding.enabled=false` — nothing wired into `search()` yet. | BM25 0.640 (unchanged) / **dense-alone 0.427** / **BM25∪dense 0.733** | 0.10 (fast, unchanged) | 0.049 | 0.0848 (unchanged) | build 28.2min / test 5s |

## Phase 3 detail — dense embedding index

Built `src/retrieval/embed_index.py`:
- **Pluggable encoder** (`Encoder` base; length-sorted batching; L2-normalise at build time). Two impls: `SentenceTransformerEncoder` (bge), `BlairEncoder` (`AutoModel`, `last_hidden_state[:,0]`, no ST dependency). `torch`/`transformers` imported lazily — the module + doc-template helpers load on numpy alone.
- **Cache**: `<key>.npy` (float32, L2-normed, row i ↔ `parent_asins[i]`) + `<key>.asins.txt` (parallel id index) + `<key>.meta.json` (provenance). `key = sha256(model + doc_template.version + catalog_sha256)[:16]`. On load, all three inputs are re-verified against the meta; any mismatch **rebuilds or hard-fails** (`allow_rebuild=False`), never serves stale vectors. Reload of the built cache: **0.10s** (vs 28min build).
- `.cache/` gitignored. bge matrix = 50000×384 float32 = **76.8 MB**.

### Model + template A/B

Subsample recall (150 dev gold + 8k random distractors — absolute numbers optimistic, deltas real):

| model | template | details | R@100 (subsample) |
|---|---|---|---|
| bge-small-en-v1.5 | cats[2:5] | ON | **0.767** |
| bge-small-en-v1.5 | cats[1:4] | ON | 0.713 |
| bge-small-en-v1.5 | cats[1:4] | OFF | 0.707 |
| `hyp1231/blair-roberta-base` | cats[1:4] | ON | 0.593 |

- **Model = bge-small-en-v1.5.** BLAIR lost by ~12pts and builds ~5× slower (CPU throughput: bge ~22 doc/s @ seq160, BLAIR ~6 doc/s). BLAIR's (item-metadata ↔ review-text) contrastive pretraining doesn't transfer to short synthetic templated queries; `BlairEncoder` kept in the code (pluggable) for a later revisit with mean-pooling / a query prefix. "Let the number decide" — it did.
- **`details` in the doc template: a wash** (+0.6pt R@100 / −1.3pt R@50). The build plan's spec omitted `details`; including it costs nothing and gives a small browsing edge, so it's ON — but this is **not** a differentiator, which is itself the defensible writeup finding.
- **`categories[2:5]`** (drop root + the gender/promo level) beat `[1:4]` by +5pt. Locked into `doc_template.version: v2`.
- bge **query prefix** (`"Represent this sentence for searching relevant passages: "`) adds +2.7pt on full-catalog dense; applied to queries only in Phase 4.

### Full-catalog reality check (all 50k, dev-150 turn-1 queries)

| | R@10 | R@50 | R@100 | R@500 |
|---|---|---|---|---|
| BM25 (Phase 2) | 0.213 | 0.460 | **0.640** | 0.913 |
| dense (bge, v2, +qprefix) | 0.173 | 0.300 | **0.427** | 0.733 |
| best-of-both (oracle min-rank) | 0.287 | 0.560 | **0.733** | 0.960 |

**Dense alone is materially worse than BM25** — the 8k-distractor subsample massively over-flattered it; at 50k the near-duplicate clothing crowd buries the gold. But dense is **complementary**: at @100 it recovers **14 golds BM25 misses** (8 browsing, 5 buying, 1 override) vs 46 the other way; union 110/150. At @500, union 144/150 (0.960).

**Verdict for Phase 5:** dense earns its place *only in fusion*, and mostly for browsing (BM25's weakest scenario — thin queries). The fusion ceiling is **R@100 0.733** (from BM25's 0.640, a +9.3pt headroom). If Phase 5 fusion can't clear ~0.68 it isn't worth the dependency.

---

| date | phase | what changed | Recall@100 | Hit@10 | MRR | TechScore | wall-clock |
|---|---|---|---|---|---|---|---|
| 2026-08-30 | 4 | `src/retrieval/dense.py` (`dense_search`/`dense_search_batch`) + `tests/test_dense.py` (5). `recall_probe.py` gains `--dense`. `bm25.load_products()` extracted. Nothing wired into `search()`/agent. | BM25 0.640 / dense 0.427 / **union 0.733** | 0.10 (fast, unchanged) | 0.049 | 0.0848 (unchanged) | probe+dense 26s / test 3.3s |

## Phase 4 detail — `dense_search`

- **`dense_search(index, query, k, mask=None) -> RetrievalResult`** — same `list[Candidate]` shape BM25 returns, `route="dense"`. `emb @ q` then `argpartition` top-k. `mask` (bool array aligned to `embedding.parent_asins`) scores rows to `-inf` before selection — the Phase 6 hard-filter hook, and masked-out rows are never padded back in.
- **`dense_search_batch(index, query_vectors, k, masks=None)`** — vectorised path for offline eval (one `(nq, n)` matmul); `recall_probe.py --dense` uses it.
- Fail-soft: `build_dense_index()` returns `None` if `embedding.enabled` is false, the cache is missing and unbuildable, or torch won't import — retrieval then runs BM25-only.
- **Timing (dev-150):** vector search **1.51 ms/query** (well under the ~5ms target). Query *encoding* (the transformer forward, not the search) is ~99 ms/query on this CPU — fine inside the 5s retrieval timeout, but the reason the probe batch-encodes.

### Dense vs BM25 vs union (full 50k, dev-150, turn-1 queries) — via the production code path

| method | R@10 | R@50 | R@100 | R@500 | found |
|---|---|---|---|---|---|
| BM25 | 0.213 | 0.460 | **0.640** | 0.913 | 137 |
| dense (bge v2) | 0.173 | 0.300 | 0.427 | 0.733 | 110 |
| **BM25 ∪ dense** | 0.287 | 0.560 | **0.733** | 0.960 | 144 |

Per-scenario R@100, BM25 → union: browsing **0.483 → 0.617** (+13pt), buying 0.733 → 0.817, intent_override 0.909 → 0.955, boundary 0.375 (flat, n=8). Dense recovers **14** golds BM25 ranks outside 100 (concentrated in browsing/buying), loses 46 the other way — so it is strictly a *complement*, never a replacement. `dense_search` confirms the Phase 3 scratch numbers exactly.

---

| date | phase | what changed | Recall@100 | Hit@10 | MRR | TechScore | wall-clock |
|---|---|---|---|---|---|---|---|
| 2026-08-30 | 5 | `src/retrieval/fusion.py` (RRF + min-max + z-score), `src/retrieval/retriever.py` (`HybridIndex`, fusion-aware `search()`), `tests/test_fusion.py` (8). `config.retrieval.fusion` block. **Default `enabled: false`** — capability shipped, not switched on. | 0.647 (probe, ~flat vs 0.640) | 0.147 → **0.180** | 0.087 → **0.094** | **0.1278 → 0.1535** (+20%, fusion ON) | probe A/B 3min / test 3.6s / scored full-150 A/B 8min |

## Phase 5 detail — BM25 + dense fusion

`fusion.py` implements three fusers behind `config.retrieval.fusion.method`:
- **`rrf`** — Reciprocal Rank Fusion, `score(d) = Σ w_i / (k + rank_i(d))`. Position only, magnitude ignored.
- **`minmax`** — per-list min-max normalise raw scores to [0,1], weighted sum. Absent item → 0.
- **`zscore`** — per-list z-score normalise, weighted sum. Absent item → ~mean.

`retriever.py` is the new public entry point: `build_index()` returns a `HybridIndex` (BM25 + optional dense, delegating `.products`/`.fallback_pool`/`.connection` to BM25 so `agent.py` needs no change), `search()` is **exactly BM25** unless `fusion.enabled` AND a dense index built. On enable it widens the BM25 request to `depth` (200), runs `dense_search` to the same depth, fuses. Any failure on the dense/fusion path → plain BM25 result + a loud log. Encoder is preloaded on `build_index()`'s thread (`dense.encode_queries(["warmup"])`) so the ~20-70s model load never happens inside `agent.py`'s timeout executor.

### The A/B — two harnesses, two different stories

**Recall probe (dev-150, turn-1 query, R@100 = coverage):** fusion is ~flat.

| config | R@10 | R@50 | R@100 | R@500 | browsing R@100 |
|---|---|---|---|---|---|
| BM25 alone | 0.213 | 0.460 | **0.640** | 0.913 | 0.483 |
| dense alone | 0.173 | 0.300 | 0.427 | 0.733 | 0.367 |
| rrf d200 w1/1 k60 | 0.207 | 0.453 | 0.600 | 0.860 | 0.417 |
| rrf d500 w1/1 k60 | 0.207 | 0.453 | 0.607 | 0.920 | 0.450 |
| minmax d200 w1/1 | 0.227 | 0.453 | **0.647** | 0.860 | 0.483 |
| minmax d200 w2/1 | 0.240 | **0.487** | 0.633 | 0.860 | 0.467 |
| zscore d200 w3/1 | **0.247** | 0.440 | 0.593 | 0.860 | 0.417 |

Best fusion R@100 is 0.647 vs BM25's 0.640 — inside noise (1 session / 150). The Phase 4 gate ("if fusion can't clear ~0.68 R@100 it isn't worth the dependency") **fails on this metric.**

**Scored eval (dev-150, `run_eval` full, NullDialog + NullReranker so retrieval order == final order):** fusion is a large win.

| config | hit@10 | MRR | MTTC | **TechScore** | Δ vs BM25 |
|---|---|---|---|---|---|
| BM25 only (fusion off) | 0.147 | 0.087 | 9.57 | **0.1278** | — |
| RRF d200 w1/1 k60 | 0.153 | 0.072 | 9.51 | **0.1281** | +0.0003 |
| minmax d200 w1/1 | 0.153 | 0.083 | 9.51 | 0.1315 | +0.004 |
| minmax d200 bm25:2 dense:1 | 0.173 | 0.085 | 9.31 | 0.1459 | +0.018 |
| minmax d200 bm25:3 dense:1 | 0.173 | 0.085 | 9.31 | 0.1459 | +0.018 |
| zscore d200 bm25:2 dense:1 | 0.173 | 0.095 | 9.31 | 0.1490 | +0.021 |
| **zscore d200 bm25:3 dense:1** | **0.180** | **0.094** | **9.24** | **0.1535** | **+0.0257 (+20%)** |

### Reads

- **The recall probe measured the wrong thing for this phase.** R@100 is coverage; the score is top-10 rank quality. Fusion barely adds *new* golds to the top-100 (probe ~flat) but it substantially *reorders* the ones already in the lexical neighbourhood — pulling golds from rank 20-90 into the top-10. R@500 union was 0.733 (lots of headroom at depth) but that headroom only converts to score if a fuser lifts those items to the top, which z-score does and RRF does not.
- **RRF is a wash — the honest negative of this phase.** hit@10 +0.6pt but MRR −1.5pt: RRF's flat `1/(k+rank)` reshuffles the top-10 but also demotes BM25's rank-1/2 golds (which RRF only credits `1/61`) whenever dense disagrees. Net TechScore +0.0003. On a catalog this lexically homogeneous, discarding score magnitude discards the signal.
- **z-score beat min-max** (0.1535 vs 0.1459 at the same weights). Magnitude-aware, and z-score's wider spread for an outlier top hit (vs min-max flattening everything between the min and max) matches how a strong dense match should behave — the top cosine hit on a near-duplicate catalog really is much better than #10, and the rank gap alone understates that.
- **BM25 stays primary (weight 3×).** Monotonic: w1/1 → w2/1 → w3/1 improves TechScore. Dense alone loses badly (R@100 0.43 vs 0.64); it's a corrective boost, never the base. This is consistent with Phase 4.
- Per-scenario hit@10, BM25 → zscore w3/1: buying 0.283 → 0.300, browsing 0.050 → 0.083, boundary 0.000 → 0.250 (n=8, so +2), intent_override 0.091 (flat — its query is already distinctive enough that BM25 nails it). Broad small gains, not one scenario carrying it.

### Locked config + the `enabled` decision

`config.yaml`: `method: zscore`, `depth: 200`, `weights: {bm25: 3.0, dense: 1.0}`, `rrf_k: 60` (inert for zscore). **`enabled: false`.**

Kept off by default despite the +0.0257 because turning it on is not self-contained — it forces onto the rest of the team: (1) the `torch` + `sentence-transformers` dependency in `requirements.txt`, (2) a ~20-70s `Agent.__init__` cost (cold vs warm model load) + ~100 ms/turn query encoding, (3) a 77 MB embedding cache that must ship with the submission or be rebuilt (~30 min, needs an HF download). Fail-soft covers the missing-cache case (→ BM25-only), so the downside is bounded at "no worse than BM25". **Recommendation: flip to `true` once the team commits to packaging the cache + dependency — it is the single biggest retrieval lever measured (bigger than Phase 2's +0.0137).**

---

| date | phase | what changed | Recall@100 | Hit@10 | MRR | TechScore | wall-clock |
|---|---|---|---|---|---|---|---|
| 2026-08-30 | 6 | `src/retrieval/postprocess.py` (popularity prior, category hard filter + relaxation ladder, soft prefs) + `tests/test_postprocess.py` (15). `config.retrieval.{popularity,filters,relaxation,soft_prefs}`. **Popularity prior `enabled: true` (weight 0.2)** — the others are inert R3/R5 hooks. | 0.773 (probe, pop 0.2) | 0.147 → **0.193** | 0.087 → **0.116** | **0.1278 → 0.1693** (+32%) | test 4.6s / probe sweep 2min / scored full-150 sweep 3min |

## Phase 6 detail — pool post-processing

`postprocess.py` runs three passes on the candidate pool after BM25/fusion, before the result leaves `retriever.search()`. Each is independently flagged and a no-op on empty input; reordering passes **preserve `Candidate.score` and change list order only** (rewriting score with a blended [0,1] value would mislead a downstream reranker about signal strength).

### 1. Popularity prior — the big lever

`final_order = sort by (1 - w) * lexical_norm + w * log1p(rating_number)_norm`, both min-max normalised across the current pool. Applies to **every** query (no dialog dependency).

**Why it works — a structural property of the benchmark.** The evaluator builds every session around a hidden target product, and it picks *popular* products: **148 of 150 dev targets are above the catalog's median rating count; the median target sits at the 99.53rd popularity percentile** (7,735 ratings vs a catalog median of 12; only 2 targets below median). The submission spec says the private set is constructed the same way. So leaning on `rating_number` is benchmark-aligned retrieval, not overfitting to specific asins — and Phase 0 flagged exactly this.

Scored A/B (dev-150, `run_eval` full, NullDialog + NullReranker so retrieval order == final order):

| weight | hit@10 | MRR | MTTC | **TechScore** | Δ vs BM25 |
|---|---|---|---|---|---|
| 0.00 (off) | 0.147 | 0.087 | 9.57 | **0.1278** | — |
| 0.05 | 0.160 | 0.090 | 9.44 | 0.1383 | +0.011 |
| 0.10 | 0.180 | 0.100 | 9.24 | 0.1553 | +0.028 |
| **0.20 (committed)** | **0.193** | **0.116** | **9.11** | **0.1693** | **+0.042 (+32%)** |
| 0.30 | 0.233 | 0.150 | 8.71 | 0.2074 | +0.080 |
| 0.50 | 0.313 | 0.208 | 7.91 | 0.2809 | +0.153 |

Recall probe (turn-1 queries) agrees and is even steeper: R@10 0.213 → 0.453 at w=0.2, → 0.680 at w=0.5. Per-scenario, `intent_override` hit@10 stays **flat at 0.091** at every weight while everything else climbs — its query already carries a distinctive feature sentence so BM25 nails it and popularity has nothing to add. That flat line is the tell that this is a real popularity effect, not a scoring artifact.

**The weight is deliberately left at 0.2 — the conservative end of a curve that keeps climbing.** Three reasons not to chase the dev-150 maximum: (a) measured with the Null reranker, so a real R2 cross-encoder re-sorting the top-30 changes the marginal value; (b) tuned on dev, not the R4-gatekept holdout; (c) a high weight makes the agent lean on "popular + loosely matching" and largely ignore the conversation — a hard regression if the private targets are even slightly less skewed. **R4 should sweep this on holdout with the real pipeline.** Zero dependency cost (pure arithmetic on `rating_number`), so unlike fusion this ships on by default.

### 2. Category hard filter + relaxation ladder — works, negligible payoff

`CategoryIndex` maps `parent_asin -> breadcrumb token set` (scaffolding words — clothing/shoes/men/women/… — stripped). `request.hard_filters["category"]` is matched **AND over tokens**. If the surviving pool drops below `relaxation.min_pool_size` (10), the lowest-priority filter is dropped and retried; every relaxed *or* unenforceable key (`color`, `size`, … — R1 has no index for those) is reported in `RetrievalResult.dropped_constraints`.

`hard_filters` is `{}` under NullDialog, so this is **inert live** — it is the hook R3 wires into. Measured offline with an **oracle** filter (feed `search()` the target's own coarse category, i.e. a perfect slot extractor):

| | R@10 | R@100 | R@500 |
|---|---|---|---|
| BM25, no filter | 0.213 | 0.640 | 0.913 |
| BM25 + oracle category filter | 0.233 | 0.653 | 0.927 |
| BM25 + pop 0.2 | 0.453 | 0.773 | 0.913 |
| BM25 + pop 0.2 + oracle category filter | 0.447 | 0.787 | 0.947 |

**+1-2pt at the oracle ceiling, ~0 once the popularity prior is in.** The `categories` field is already BM25-weighted 4.0, so the coarse category is mostly priced into the lexical score; a hard filter on it removes a little noise and nothing more. Real dialog will extract categories worse than the oracle, so the live payoff is ≤ this. Shipped because it is R3's contracted hook and the relaxation-ladder plumbing is needed regardless — not because it moves the number.

### 3. Soft preferences — inert R5 hook

`request.soft_prefs` (populated by R5/memory; `{}` today) can carry `store` (substring match) and `price_max`. Additive nudges on the min-max lexical score; **a missing attribute is neutral, never a penalty** (`price` is null in ~79% of the catalog). `enabled: false`. Shipped minimal — the contract slot exists so R5 has somewhere to write.

### The full stack (dev-150 scored) — levers stack super-additively

| config | hit@10 | MRR | **TechScore** |
|---|---|---|---|
| BM25 only | 0.147 | 0.087 | 0.1278 |
| + fusion (zscore, Phase 5) | 0.180 | 0.094 | 0.1535 (+0.026) |
| + popularity 0.2 (Phase 6) | 0.193 | 0.116 | 0.1693 (+0.042) |
| fusion + popularity 0.2 | 0.253 | 0.172 | 0.2281 (+0.100) |
| fusion + popularity 0.3 | 0.340 | 0.202 | **0.2977** (+0.170) |

fusion + pop together beat the sum of their individual gains — fusion widens the pool that reaches the popularity re-sort and z-score fusion's magnitude awareness compounds with the popularity blend. The **committed default is BM25 + popularity 0.2 = 0.1693** (fusion still off for its dependency cost); flipping fusion on is a one-line config change that R4 can make when the cache ships.

---

| date | phase | what changed | Recall@100 | Hit@10 | MRR | TechScore | wall-clock |
|---|---|---|---|---|---|---|---|
| 2026-08-30 | 7 | `src/retrieval/multiturn.py` (`SessionMemory`, `accumulate_query`, `blend_profile`, `rocchio_terms`, `build_effective_query`) + `tests/test_multiturn.py` (19). `eval/recall_probe.py --multiturn`. `config.retrieval.multiturn`. Wired into `search()` behind a turn/session gate. | — (inert live) | 0.193 (unchanged) | 0.116 (unchanged) | **0.1693 (unchanged — not wired live)** | test 4.6s / multiturn probe 60s |

## Phase 7 detail — multi-turn retrieval

`retriever.search()` now calls `build_effective_query()` first: it records the turn in a per-session `SessionMemory` (retrieval's own state, keyed by `RetrievalRequest.session_id` per the contract) and rebuilds the query from the whole conversation. **Gated on `config.retrieval.multiturn.enabled` AND `request.turn >= 1` AND a non-empty `session_id`.** agent.py sends `turn=0` today, so this is inert and **eval-full is byte-identical to Phase 6 (0.1693)** — verified, and the code path *does* run (returns the raw query), it is not a silent fallback. Activation is R4's ~8-line glue step (`docs/plan/r1_contract_change.md`).

### Query accumulation — correctness, not tuning

Measured against an **oracle turn stream** (`recall_probe.py --multiturn`): a perfect dialog that asks exactly the right attribute each turn, so the customer discloses one held-back constraint per turn in the evaluator's own `customer_reply` phrasing. Real dialog extracts fewer, later — this is the ceiling.

| strategy | turn 1 | turn 2 | turn 3 | turn 4 | (Recall@10, dev-150) |
|---|---|---|---|---|---|
| **latest turn only** (stateless) | 0.453 | **0.180** | 0.293 | 0.340 |
| **accumulate** (committed) | 0.453 | **0.700** | 0.867 | 0.927 |
| accumulate + profile blend | 0.360 | 0.587 | 0.793 | 0.900 |

(Recall@100 at turn 4: latest 0.387, accumulate **0.987**.)

**Latest-turn-only collapses at turn 2** — the turn-2 message is `"For that, what matters is: cotton."`, no category, no earlier context; retrieving on that alone is near-useless (R@10 0.18). **Accumulation is a correctness requirement once dialog is real**, not an optimisation: it takes turn-4 R@10 from 0.34 → 0.93. Per-scenario the pattern holds everywhere (`intent_override` turn-2: 0.09 latest → 0.86 accumulated).

Left **`enabled: true`** (still inert until R4 wires the turn/session fields) because there is no weight to tune and no dependency — the moment multi-turn retrieval is wired, this is the behaviour you want. Even with NullDialog wired, accumulation *helps*: it retrieves on the preserved turn-1 query instead of the `"Those options are not quite right yet"` filler string.

### Profile blend — measured and rejected

`blend_profile()` appends `user_profile.preference_tags`. Phase 1 saw a small +5pt when the baseline was the much weaker turn-1-only BM25. Against the **accumulated query on the full pipeline it is a net negative at every turn** (0.700 → 0.587 at turn 2), worst on browsing (turn 1: 0.333 → 0.083). The tags are generic — `warmth`, `fit`, `comfort`, `durability` — and just add noise to an already-good query, pulling in loosely-related popular items. `profile_blend: false`.

### Rocchio — shipped, unmeasurable here

`rocchio_terms()` does positive-feedback query expansion (Rocchio β: most frequent terms from the title/features of accepted items) and computes an `avoid_terms` list (γ). The evaluator provides **no accept/reject feedback** — `request.accepted` / `request.negatives` are always empty — so this cannot be measured on the public/dev sets. Unit-tested on fabricated metas; `rocchio: false`. It is the hook for when R5's memory component supplies a feedback signal.

### Discriminative-attribute helper — deliberately not built

The build plan listed "an attribute-discriminativeness helper" for Phase 7. It is purely an input to *R3's* decision of what to ask, R3 cannot import `src/retrieval/` (components never import each other), and building it would be acting on a coordinate-with-R3 step the self-contained constraint says to avoid. Skipped. If R3 wants it, the shape is trivial: given the candidate pool, score each `ASK_ATTRIBUTE` by how evenly it partitions the pool's `details` values (max entropy = most discriminating).

## Phase 1 detail — current BM25 Recall@100 = **0.573** (the team-wide ceiling)

Turn-1 reconstructed customer message → current `search()` (SQLite FTS5 BM25, config weights
`title 6 / categories 4 / features 2.5 / details 2.5 / store 1.5 / description 1`), dev-150.

```
                 R@10    R@50    R@100   R@500   median rank when found
OVERALL  (n=150) 0.193   0.420   0.573   0.907   64.5      (found 136/150)
 buying  (n= 60) 0.267   0.550   0.633   0.950   39        (found 57/60)
 browsing(n= 60) 0.017   0.200   0.417   0.867   108       (found 52/60)
 override(n= 22) 0.545   0.727   0.909   0.955   3         (found 21/22)
 boundary(n=  8) 0.000   0.250   0.375   0.750   132       (found 6/8)
 ORACLE  (n= 50) 1.000   1.000   1.000   1.000   1         <- index is healthy
```

Reads:
- **Oracle Recall@100 = 1.000** — feeding the gold's own title retrieves it at rank 1 every
  time. The index is not the problem; query↔document vocabulary mismatch is.
- **R@500 = 0.907 overall but R@100 = 0.573** — for ~1/3 of sessions the gold *is* in the
  lexical neighbourhood but ranked 100–500. Recall depth / scoring is the lever, not coverage.
- **browsing is the floor** (R@10 = 0.017, R@100 = 0.417). Its turn-1 query is only
  `"I'm looking for <coarse category>, but I'm still exploring."` — 2–3 content tokens, no
  material/feature. Nothing R1 does to BM25 scoring fixes a query that thin; this needs
  dialog/clarification to add constraints, or the popularity prior (Phase 0: gold sits at
  catalog popularity percentile ~99.5) as a browse-mode tie-breaker.
- **override is already strong** (R@100 = 0.909) — its query carries a full distinctive
  feature sentence.
- Appending `user_profile.preference_tags` to the query (measured, not wired):
  overall 0.573 → 0.620, buying 0.633 → **0.750**, browsing 0.417 → 0.383 (slight regression),
  found 136 → 133. Net positive on the KPI, driven entirely by buying. Flagged for R3/R5 —
  R1 does not inject profile into the query itself.

Ceiling implication for the team: **no downstream component can exceed 57% Hit-rate ceiling
on turn-1-only retrieval as it stands.** Phase 2 (BM25 field weighting / tokenisation /
stopwords) targets this number first.
