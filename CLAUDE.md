# CLAUDE.md — Operating Manual

TechJam Conversational E-Commerce Search Challenge (72h hackathon). Build a multi-turn shopping
`Agent` that finds a hidden target product (`parent_asin` from a frozen 50k-item Clothing catalog)
as early and highly ranked as possible, within 10 turns. Submission = one `Agent` entry file +
helpers + setup instructions + a short report. Full detail: `docs/plan/RECON.md`,
`docs/plan/FEASIBILITY.md`.

## THE FOUR EVALUATOR FACTS (everything downstream depends on these)

1. **The user simulator reacts ONLY to the structured `ask_attribute` field, never to the
   free-text `message`.** It is deterministic per sample (seeded by `sample_id`+`scenario_type`),
   but not a fixed script — what it says next depends on which `ask_attribute` we ask for.
   Writing clever natural-language questions extracts nothing extra; only picking the right
   `ask_attribute` value does. (`evaluator/local_evaluator.py:166-185`)
2. **Recommendations are scored every turn, not just the last one.** First turn the target
   appears in the top-10 wins the session. (`local_evaluator.py:251-255`)
3. **Scoring:** `TechnicalScore = 0.50*HitRate@10 + 0.30*MRR + 0.20*Efficiency`,
   `Efficiency = clip((11-MTTC)/10, 0, 1)`, miss = turn 11 for MTTC, 0 for MRR.
   (`local_evaluator.py:278-280`)
4. **Failure handling is asymmetric.** `respond()` exceptions / malformed output are caught and
   downgraded to an empty-recommendation miss for that turn only (`local_evaluator.py:239-244`).
   **`reset()` exceptions are NOT caught — they crash the entire evaluation run.**
   `reset()` must never be allowed to raise.

## Agent entry-point interface (verbatim, do not change signatures)

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None: ...
    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": str,
            "ask_attribute": "category"|"material"|"color"|"size"|"style"|"brand"|"budget"|"feature"|"use_case"|"other"|None,
            "recommendations": [{"parent_asin": str, "score"?: number}, ...],  # only first 10 valid unique scored
            "usage": {"prompt_tokens": int, "completion_tokens": int},  # optional
        }
```

State is carried entirely by the Agent instance, keyed by `session_id` — the harness never
replays prior turns back to `respond()`. `starter/agent.py` is a thin shim re-exporting
`src.agent.Agent` (the real implementation) — the evaluator's import path never changes.

## Repo map

```
data/            catalog.jsonl (50k, gitignored), public_set.jsonl (200 sessions)
docs/            spec/contracts/scoring config/baseline reference — read-only, don't edit
docs/plan/       RECON.md, FEASIBILITY.md, strategy.md, architecture.md (gitignored)
evaluator/       local_evaluator.py — DO NOT EDIT
starter/agent.py thin shim: `from src.agent import Agent`. No logic here, ever.
tests/           test_evaluator.py (organizer's) + test_null_paths.py + test_failure_contract.py (ours)
src/contracts.py FROZEN. Every cross-component dataclass. Changes need unanimous agreement.
src/config.py    shared config-loader (not a "component" — every leaf may import it)
src/retrieval/   R1. BM25 today; will grow dense + RRF.
src/ranking/     R2. NullReranker today; will grow cross-encoder + optional LLM listwise.
src/dialog/      R3. NullDialog today; will grow routing + slot extraction + override detection.
src/memory/      R5. NullMemory today; intra-session distillation only (see Decisions).
src/agent.py     R4. The only module that imports every component.
eval/            R4. run_eval.py, generate_split.py, dev_holdout_split.json (committed), results_log.jsonl
config.yaml      every tunable — grep src/ to verify no magic numbers outside it.
Makefile         eval / eval-fast / eval-holdout / test / smoke targets.
results.json     evaluator output, gitignored, regenerated each `py -m evaluator.local_evaluator` run.
```

One owner per directory; nobody edits another's directory except `agent.py`, which knows the graph.

## Architectural invariants (implemented, not aspirational)

- Components are leaves, never import each other; only `src/agent.py` imports the whole graph
  (verified: `retrieval/`, `ranking/`, `dialog/`, `memory/` each only import `src.contracts`/`src.config`).
- Every component is a pure function over `src/contracts.py` dataclasses.
- Every component has a permanent Null implementation, not scaffolding: `NullDialog` → raw
  utterance as `canonical_query`, `ask_attribute=None`. `NullMemory` → empty boosts/summary.
  `NullReranker` → input order unchanged. Retrieval's own fallback is the BM25 baseline itself; if
  BM25 fails too, `agent.py._fallback_candidates` drops to a precomputed rating-sorted pad pool.
- Every tunable lives in `config.yaml`. `grep`ped `src/` for bare numeric literals — the only two
  left are `0.0` sentinels for "no score/rating on record," not behavioral tunables.
- Failure contract, enforced in `agent.py`: every component call goes through
  `_call_with_fallback()` (1-worker `ThreadPoolExecutor`, per-component `config.yaml` timeout),
  falling back to the **explicit** `null_*.py` function — never re-invoking the primary — on any
  exception or timeout. `reset()` has its own nested try/except and can never propagate.
  `respond()`'s outer try/except catches even a bug in `agent.py`'s own glue code.
  `_ensure_top_k()` guarantees exactly `top_k` valid, unique IDs. Proven in
  `tests/test_failure_contract.py` (14 tests).

**Gotcha:** `agent.py` queries the shared BM25 SQLite connection from the `ThreadPoolExecutor`
worker thread, not the thread that built it. `sqlite3.connect` defaults to
`check_same_thread=True`, which silently raised on every retrieval call during Phase 2 build — the
try/except swallowed it and every response quietly fell back to the pad pool, still "scoring"
instead of crashing loudly. Fixed with `check_same_thread=False`
(`src/retrieval/bm25.py:build_index`). Any non-thread-safe resource needs the same care.

## Real, tested commands

`python`/`python3` are **not** on PATH here — use **`py`**. `make` is **not installed** on this
Windows box (`make --version` → not found) — the targets below are the documented commands for
any environment that has `make`; here, run the right-hand `py -m ...` command directly.

- `make eval` / `py -m eval.run_eval --mode full` — all 150 dev sessions.
- `make eval-fast` / `py -m eval.run_eval --mode fast` — first 50 of the 150 dev sessions.
- `make eval-holdout` / `py -m eval.run_eval --mode holdout` — 50 holdout sessions. **Use
  rarely** — the only defense against overfitting to the public set.
- `make test` / `py -m unittest discover -s tests -v` — full suite (17 tests).
- `make smoke` — failure-contract tests only.
- Original unmodified baseline: `py -m evaluator.local_evaluator`.
- Catalog download/verify/unpack: see `docs/plan/RECON.md` V4 (already done here).

## Frozen `src/contracts.py` (verbatim; changes need unanimous team agreement)

```python
ASK_ATTRIBUTES = ("category","material","color","size","style","brand","budget","feature","use_case","other")
@dataclass(frozen=True)
class ProductMeta:
    title: str; price: float | None; categories: list[str]; features: list[str]
    description: list[str]; store: str | None; details_brand: str | None
    average_rating: float; rating_number: int
@dataclass(frozen=True)
class Candidate:
    parent_asin: str; score: float; route: str; meta: ProductMeta
@dataclass(frozen=True)
class RetrievalRequest:
    canonical_query: str; intent: str; hard_filters: dict; soft_prefs: dict; top_k: int
@dataclass
class SessionState:
    session_id: str; turn: int; intent: str; slots: dict; slot_turn_added: dict
    asked_attributes: list[str]; negatives: list; canonical_query: str; history: list; profile: dict
@dataclass(frozen=True)
class DialogResult:
    canonical_query: str; ask_attribute: str | None; slots: dict; message: str
@dataclass(frozen=True)
class MemoryProfile:
    boosts: dict; summary: str
@dataclass(frozen=True)
class AgentResponse:
    recommendations: list[str]; message: str; ask_attribute: str | None; usage: dict
```

## Per-directory ownership

| Dir | Owner | Null does today | Replacing it means |
|---|---|---|---|
| `src/retrieval/` | R1 | BM25 baseline port, config-driven weights | Add dense route + RRF fusion behind the same `search()` signature |
| `src/ranking/` | R2 | Identity pass-through | Add cross-encoder as primary; keep `null_reranker.rerank` forever |
| `src/dialog/` | R3 | Echoes utterance, `ask_attribute=None` | Add routing/slots/override detection — `ask_attribute` is the only lever that scores (Fact 1) |
| `src/memory/` | R5 | Empty boosts/summary | Intra-session distillation only (see Decisions) |
| `src/agent.py`+`eval/` | R4 | orchestrator/harness itself | Keep every fallback pointed at the explicit `null_*` function, never the primary |

## Current scores (all actually run, 2026-08-29)

- Original baseline, 200 public sessions: `hit=0.125, mrr=0.068034, mttc=9.81, score=0.10671`
  (matches `docs/baseline_results.json`).
- Skeleton, dev-150 (`make eval`): `hit=0.133333, mrr=0.073378, mttc=9.726667, score=0.114147` —
  **exactly equal, to 6 decimals**, to the baseline run on the identical dev-150 subset. The full
  pipeline reproduces the baseline losslessly; orchestration adds zero degradation.
- Skeleton, fast-50 (`make eval-fast`): `hit=0.08, mrr=0.035, mttc=10.26, score=0.0653`.
  Wall-clock ~9s for `evaluate()`, ~13s end-to-end incl. index build — well under the 90s target.
- Holdout-50 (run once for this report only): `hit=0.1, mrr=0.052, mttc=10.06, score=0.0844`. No
  red flags vs. dev; don't re-run routinely.
- Determinism: `make eval-fast` twice — every metric field byte-identical; only
  `wall_clock_seconds` (timing, not a score) differed.
- Tests: `make test` → 17/17 pass (3 organizer + 14 ours, incl. every component failing alone and
  all at once, `reset()` under hostile/broken inputs, forced turn=15).

## Environment constraints

- No LLM API keys or org endpoint anywhere on this machine. LLM components need a runtime key +
  non-LLM fallback, since final scoring **may** run offline (`docs/submission_rules.md:59`).
- Only `numpy`+`pyyaml` installed. `sentence-transformers`/`faiss-cpu`/`torch` **not installed but
  confirmed installable** (current win_amd64/cp313 wheels; a live HF Hub download succeeded).
- Network to pypi.org/github.com/huggingface.co confirmed working.
- No `requirements.txt` yet — add one (`pyyaml` at minimum) before submission.
- Unresolved: whether the ML libs actually install cleanly end-to-end (untested); private-800
  scenario-mix counts (spec states same ratios as public, unverifiable here); whether final
  judging truly disables network or only reserves the right to.

## Data facts

- Catalog (50,000 rows): `parent_asin, title, features, description, price, categories, details,
  average_rating, rating_number, store`. **No top-level `brand`** — `details.Brand` covers only
  4.7%; `store` (99.4% present) is the closest proxy, → `ProductMeta.store`. **`price` null in
  78.9%** — `ProductMeta.price` is `float | None`, never hard-filter without a fallback.
  `categories` (breadcrumb list) is the only field always present and non-empty.
- Sessions (200: 80/80/30/10 buying/browsing/override/boundary): conversational content is **not
  stored** — derived at eval time from the target product's fields, seeded by
  `sample_id`+`scenario_type`. Only real slot vocabulary shipped: `user_profile.preference_tags`
  (`fit`, `material`, `comfort`, `style`, `durability`, `performance`, `warmth`, `weather`).
  `purchase_frequency` is constant across all 200 — not a useful signal.
- Dev/holdout split (`eval/dev_holdout_split.json`, committed, seed=42): 150/50, stratified so
  both halves keep the 40/40/15/5 scenario mix.

## Decisions

Full justification in `docs/plan/FEASIBILITY.md`.

| Component | In/Out | One-line reason |
|---|---|---|
| Dual-track Buy/Browse router | **In** | `scenario_type` never sent to the agent — inferred either way. |
| BM25 + dense + RRF fusion | **In** | BM25 exactly matches baseline through the full pipeline; embedding libs downloadable. |
| Structured (category) route | **In, category-only** | `categories` always populated; `brand`/`price` become soft signals only. |
| Cross-encoder rerank | **In, primary ranker** | Wheels available; still needs a real install + smoke test. |
| LLM listwise rerank | **In, optional, off by default** | Zero API keys on this machine; gate behind config + key check. |
| Dialogue state + override detection | **In** | Override turns are a fixed literal message — regex detection suffices. |
| Clarification policy | **In, `ask_attribute`-only** | Simulator never reads `message` — only the enum choice scores. |
| Cross-session long-term memory | **Out** | No user ID anywhere in the contract to key a store by. |
| Intra-session distillation | **In** | Fully supported by state the agent already owns privately. |

## Do not do this

- Delete a `null_*.py` implementation after a real component replaces it — they're permanent.
- Let a component import another component — only `src/agent.py` imports across the graph.
- Tune anything on `eval/dev_holdout_split.json`'s `holdout` list — dev-only for iteration.
- Hard-code a behavioral tunable in `src/` instead of `config.yaml`.
- Edit `evaluator/local_evaluator.py` or `tests/test_evaluator.py` — reuse via `eval/` instead.
- Open a shared resource (DB, file handle, model) without checking thread-safety against
  `agent.py`'s `ThreadPoolExecutor` worker thread — see the SQLite gotcha above.
