# Phase 0 Recon — Verified Findings

All answers below are grounded in code read or commands actually executed in this environment on
2026-08-29. Every claim has a `file:line` citation or a command output. Nothing here is inferred
from `docs/plan/strategy.md` or `docs/plan/architecture.md` — those were not opened during this
phase.

---

## E1. Fixed script vs. user simulator?

**It is a deterministic, code-driven user simulator that reacts only to the agent's structured
`ask_attribute` field — it never reads or reacts to the agent's free-text `message`.**

The loop, `evaluator/local_evaluator.py:238-268`:

```python
for turn in range(1, MAX_TURNS + 1):
    try:
        response = agent.respond(session_id, user_message, turn, TOP_K)
    except Exception:
        response = {"message": "", "ask_attribute": None, "recommendations": []}
    ...
    ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
    if override_applied and target in ranked:
        best_rank = ranked.index(target) + 1
        hit_turn = turn
        break
    if turn == MAX_TURNS:
        break
    override = effective_sample.get("behavior", {}).get("override") or {}
    if not override_applied and turn + 1 == int(override.get("turn", 3)):
        override_applied = True
        ...
        user_message = str(override.get("message", ...))
    else:
        user_message, boundary_used = customer_reply(
            effective_sample, response.get("ask_attribute"), disclosed, boundary_used
        )
```

`customer_reply` (`evaluator/local_evaluator.py:166-185`) takes `ask_attribute` as its only
signal from the agent's response — it classifies which pre-computed constraint to reveal next by
matching `ask_attribute` against a fixed vocabulary (`ALLOWED_ATTRIBUTES`,
`local_evaluator.py:17-20`). It never inspects `response["message"]` text. Confirmed by the
schema note in `docs/competition_specification.md:61`: *"the simulator uses this field instead of
guessing from prose."*

So: the customer's reply content is not fixed in advance (it depends on which `ask_attribute` the
agent asks for, and in what order), but the simulator's *policy* is fully deterministic given the
sample — same `ask_attribute` sequence always produces the same reply text, seeded via
`random.Random(f"{sample_id}\0{scenario_type}")` (`local_evaluator.py:210-212`). There is no
LLM or paraphrasing in the simulator; asking a natural-language question does not extract more
information than asking a bare attribute name would.

## E2. Scored every turn, or only the final turn?

**Every turn.** The hit check `if override_applied and target in ranked` (`local_evaluator.py:252`)
runs inside the per-turn loop and breaks on first hit. `best_rank`/`hit_turn` record the first
turn the target appears in the returned top-10, not just turn 10.

## E3. Exact scoring formula and constants

From `evaluator/local_evaluator.py:278-280` (executable) and confirmed identical in
`README.md:78-80`, `docs/competition_specification.md:69-75`, `docs/evaluation_config.json:1-15`:

```
HitRate@10   = hits / N
MRR          = mean(1/rank), 0 for a miss                    (local_evaluator.py:191-192, 275)
MTTC         = mean(first_hit_turn), 11 for a miss            (local_evaluator.py:193-195)
Efficiency   = clip((11 - MTTC) / 10, 0, 1)                    (local_evaluator.py:279)
TechnicalScore = 0.50 * HitRate@10 + 0.30 * MRR + 0.20 * Efficiency   (local_evaluator.py:280)
```

Constants (`docs/evaluation_config.json:1-15`): `top_k=10`, `max_turns=10`, `miss_turn_value=11`,
weights `{hit_rate_at_10: 0.5, mrr: 0.3, efficiency: 0.2}`.

Verified against the shipped reference: running the unmodified BM25 starter reproduces
`docs/baseline_results.json` exactly (see V4).

## E4. Malformed response / exception handling

Empirically tested (`respond()` failures are caught; `reset()` failures are NOT):

| Condition | Result |
|---|---|
| `respond()` raises every turn | Caught at `local_evaluator.py:239-242`, replaced with `{"message": "", "ask_attribute": None, "recommendations": []}`. That turn scores as a miss; the loop continues to the next turn. Full session: `hit=False, mrr=0.0, mttc=11`. Verified live: see test output below. |
| `respond()` returns a non-dict, or `message` isn't a `str` | Same fallback path, `local_evaluator.py:243-244`. Verified live (`respond()` returning a bare string). |
| `recommendations` is not a list (e.g. a string) | `normalize_recommendations` returns `[]` immediately (`local_evaluator.py:96-97`). Verified live. |
| Fewer than K items / invalid or duplicate `parent_asin` | Silently filtered/deduped/truncated in `normalize_recommendations` (`local_evaluator.py:95-109`); never an error, just fewer scorable candidates. |
| **`reset()` raises** | **NOT caught.** `local_evaluator.py:228` calls `agent.reset(...)` with no try/except. Verified live: this raises out of `evaluate()` and **aborts the entire evaluation run**, not just one session. |

Live verification (`scratchpad/test_e4_failure.py`, run 2026-08-29):
```
Case 1 (respond() always raises):        hit_rate=0.0, mrr=0.0, mttc=11.0  (no crash)
Case 2 (reset() raises):                 CRASHED with ValueError: reset boom
Case 3 (recommendations = "not-a-list"): hit_rate=0.0, mrr=0.0, mttc=11.0  (no crash)
Case 4 (respond() returns a bare str):   hit_rate=0.0, mrr=0.0, mttc=11.0  (no crash)
```

**Implication for our own agent:** `reset()` must never be allowed to raise — the harness gives it
zero protection. `respond()` failures only cost that one turn/session, not the whole run.

## E5. Required interface and state ownership

`docs/agent_api_contract.json:24-68`, mirrored in `README.md:52-65` and
`docs/competition_specification.md:44-56`, and physically implemented at `starter/agent.py:35-102`:

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None: ...
    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": str,
            "ask_attribute": "category"|"material"|"color"|"size"|"style"|"brand"|"budget"|"feature"|"use_case"|"other"|None,
            "recommendations": [{"parent_asin": str, "score"?: number}, ...],  # <=100 items, only first 10 valid unique scored
            "usage": {"prompt_tokens": int>=0, "completion_tokens": int>=0},   # optional
        }
```

Entry point: a module exporting a class literally named `Agent` with this interface
(`docs/submission_rules.md:18-32`: *"one Python agent entry file exporting `Agent`"*). The local
harness imports it as `from starter.agent import Agent` (`local_evaluator.py:12`) and constructs
it as `Agent(args.catalog)` (`local_evaluator.py:306`) — i.e. the constructor also receives the
catalog path in the local harness; the contract itself does not fix `__init__`'s signature.

**State is carried entirely by the Agent instance**, not the harness. `respond()` only receives
`session_id`, the latest `user_message`, `turn`, and `top_k` — never prior turns, prior
recommendations, or prior `ask_attribute`s. `reset()` is called once per session
(`local_evaluator.py:228`) to hand over `user_profile`; everything else the agent needs (dialogue
history, disclosed constraints, candidate set) must be kept in `self`, keyed by `session_id`. The
starter agent does this minimally via `self._sessions: set[str]` (`starter/agent.py:41,73-75`).

---

## D1. Catalog

- Path: `data/catalog.jsonl` (not shipped in the repo; downloaded during this session — see V-section).
- Format: JSON Lines, one product object per line.
- Row count: **50,000** (verified: `wc -l` and JSON parse count both = 50000).
- Fields (union over all 50,000 rows, no field is ever absent from a row):

| field | dtype(s) observed | count None | count empty (`""`/`[]`/`{}`) |
|---|---|---|---|
| `parent_asin` | `str` (50000) | 0 | 0 |
| `title` | `str` (50000) | 0 | 2 |
| `features` | `list` (50000) | 0 | 5,219 |
| `description` | `list` (50000) | 0 | 23,887 |
| `price` | `NoneType` (39473), `float` (10410), `str` (117) | 39,473 | n/a |
| `categories` | `list` (50000) | 0 | 0 |
| `details` | `dict` (50000) | 0 | 1,670 |
| `average_rating` | `float` (50000) | 0 | n/a |
| `rating_number` | `int` (50000) | 0 | n/a |
| `store` | `str` (49686), `NoneType` (314) | 314 | n/a |

Participant-visible field list matches `docs/competition_specification.md:17` exactly.

**Fields the plan docs may assume that need flagging:**
- **`brand` does not exist as a top-level field at all.** The closest analogue is nested inside
  `details`: `details.Brand` appears in only **2,328/50,000 rows (4.7%)**, `details["Brand Name"]`
  in 610/50,000 (1.2%). `store` (a seller/storefront name, not necessarily the product brand) is
  present in 49,686/50,000 (99.4%) and is the only near-universal brand-like signal.
- **`price` is null in 39,473/50,000 rows (78.9%)** — a majority-missing field. An additional 117
  rows have `price` as a string, and those strings are mostly the placeholder `"—"` (an em dash,
  effectively another null) or free text like `"from 12.99"` requiring parsing.
- **There is no single "category path" string field.** `categories` is a list acting as a
  breadcrumb, e.g. `["Clothing, Shoes & Jewelry", "Women", "Jewelry", "Earrings", "Hoop"]`. It is
  never null/empty, so it's the most reliable structured signal in the catalog.
- `features` is empty in 10.4% of rows; `description` is empty in nearly half (47.8%).
- Verified via `scratchpad/analyze_catalog.py`, run 2026-08-29.

Common `details` sub-keys and counts (top ones): `Date First Available` 46886, `Department` 43582,
`Item model number` 27729, `Package Dimensions` 27061, `Manufacturer` 23512,
`Is Discontinued By Manufacturer` 13070, `Product Dimensions` 10210, `Item Weight` 3243,
`Color` 2439, `Brand` 2328, `Material` 2069, `Style` 1752.

## D2. Sessions

- Path: `data/public_set.jsonl`. Format: JSON Lines. Count: **200** (verified).
- One full example (verbatim, `data/public_set.jsonl:1`):

```json
{"category_bucket": "clothing", "difficulty_bucket": "easy", "ground_truth": {"parent_asin": "B09PYB7B6Z"}, "sample_id": "public_0001", "scenario_type": "buying", "user_profile": {"average_prior_rating": 5.0, "preference_tags": ["fit", "comfort", "durability"], "purchase_frequency": "3-4 prior purchases", "rating_style": "usually positive", "summary": "Prior purchases emphasize fit, comfort, durability; ratings are usually positive."}}
```

Top-level keys (union, all 200 rows): `category_bucket`, `difficulty_bucket`, `ground_truth`,
`sample_id`, `scenario_type`, `user_profile`. `user_profile` keys: `average_prior_rating`,
`preference_tags`, `purchase_frequency`, `rating_style`, `summary`. `ground_truth` has exactly one
key: `parent_asin`.

**Public sessions ship zero `intent_card` / `behavior` fields** (verified: 0/200 rows have either
key). These — and therefore the opening customer utterance, the hard/soft constraint list, and the
override text — are **derived at evaluation time** from the catalog product referenced by
`ground_truth.parent_asin`, via `intent_card()` and `behavior_for()`
(`local_evaluator.py:52-87`), seeded deterministically by `sample_id` + `scenario_type`
(`local_evaluator.py:210-212`). This is why local scoring is reproducible without a stored
conversation log.

## D3. 15 real sessions — slots, overrides, turn distribution

Scenario mix (all 200, matches `data/README.md:5`): `buying` 80, `browsing` 80,
`intent_override` 30, `boundary` 10. `difficulty_bucket`: `medium` 90, `easy` 80, `hard` 30.
`category_bucket` is constant `"clothing"` for all 200 rows. `purchase_frequency` is constant
`"3-4 prior purchases"` for all 200 rows (no variation — likely not a useful discriminating slot).

The only slot vocabulary that actually appears *in the shipped session file* is
`user_profile.preference_tags` (frequency across all 200 sessions, from
`scratchpad/analyze_sessions.py`):

```
fit: 163   material: 154   comfort: 144   style: 101   durability: 47
performance: 26   warmth: 18   weather: 12   general shopping: 1
```

`rating_style`: `usually positive` 134, `critical` 45, `mixed` 21.

The *conversational* attribute vocabulary (`ask_attribute` values the agent may use, and the
simulator's constraint classifier) is fixed in code, not data:
`category, material, color, size, style, brand, budget, feature, use_case, other`
(`local_evaluator.py:17-20`, `docs/agent_api_contract.json:42`).

**Verbatim intent-override examples** (regenerated live from real catalog products via
`materialize_hidden_fields`, `scratchpad/turn_dist_and_override_example.py`):

- `public_0002` (target `B071X54486`, a leather belt): opening turn-1 message —
  `"I'm looking for Accessories Belts. Buckle closure"`; override fires on turn 3 —
  `"Actually, ignore my earlier preference. What I need is: leather."`
- `public_0003` (target `B09YMTWDXJ`, a Casio watch): opening —
  `"I'm looking for Watches Wrist Watches. Stainless Steel Band"`; override turn 3 —
  `"Actually, ignore my earlier preference. What I need is: Water Resistant."`
- `public_0004` (target `B07C2XPZ6D`, a camisole top): opening —
  `"I'm looking for Tops & Tees Tanks & Camis. Long torso camisole for extra coverage with spagetti adjustable strap for perfect fit"`;
  override turn 3 — `"Actually, ignore my earlier preference. What I need is: polyester."`

All override turns observed for `intent_override` sessions were **turn 3** in these three
examples; code allows turn 3 or 4 (`rng.choice([3, 4])`, `local_evaluator.py:83`) — both are
possible depending on the per-sample RNG draw.

**Turn-count distribution** (first_hit_turn, observed by actually running the shipped BM25
baseline over all 200 public sessions — see V4 for the run):

```
turn 1:     21 sessions (10.5%)
turn 4:      4 sessions (2.0%)
miss (11): 175 sessions (87.5%)
```

Note this is the *baseline's* empirical distribution, not a property of the data itself — a
better agent would shift mass toward earlier turns and away from misses. It's included here as
the only concrete "turn distribution" obtainable without guessing, since turn numbers don't exist
in the static session file (they're a runtime artifact of a given agent's performance).

## D4. Gold/target representation

`ground_truth.parent_asin` (`data/public_set.jsonl`, e.g. line 1: `"ground_truth": {"parent_asin": "B09PYB7B6Z"}`)
is the only gold signal, and it is fixed for the whole session — it is not turn-specific. It is
also the seed for everything else about the session: `materialize_hidden_fields`
(`local_evaluator.py:204-213`) looks up this exact catalog row and derives the opening utterance,
hard/soft constraints, and (for override sessions) the override text from that one product's
`title`/`features`/`details`/`price`. So the "opening utterance" is not independent of the target —
it is a lossy, regex-extracted paraphrase of the target product's own catalog fields
(`intent_card()`, `local_evaluator.py:52-71`, using `MATERIAL_RE`/`COLOR_RE` and the first ~2
cleaned feature/detail strings). Only exact `parent_asin` string equality counts as a hit
(`docs/competition_specification.md:40`: *"Hits are always exact code matches."*).

---

## V1. Python, deps, lockfile, network

- Python: **3.13.14** (`py --version`; the `python`/`python3` commands are not on PATH on this
  machine — only the `py` launcher is; use `py` for all commands).
- No lockfile, no `requirements.txt`, no `pyproject.toml` anywhere in the repo (verified via
  recursive glob).
- Installed packages: only `numpy 2.4.3` (verified: `py -m pip list`). No `sentence-transformers`,
  `faiss`, `torch`, `transformers`, `scikit-learn`, `openai`, or `anthropic` installed.
- Network access: **confirmed working.** `curl -sI https://pypi.org` → `200 OK`;
  `curl -sI https://github.com` → `200 OK`; `curl -sI https://huggingface.co` → `200 OK`; the
  GitHub Releases API and asset-download URLs are reachable and were used to fetch the catalog
  (see V3/D1). No corporate proxy or firewall observed blocking any of these hosts.

## V2. sentence-transformers / FAISS / cross-encoder availability

Not installed, but **downloadable and installable right now** — actually tested, not assumed:

- Installed `huggingface_hub` (a ~small pure-Python package) via
  `py -m pip install --target <scratch> huggingface_hub` — succeeded.
- Used it to actually download `config.json` from
  `sentence-transformers/all-MiniLM-L6-v2` on the Hub — succeeded, file exists and is valid JSON
  (`scratchpad/test_hf_download.py` output, 2026-08-29). Model downloads work end-to-end from this
  machine.
- Confirmed via the PyPI JSON API that current wheels exist for this exact interpreter:
  `faiss_cpu-1.15.0-cp313-cp313-win_amd64.whl`,
  `torch-2.13.0-cp313-cp313-win_amd64.whl` (122.1 MB), and `sentence-transformers==6.0.0`
  (`requires_python >= 3.10`, satisfied).
- **Not actually installed** (torch + sentence-transformers is a large download,
  hundreds of MB to ~1GB+); that install was deliberately not performed during read-only recon,
  but there is no evidence it would fail. Windows-specific caveat observed: `huggingface_hub`'s
  local disk cache warns it can't create symlinks on this machine (dev mode / admin not enabled) —
  it falls back to a "degraded" (duplicate-file) cache automatically; this is a disk-space
  nuisance, not a functional blocker. Set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to silence it.
- Default HF cache location on this machine (unless overridden): `~/.cache/huggingface/hub` i.e.
  `C:\Users\tohji\.cache\huggingface\hub` (standard `huggingface_hub` default; confirmed by
  passing an explicit `cache_dir` successfully in the test, implying the unset default resolves
  under the user profile the same way on any machine).

## V3. LLM API keys / org endpoint

**None.** Checked process environment variables for `API|KEY|OPENAI|ANTHROPIC|TOKEN|LLM|AZURE|COHERE`
— no matches. Checked for `.env*` files and any config file anywhere in the repo — none found.
`docs/competition_specification.md:91-93` and `README.md:86-88` both state explicitly that teams
must supply and manage their own model credentials and that the organizer does not provide any.
**There is no usable LLM API access on this machine right now**; any LLM-based component needs a
key supplied later (e.g., via environment variable at run time) or must have a working non-LLM
fallback, since `docs/submission_rules.md:57-63` also warns official final scoring may run with
network disabled entirely.

## V4. Baseline: how to run it, and its score

Command (from `techjam-conversational-search/`): `py -m evaluator.local_evaluator`
(the README's `python3 -m evaluator.local_evaluator` doesn't resolve on this machine — no
`python3` on PATH — use `py` instead). Requires `data/catalog.jsonl` to exist (download +
decompress `catalog.jsonl.gz` first; see below).

**Catalog was not shipped in the repo or in the fork's own GitHub Releases** (the fork
`jireh0108/techjam-conversational-search` has zero releases — `gh api` returned `[]`). It lives on
the **upstream/parent repo's** release instead: `TechJam2026/techjam-conversational-search`,
release tag `participant-kit`, asset `catalog.jsonl.gz`
(`https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz`).
Downloaded, SHA-256 checksum verified against the release's `SHA256SUMS` asset
(`07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8` — matched exactly), decompressed
to `data/catalog.jsonl` (50,000 lines, matches expected row count).

Actually ran `py -m evaluator.local_evaluator` end-to-end. Result (written to `results.json`,
printed to stdout):

```
hit_rate_at_10: 0.125
mrr:            0.068034
mttc:           9.81
efficiency:     0.119
recommended_technical_score: 0.10671
```

This **exactly matches** the shipped reference in `docs/baseline_results.json:1-11` — the baseline
is fully reproducible on this machine. Wall-clock time for the full 200-session run: **~34.5s**
(`time py -m evaluator.local_evaluator` → `real 0m34.492s`). Per-scenario breakdown (from the same
run): `buying` hit_rate 0.2375/mrr 0.1265/mttc 8.625; `browsing` hit_rate 0.025/mrr 0.0045/mttc
10.75; `intent_override` hit_rate 0.1333/mrr 0.1042/mttc 10.07; `boundary` hit_rate 0.0/mrr
0.0/mttc 11.0 (zero hits at all on the 10 boundary sessions).

## V5. Makefile / CI / test runner / seeds / determinism

- **No Makefile.** No `.github/` directory / CI config. No `.yml`/`.yaml` files anywhere in the
  repo (verified via recursive glob across the whole repo tree).
- Test runner: standard library `unittest`. Command: `py -m unittest discover -s tests -v`. Ran
  it — 3 tests, all pass, in 0.011s (`tests/test_evaluator.py`).
- Determinism: the simulator seeds its per-sample RNG deterministically —
  `random.Random(f"{sample.get('sample_id','')}\0{sample.get('scenario_type','')}")`
  (`local_evaluator.py:210-212`) — so the same sample always produces the same intent card,
  constraints, and override turn/text across runs. `session_id` itself is a fresh random UUID each
  run (`local_evaluator.py:227`, `uuid.uuid4().hex`) but it's only ever used as a dict key by the
  agent, never as a scoring input, so it doesn't affect metrics. Verified live: ran the baseline
  evaluator twice end-to-end and diffed the full aggregate-metrics JSON (everything except the
  `sessions` list, which differs only in the random `session_id`s embedded... actually not even
  that, since `sample_id` not `session_id` is stored per-session) — **byte-identical** across both
  runs.

---

## Known Unknowns

- Whether `sentence-transformers`, `faiss-cpu`, and `torch` actually install and run correctly on
  this machine end-to-end (only their PyPI wheel *availability* for this interpreter/OS and a
  small `huggingface_hub` file download were verified — the full multi-hundred-MB install itself
  was not attempted during read-only recon).
- Whether the private 800-session holdout set follows the same scenario-mix proportions
  (40/40/15/5%) as the public 200 — `docs/competition_specification.md:23-29` states the "same
  fixed scenario mix" applies to "both splits," so this is actually stated, not unknown — but the
  *exact* per-scenario counts for the private 800 are not observable from this machine.
  UNVERIFIED beyond the stated ratio.
  - **Correction: this is not unknown** — the spec states it directly. Leaving this line only to
    flag that we cannot independently re-verify it (no access to the private set).
- Whether official final-round scoring disables network access entirely, or only "may" do so —
  `docs/submission_rules.md:59` says "organizer policy **may** disable network access" — not
  confirmed either way, must be treated as **possible** (design for an offline fallback).
- Exact default `sentence-transformers`/`huggingface_hub` cache path on the machine that will
  actually run final judging (this machine's default was inferred, not the judging machine's).
- Whether `Agent.__init__` accepts a `catalog_path` argument as a *contractual* requirement, or
  whether that's just how the local harness happens to construct it — `docs/agent_api_contract.json`
  only specifies `reset`/`respond`, not `__init__`, so the constructor signature is genuinely
  ours to choose as long as `starter.agent.Agent` (or our replacement) stays importable.
