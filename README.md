# TechJam Conversational Product Search

A deterministic, offline shopping agent for the TechJam Conversational
E-Commerce Search Challenge. It combines structured preference extraction,
hybrid retrieval, adaptive clarification, and current-intent recommendation
deduplication to search a frozen 50,000-product Amazon catalog over at most ten
conversation turns.

## Current public-set result

Measured on all 200 released sessions with the untouched local evaluator and
the full fused index:

| Metric | Result |
|---|---:|
| TechnicalScore | **0.8601** |
| Hit Rate@10 | **0.9900** |
| MRR | **0.6618** |
| MTTC | **2.670** |
| Reported tokens | **0** |

The two remaining misses are assigned turn 11 by the evaluator. Intent-override hits
cannot count before the override arrives on turn 3 or 4. Run the benchmark
commands below to reproduce the aggregate and per-scenario results.

## Quick start

Python 3.10 or later is recommended.

1. Download `catalog.jsonl.gz` and `SHA256SUMS` from this repository's GitHub
   Release, verify the checksum, and place the decompressed file at
   `data/catalog.jsonl`:

   ```bash
   shasum -a 256 -c SHA256SUMS
   gzip -dk catalog.jsonl.gz
   mv catalog.jsonl data/catalog.jsonl
   ```

2. Build the attribute table, SQLite FTS index, pinned MiniLM model, and dense
   vectors:

   ```bash
   python3 -m tools.build_index
   ```

   The first full build downloads the pinned
   `sentence-transformers/all-MiniLM-L6-v2` ONNX export and installs
   `onnxruntime==1.22.1` plus NumPy under the gitignored `artifacts/_vendor`
   directory. It does not modify the global Python environment. Later builds
   reuse those files.

3. Run the official local evaluator or the richer benchmark report:

   ```bash
   python3 -m evaluator.local_evaluator
   python3 -m tools.bench
   python3 -m tools.bench --verify
   ```

   The evaluator writes `results.json`. `--verify` checks that the diagnostic
   replay loop still agrees exactly with the official evaluator.

For a fast, dependency-light development build, use
`python3 -m tools.build_index --skip-dense`. The agent automatically falls back
to BM25 when dense artifacts or dependencies are absent, though the fused route
scores better.

## How it works

Each turn follows one deterministic pipeline:

```text
customer message
  -> extract and update current preference state
  -> compile a Buying/Browsing context program from state + profile
  -> BM25 + Porter + MiniLM dense + exact-phrase retrieval
  -> route-aware fusion, semantic ranking, and guarded constraint locking
  -> remove products already shown under this intent
  -> cut off overloaded broad pools and choose a clarification
  -> return up to 10 ranked ASINs with an explanatory response
```

- **Understanding:** phrase and pattern extraction maps customer language into
  category, material, color, size, style, brand, budget, feature, and use-case
  slots. It handles canonical synonyms, clause-scoped negation, explicit
  boundaries, and budget ranges.
- **Retrieval:** SQLite FTS5 BM25 runs concurrently with a local 384-dimensional
  MiniLM ONNX encoder. Porter morphology and exact catalog clauses add two more
  routes. Reciprocal-rank fusion combines ranks rather than incomparable raw
  scores; MiniLM provides model-based semantic ranking without an API call.
- **Dynamic routing:** every turn compiles a fresh context program. Explicit
  requirements select the Buying track; exploratory category-only requests use
  the Browsing track, a semantic/profile prior, and a 200-candidate processing
  cutoff. Supplying a concrete detail switches the next turn to Buying.
- **Reranking:** live constraints receive confidence-weighted bonuses; negated,
  retracted, and over-budget evidence receives soft penalties. Explicit Buying
  requirements lock the top ten only when catalog metadata supplies ten full
  matches; otherwise the ranker backs off without dropping recall.
- **Dialogue policy:** expected information gain scores concrete questions.
  The open-ended `other` action starts on the same scale, decays after each use,
  pauses after an unproductive answer, and retires after repeated silence or
  refusals. The agent stops asking once all remaining topics are exhausted. An
  overloaded Browsing pool triggers narrowing guidance only while that action
  remains eligible.
- **Personalized context:** safe aggregate `preference_tags` enrich the semantic
  query and contribute a small Browsing-only rank prior among already relevant
  candidates. Live customer constraints always take precedence over profile
  history.
- **Intent handling:** “ignore my earlier preference” retires the replaceable
  opener preference even when the new value belongs to another slot, while
  preserving the category and valid constraints learned later. An override
  starts a new recommendation epoch.
- **No repeats:** a continued session treats every previously returned ASIN as
  implicit negative feedback. Selection, question scoring, and explanations
  all use the same unseen candidate pool. The exclusion set is cleared only
  when an intent override arrives, not for every turn or for the whole session.

## Agent interface

The evaluator imports `Agent` from `starter/agent.py`:

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        return {
            "message": "Here are my closest matches. Do you prefer a material?",
            "ask_attribute": "material",
            "recommendations": [{"parent_asin": "B000..."}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
```

`ask_attribute` may be one of the nine modeled slots, `other`, or `null`. See
[`docs/agent_api_contract.json`](docs/agent_api_contract.json) for the complete
contract.

## Cost and latency

The scoring path uses deterministic parsing and makes no external API calls.
It reports zero prompt and completion tokens and has an estimated per-session
API cost of $0. Dense query encoding runs locally with the pinned MiniLM ONNX
model; after artifacts are built, evaluation requires no network access.

On the documented Apple ARM development machine, full agent initialization was
0.66 seconds. Warm full-catalog search plus reranking measured about 27.5 ms
median for natural queries and 34.5 ms median / 49.8 ms maximum for exact
catalog-style clauses. Hardware will vary. See
[`docs/lane_b_notes.md`](docs/lane_b_notes.md) for artifact sizes, build times,
latency methodology, and retrieval ablations.

## Useful development commands

```bash
python3 -m unittest discover -s tests -v
python3 -m tools.robustness
python3 -m tools.bench --failures 5
python3 -m tools.bench --only public_0003 --output ""
python3 -m tools.bench --depth
python3 -m tools.chat
```

Non-obvious runtime variables:

- `TECHJAM_ARTIFACTS`: artifact directory; defaults to `artifacts`.
- `TECHJAM_DEBUG=1`: re-raise response failures instead of serving the safe
  fallback.
- `TJ_RETRIEVAL_MODE`: `fused` (default), `bm25`, or `dense`.
- `TJ_BM25_WEIGHT`, `TJ_DENSE_WEIGHT`, `TJ_EXACT_WEIGHT`, `TJ_SLOT_WEIGHT`, and
  `TJ_BUDGET_WEIGHT`: ablation and tuning controls.
- `TJ_OPEN_QUESTION_BASELINE` and `TJ_OPEN_QUESTION_DECAY`: initial value and
  per-use discount for asking “anything else that matters?”.
- `TJ_OPEN_QUESTION_EXPECTED_YIELD`: new facts expected from a highly
  productive open-ended answer.
- `TJ_OPEN_QUESTION_MAX_CONSECUTIVE`,
  `TJ_OPEN_QUESTION_ZERO_YIELD_PATIENCE`, and
  `TJ_OPEN_QUESTION_DECLINE_PATIENCE`: hard repetition and exhaustion limits.

Question-policy settings are read from the process environment when
`src.policy.question` is imported. Configure one run inline:

```bash
TJ_OPEN_QUESTION_BASELINE=8 TJ_OPEN_QUESTION_DECAY=0.8 python3 -m tools.chat
```

For a reusable local configuration, copy
`config/question-policy.env.example` to the git-ignored `.env`, edit it, and
source it before starting the agent:

```bash
cp config/question-policy.env.example .env
source .env
python3 -m tools.chat
```

## Repository map

```text
starter/agent.py              evaluator entry point and turn orchestration
src/extract.py                customer constraint extraction
src/attributes.py             catalog attribute table
src/retrieval/                BM25, MiniLM dense retrieval, fusion, reranking
src/orchestration.py          intent routing and turn-scoped context programs
src/policy/                   state transitions, question policy, response text
tools/build_index.py          reproducible artifact builder
tools/bench.py                metrics, replay, transcripts, sweeps, verification
tools/chat.py                 interactive local conversation
tests/                        unit, integration, evaluator, and robustness tests
data/public_set.jsonl         200 labeled development sessions
evaluator/local_evaluator.py  frozen public simulator and scorer
```

## Limitations

- Preference extraction is deterministic rather than fully semantic, so unseen
  paraphrases, brands, fashion subcultures, and non-US sizing can be missed.
- Canonicalization intentionally merges nearby concepts such as navy/blue and
  water-resistant/waterproof, losing some nuance.
- Most catalog products have no price; missing price is treated as unknown, not
  over budget.
- The public simulator copies catalog constraints into replies. Real customers
  are less structured, so open-question settings and extraction coverage should be
  re-evaluated on natural-language conversations.
- The evaluator supplies an aggregate profile but no stable user identifier or
  write-back API. The agent can personalize and adapt within a session, but it
  deliberately does not invent cross-session identity or durable profile
  mutations.
- Full fused setup needs a one-time model/runtime download and roughly 484 MiB
  of generated artifacts in the current development environment.

Competition rules and submission requirements are in
[`docs/competition_specification.md`](docs/competition_specification.md) and
[`docs/submission_rules.md`](docs/submission_rules.md). Data is derived from
Amazon Reviews 2023 by McAuley Lab, UCSD; read
[`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md) before redistribution.
