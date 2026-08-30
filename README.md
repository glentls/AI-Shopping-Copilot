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
| TechnicalScore | **0.8553** |
| Hit Rate@10 | **0.9850** |
| MRR | **0.6547** |
| MTTC | **2.68** |
| Reported tokens | **0** |

The three misses are assigned turn 11 by the evaluator. Intent-override hits
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
  -> BM25 + MiniLM dense + exact-phrase retrieval
  -> reciprocal-rank fusion and confidence-weighted soft reranking
  -> remove products already shown under this intent
  -> choose a clarification from the eligible candidates
  -> return up to 10 ranked ASINs with an explanatory response
```

- **Understanding:** phrase and pattern extraction maps customer language into
  category, material, color, size, style, brand, budget, feature, and use-case
  slots. It handles canonical synonyms, clause-scoped negation, explicit
  boundaries, and budget ranges.
- **Retrieval:** SQLite FTS5 BM25 runs concurrently with a local 384-dimensional
  MiniLM ONNX encoder. Exact catalog clauses form a third route. Reciprocal-rank
  fusion combines ranks rather than incomparable raw scores.
- **Reranking:** live constraints receive confidence-weighted bonuses; negated,
  retracted, and over-budget evidence receives soft penalties. Constraints do
  not hard-delete products.
- **Dialogue policy:** expected information gain scores concrete questions.
  The `other` wildcard is valued from what it has actually yielded. Two
  refusals retire it even when another question occurred between them, and the
  agent stops asking once all remaining topics are exhausted.
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
- `TJ_OTHER_BASELINE`, `TJ_OTHER_PRIOR_YIELD`, `TJ_OTHER_PRIOR_STRENGTH`, and
  `TJ_DECLINE_PATIENCE`: question-policy controls.

## Repository map

```text
starter/agent.py              evaluator entry point and turn orchestration
src/extract.py                customer constraint extraction
src/attributes.py             catalog attribute table
src/retrieval/                BM25, MiniLM dense retrieval, fusion, reranking
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
  are less structured, so wildcard priors and extraction coverage should be
  re-evaluated on natural-language conversations.
- Full fused setup needs a one-time model/runtime download and roughly 484 MiB
  of generated artifacts in the current development environment.

Competition rules and submission requirements are in
[`docs/competition_specification.md`](docs/competition_specification.md) and
[`docs/submission_rules.md`](docs/submission_rules.md). Data is derived from
Amazon Reviews 2023 by McAuley Lab, UCSD; read
[`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md) before redistribution.
