# CLAUDE.md

Project context for Claude Code. Read this before touching anything.

## What this is

TechJam 2026 Track 4: Shopping Copilot — a conversational e-commerce search agent
scored by a local evaluator against 200 public sessions. Hidden target product per
session; agent has at most 10 turns to surface it in a Top-10 list.

Imported from `TechJam2026/techjam-conversational-search`. Repo is private during
development and must be flipped to public before the Devpost deadline.

## Hard constraints — do not violate

- **Never modify `evaluator/`, `data/public_set.jsonl`, or `docs/evaluation_config.json`.**
  Local scores reported with a modified evaluator are invalid.
- **Never commit API keys.** Secrets go in `.env`, which is gitignored. If you need a
  key, read it via `os.environ`, never inline.
- **Never commit `data/catalog.jsonl`.** It's ~50k rows and covered by
  `DATA_ATTRIBUTION.md` redistribution terms.
- The catalog is read-only. No mutations, no injected ASINs.
- Hard limit of 10 turns per session — exceeding it scores zero. Enforce our own
  stop at turn 10 defensively.
- No UI work. Backend and headless pipelines only.
- Everything runs in-memory. No external vector DB clusters.
- Text only. No multi-modal.
- No fine-tuning of base LLMs.

## The scoring function — optimise this, not the brief's vibes

```
TechnicalScore = 0.50 x HitRate@10 + 0.30 x MRR + 0.20 x Efficiency
Efficiency     = clip((11 - MTTC) / 10, 0, 1)
MTTC           = mean first-hit turn; a MISS is assigned turn 11
```

Only exact `parent_asin` equality counts as a hit. No partial credit.

**Key derived insight.** Because misses are scored as turn 11, if we return a Top-10
on *every* turn from turn 1, then MTTC ~= 11 - 10h and Efficiency ~= h. The score
collapses to roughly:

```
TechnicalScore ~= 0.70 x HitRate@10 + 0.30 x MRR
```

Efficiency stops being an independent axis we can trade against. Consequences:

1. **Always return recommendations.** The API contract permits a clarification
   question and a ranked list in the same response. A turn that asks without
   recommending throws away a free shot at the target. There is no penalty for a
   wrong list.
2. **Recall is the ceiling.** We cannot rerank an item we never retrieved. Track
   Recall@100 as the internal north star during the retrieval phase, not Hit@10.
3. **The open question is *which* attribute to ask, not whether to ask.**
   Highest-information-gain slot selection from the fixed enum is where the
   differentiation lives.

VERIFY ALL OF THIS against `evaluator/local_evaluator.py` before building on it.
The above is inferred from the README. Record findings in `docs/evaluator_notes.md`.

## Baseline reference

Weak BM25 starter on the 200 public sessions:

| Metric | Value |
|---|---|
| Hit@10 | 0.125 |
| MRR | 0.068034 |
| MTTC | 9.81 |
| TechnicalScore | ~0.107 |

Implied mean hit turn among hits: ~1.5. The baseline hits on turn one or never —
it is failing at retrieval, not at conversation.

Stored in `baseline_reference.json`. Every change gets measured against it.

## Agent contract

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None: ...

    def respond(self, session_id: str, user_message: str,
                turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [{"parent_asin": "B000..."}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30},
        }
```

`ask_attribute` must be one of: `category`, `material`, `color`, `size`, `style`,
`brand`, `budget`, `feature`, `use_case`, `other`, or `null`.
Full contract in `docs/agent_api_contract.json`.

## Architecture we are building

1. **Retrieval (highest priority).** Three routes over the 50k catalog — BM25,
   dense vectors (CPU-friendly small embedding model), structured metadata filters
   on category/price/brand — fused with Reciprocal Rank Fusion.
2. **Ranking.** LLM reranks top-50 down to top-10.
3. **Dialog state.** Slot state machine handling incremental accumulation and abrupt
   intent override (erase-and-rewrite, e.g. "actually, boots not sneakers").
4. **Question policy.** Pick the slot with highest expected information gain given
   the current candidate pool. Always piggyback the Top-10.

## Working rules for this repo

- **Every change gets an ablation number.** Run the evaluator, record the delta.
  The target artifact is a table: BM25 baseline -> +dense -> +RRF -> +LLM rerank ->
  +dialog policy, with Hit@10 / MRR / MTTC at each step. Maintain it in
  `docs/ablations.md` as we go, not at the end.
- **Determinism.** Temperature 0 on all LLM calls, cache and seed everything.
  Two runs of the evaluator must produce identical scores.
- **Graceful degradation.** The agent must fall back to pure hybrid retrieval if the
  LLM errors, times out, or no API key is present. A judge cloning this repo without
  credentials must still get results. Test this path explicitly.
- Read `docs/submission_rules.md` and `docs/participant_release_checklist.md` before
  changing repo structure. Those override anything here.
- Prefer small, measurable commits over large refactors.

## Deliverables (do not leave to the last two hours)

- Devpost writeup: tools, APIs, libraries, datasets used
- Public repo with README: overview, setup, reproduction steps, limitations, team
  contributions
- YouTube demo video, public visibility, linked from Devpost
- Disclosed model choice, estimated cost, token usage, latency
