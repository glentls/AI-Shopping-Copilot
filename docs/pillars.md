# TechJam Pillars — Official Framework & Implementation Notes

Team reference mapping the **official hackathon pillar language** to our codebase, metrics, and next improvements.

---

## Pillar overview (I–IV)

| Pillar | Official name | One-line summary |
|--------|-----------------|------------------|
| **I** | Multi-Route Retrieval → Ranking | Hybrid search over 50k catalog; buying/browsing paths; filters + rerank |
| **II** | Dynamic State Machine | Session slots, intent routing, override/boundary, proactive `ask_attribute` |
| **III** | Self-Evolution: Dynamic Context Programming | Distill dialog + profile into evolving context; adapt strategy each turn |
| **IV** | Evaluation Matrix: Product & Efficiency Metrics | Optimize Coverage, Precision, Efficiency on the public evaluator |

Pillars I–II are **implemented** in `starter/retrieval/` and `starter/agent.py`.  
Pillar III is **partially implemented** in `starter/personalization/` + session state in `agent.py`.  
Pillar IV is **not code** — it is the **scoring lens** we optimize via `evaluator/local_evaluator.py`.

---

## Pillar III — Self-Evolution: Dynamic Context Programming

### Official requirements

- **Runtime Adaptation:** Use accumulated dialog history for **Personalized Context Distillation** — continuously update short-term session state and long-term user profile signals.
- **Adaptive Orchestration:** Use dynamic **Context Programming** for runtime workflow re-orchestration and strategy alignment — the agent iteratively refines its own guidance logic.

### How we implement it (today)

```text
reset(user_profile)                    ← long-term profile (anonymized aggregate)
       ↓
each respond(user_message):
  1. _apply_message      → update slots, history, override/boundary state  (short-term)
  2. _route_intent       → buying ↔ browsing re-orchestration              (strategy switch)
  3. search(history, mode, filters+profile) → context distillation into query + filters
  4. retrieval_feedback → overloaded? missing_attributes?                  (closed-loop signal)
  5. profile_adjusted_priority → refine which attribute to ask next       (guidance logic)
  6. rerank + profile_boost → tie-break using profile tags + rating style
```

| Concept | Code location | What it does |
|---------|---------------|--------------|
| Short-term session state | `agent.py` — `history`, `slots`, `slot_status`, `unconstrained`, `asked`, `intent` | Grows every turn; drives `query_text` and filters |
| Long-term user profile | `agent.py` — `state["profile"]`; `starter/personalization/profile_signals.py` | Parsed once at `reset`; nudges questions and rerank |
| Context distillation | `retrieval/query_builder.py` — `strip_boilerplate`, `extract_constraint_phrases` | Turns raw dialog into FTS phrases + slot filters |
| Workflow re-orchestration | `agent.py` — `_route_intent`, override slot reset | Switches buying/browsing path; clears stale constraints on pivot |
| Guidance refinement | `agent.py` — `_choose_question` + `feedback.missing_attributes` + `profile_adjusted_priority` | Next question adapts to retrieval overload + profile |
| Retrieval feedback loop | `retrieval/feedback.py` → `state["retrieval_feedback"]` | Pillar I tells Pillar II when to clarify |

### Gaps vs full “self-evolution” (future work)

- No cross-session long-term memory (each `session_id` is isolated; profile is session-scoped from evaluator).
- No explicit “context program” DSL — distillation is heuristic (regex + BM25), not LLM rewrite.
- No learned policy — question order is rule-based, not trained.
- `get_dialog_state()` exists for orchestration handoff but no external planner consumes it yet.

### Devpost framing (copy-ready)

> **Pillar III — Dynamic Context Programming:** Each turn, we distil the full dialog history and anonymized profile into a structured search context (mode, slots, phrases). Retrieval feedback closes the loop: when the candidate pool is overloaded, the agent re-orchestrates by asking the highest-value attribute — prioritised by the user’s historical preference tags. No LLM required; strategy adapts at runtime through state + feedback.

---

## Pillar IV — Evaluation Matrix

### Official dimensions

Anchored on the final purchased record (`parent_asin` in catalog):

| Dimension | Metric | Question it answers |
|-----------|--------|---------------------|
| **Coverage** | Hit Rate@K (K=10) | Did we recall the target in the shortlist at all? |
| **Precision** | MRR / Top-K rank | When we found it, how high did we rank it? |
| **Efficiency** | MTTC (Mean Turns to Conversion) | How fast did we guide the user to a hit? |

### Mapping to our evaluator

```text
HitRate@10   → Coverage
MRR          → Precision (rank quality; #1 >> #10)
MTTC         → Efficiency (fewer turns = better)
Efficiency   = clip((11 - MTTC) / 10, 0, 1)
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
```

Run locally:

```bash
python3 -m evaluator.local_evaluator
```

Outputs aggregate + `scenario_metrics` in `results.json`.

### Current benchmark (tuned, local public set)

| Metric | Baseline | Post I+II | With Pillar III | Tuned |
|--------|----------|-----------|-----------------|-------|
| Hit Rate@10 (Coverage) | 0.125 | 0.755 | 0.855 | **0.995** |
| MRR (Precision) | 0.068 | 0.468 | 0.567 | **0.812** |
| MTTC (Efficiency) | 9.81 | 6.13 | 4.21 | **2.12** |
| TechnicalScore | 0.107 | 0.615 | 0.733 | **0.919** |

Experiment-by-experiment record, including the changes that made things worse and the
defects found on the way: [tuning_log.md](tuning_log.md).

### How to improve each Pillar IV dimension

#### Coverage (Hit Rate@10) — weight **50%**

| Lever | Module | Action |
|-------|--------|--------|
| Ask the right attribute | `agent.py` | Simulator reveals constraints matching `ask_attribute` |
| Always return recommendations | `agent.py` | Never withhold top-10 while asking |
| Boundary handling | `agent.py` | Mark refused attrs `unconstrained`; ask a different one |
| Broad recall | `retrieval/search.py` | Browsing OR-mode + BM25 fallbacks |
| Profile question order | `personalization/` | Ask high-value attrs first per `preference_tags` |

**Weakest scenario:** boundary (~50% hit).

#### Precision (MRR) — weight **30%**

| Lever | Module | Action |
|-------|--------|--------|
| Exact phrase rerank | `retrieval/reranker.py` | Boost simulator phrase overlap |
| Metadata filters | `retrieval/filters.py` | Narrow candidate pool with slots |
| Profile tie-break | `personalization/rerank_boost.py` | Small tag-based boost |
| Concise filter values | filters glue | Phrases, not full utterances |

#### Efficiency (MTTC) — weight **20%**

| Lever | Module | Action |
|-------|--------|--------|
| Profile-prioritized questions | `personalization/` | Unlock constraints earlier |
| Overload-driven clarification | `retrieval/feedback.py` | Ask when pool is too large |
| Buying fast path | `_route_intent` | AND queries on turn 1 |
| Override reset | `agent.py` | Clear stale slots on pivot |

Misses count as turn **11**.

---

## Manual testing (beyond the evaluator)

### Unit tests

```bash
python3 -m unittest tests.test_retrieval tests.test_personalization tests.test_agent_routing -v
```

### Retrieval only

```python
from starter.retrieval import HybridSearcher
r = HybridSearcher("data/catalog.jsonl").search("running shoes mesh", "browsing", {}, 10)
print(r.asins, r.feedback)
```

### Interactive agent demo

```python
from starter.agent import Agent

agent = Agent("data/catalog.jsonl")
agent.reset("demo", {
    "preference_tags": ["material", "comfort"],
    "rating_style": "usually positive",
    "summary": "Prior purchases emphasize material, comfort.",
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 5.0,
})
resp = agent.respond("demo", "I'm looking for running shoes, but I'm still exploring.", 1, 10)
print(resp["message"], resp["ask_attribute"], [x["parent_asin"] for x in resp["recommendations"][:3]])
```

### Replay one public session

Use `initial_message` and `customer_reply` from `evaluator/local_evaluator.py` with a row from `data/public_set.jsonl`.

### Inspect misses

After evaluation, open `results.json` → filter sessions where `"hit": false`.

---

## Team split (updated)

| Pillar | Primary owner | Modules |
|--------|---------------|---------|
| I | Joseph | `starter/retrieval/` |
| II | JY | `starter/agent.py` (dialog, routing, questions) |
| III | Both | `starter/personalization/` + glue in `agent.py` |
| IV | Both | Evaluator runs + metric tuning (no separate module) |
