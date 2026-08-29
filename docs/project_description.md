# TechJam 2025 – Conversational E-Commerce Search Agent
## Written Project Description

---

## Problem Statement

Online shoppers often struggle to find the right product when they cannot articulate exactly what they want. Traditional keyword search requires the user to already know what to search for. This challenge asks us to build a multi-turn conversational shopping agent that narrows down a hidden target product from a 50,000-item clothing, shoes, and jewellery catalog — by asking smart clarifying questions and retrieving progressively better candidates across up to 10 conversation turns.

The agent is scored on three metrics:
- **Hit Rate@10** — did the target product appear in the top 10 recommendations?
- **MRR (Mean Reciprocal Rank)** — how highly ranked was the target?
- **MTTC (Mean Turns to Converge)** — how quickly was it found?

The combined **TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency**, where Efficiency rewards finding the target in fewer turns.

---

## Our Solution

We built a stateful, multi-component pipeline that handles four distinct shopping scenarios: **Buying** (hard constraint disclosed early), **Browsing** (vague intent), **Intent Override** (user changes their mind mid-session), and **Boundary** (user has no preference for a requested attribute).

### Pipeline Overview

```
User Message
     ↓
LLM Message Parser   ← Extract keywords, attributes, signals (Llama 3.1)
     ↓
Intent Router        ← Classify scenario type
     ↓
Field Checker        ← Track unfilled required fields, ask if missing
     ↓
Ledger Service       ← Store and manage session state
     ↓
BM25 Retrieval       ← Search 50,000 products
     ↓
Reranker             ← Score candidates by feature importance
     ↓
Decision Engine      ← Confidence gate: ask question or return top 10
     ↓
Response
```

### Component Descriptions

**LLM Message Parser**
Uses **Llama 3.1** to extract structured attributes from raw user messages — color, material, size, budget, style, use case, category, and brand. Detects three key signals: `is_override` (user changed intent), `is_no_preference` (boundary response), and `is_vague` (insufficient information). Uses clause-bounded negation detection to correctly handle messages like "I don't want polyester, I love cotton."

**Intent Router**
Classifies each turn into one of four scenario types using the parser's signals and turn history. Routes to the appropriate strategy — Buying uses efficient direct retrieval, Browsing uses exploratory clarification, Intent Override triggers a full constraint reset, and Boundary skips the attribute and moves to the next.

**Field Checker**
Tracks which required product attributes (category, size, color, budget, material, use case) have been filled in across turns. If the agent has already asked about a field but the user has not yet provided a value, it prioritises re-asking those fields before moving to new ones. Enforces the 10-turn limit — on turn 10, the agent forces a recommendation regardless of confidence.

**Ledger Service (`starter/ledger.py`)**
A thread-safe, in-memory session store keyed by `session_id`. Maintains all accumulated session state across turns: intent, hard constraints (e.g. `{"color": ["black", "red"]}`), soft product preferences (e.g. `["boots"]`), the assembled search key for retrieval, and the ordered list of attributes already asked (to avoid repetition). Supports CRUD operations, a context manager for atomic read-modify-write, and auto-cleanup of sessions via `managed_session()`.

**BM25 Retrieval (`starter/agent.py`)**
Indexes the full 50,000-product catalog using SQLite FTS5 at startup. Accepts a structured `search_key` dict from the ledger (e.g. `{"color": ["black"], "material": ["leather"]}`), builds an AND-of-OR FTS5 expression, and returns the top candidates ranked by BM25 score. Supports post-filtering on numeric fields (price, average rating, rating count).

**Reranker**
Takes the BM25 candidate pool and scores each candidate against the session's constraints and user profile. Extracts features including BM25 score, attribute match score, preference tag alignment, product rating, and price relevance. Returns a ranked top-10 list with a confidence score.

**Decision Engine / Confidence Gate**
If the reranker's confidence exceeds a tunable threshold, the agent returns the top 10 recommendations. If confidence is low (scores are close together), the agent asks a clarifying question for the next highest-priority uncovered attribute.

---

## Development Tools

- **Visual Studio Code** — primary IDE
- **Python 3.10+** — implementation language
- **Git / GitHub** — version control and collaboration

---

## APIs Used

- **Meta Llama 3.1** — used for structured attribute extraction from user messages, intent signal detection, and clarification question generation

---

## Libraries and Frameworks

| Library | Purpose |
|---|---|
| `sqlite3` (stdlib) | SQLite FTS5 for BM25 full-text search over the product catalog |
| `re` (stdlib) | Regex-based attribute extraction and signal detection |
| `threading` (stdlib) | Thread-safe locking in the ledger service |
| `contextlib` (stdlib) | Context manager implementation for session lifecycle |
| `dataclasses` (stdlib) | Structured data types for parsed messages |
| `unittest` (stdlib) | Test suite for evaluator, ledger, and message parser |
| Llama 3.1 | LLM-based message parsing, intent detection, and clarification generation |

---

## Datasets and Assets

- **Amazon Reviews 2023 — Clothing, Shoes & Jewelry** (McAuley Lab, UCSD)
  A frozen catalog of 50,000 products. Participant-visible fields: `parent_asin`, `title`, `features`, `description`, `price`, `categories`, `details`, `average_rating`, `rating_number`, `store`.

- **Public Evaluation Set** — 200 labeled shopping sessions provided by the organiser, covering all four scenario types (Buying 40%, Browsing 40%, Intent Override 15%, Boundary 5%).

- **Catalog vocabulary** — categories and brand names extracted from the full catalog at runtime, used to improve the message parser's category and brand matching.

No additional datasets were collected or manually labelled.
