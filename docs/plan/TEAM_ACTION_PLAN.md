# TechJam Build Plan

A working, scoring, integrated skeleton already exists on `main`. This document is for anyone
picking up a piece of it: what's already true, how to add to it without breaking it, and who
builds what next. If you only read one section, read "The loop" — it's the same four steps for
every person on every task.

## Where things stand

- The full pipeline (dialog → memory → retrieval → ranking → agent) runs end to end and scores
  **exactly the same** as the original weak BM25 baseline: `hit_rate_at_10=0.133333,
  mrr=0.073378, technical_score=0.114147` on the 150-session dev split. Every "smart" component
  right now is a stub that does the dumbest safe thing — the score is 100% BM25, 0% intelligence.
  That's expected. It proves the wiring works before anyone builds anything clever on top of it.
- Four facts about how this gets scored change what's worth building. Full detail in
  `CLAUDE.md` and `RECON.md`; the short version:
  1. The simulated customer only ever reacts to the structured `ask_attribute` field you return —
     never to your `message` text. Writing a beautifully-worded question buys nothing; picking the
     *right attribute* buys real information.
  2. Recommendations are scored on every turn, not just the last one. Always send your current
     best 10 guesses, even on a turn where you're also asking a question.
  3. The catalog has no `brand` field and `price` is missing on 79% of products. `store` is the
     closest thing to a brand signal. Don't hard-filter on either — soft-score them instead.
  4. There's no user ID anywhere in the contract, so there's no way to remember a shopper between
     sessions. Memory work is scoped to *within* one conversation only.

## The loop

Every task below — no matter who's doing it — is the same four steps:

1. **Write the logic as a plain function.** Same inputs, same output shape as the stub (the
   `null_*.py` file in your folder) it's replacing. Don't touch `agent.py`. Don't touch anyone
   else's folder.
2. **Unit test it alone.** Feed it fake data, check the output looks right. No catalog, no
   evaluator — just your function. This should take seconds to run.
3. **Plug it in.** Point the `primary` import in your folder's `__init__.py` at your new function
   instead of the null one.
4. **Run the two checks, always in this order:**
   - `make test` — must stay 17/17 green. A failure here means you broke the *failure contract*
     (something you wrote can crash or hang the agent), which is a different and more urgent
     problem than a bad score.
   - `make eval-fast` — compare the score to what it was before your change.

## Testing, cheapest to most expensive

| Layer | Command | What it proves | Cost |
|---|---|---|---|
| Unit test | (your own test file) | Your function does the right thing on data you made up | milliseconds |
| Failure contract | `make test` | The whole agent survives your component crashing or hanging | ~3 seconds |
| Fast eval | `make eval-fast` | Your change actually helps (or hurts) real sessions | ~15 seconds |
| Full eval | `make eval` | Same, on all 150 dev sessions — run before merging | ~30 seconds |
| Holdout | `make eval-holdout` | Are we overfitting to the public set? | run rarely, by agreement |

**If a score looks suspiciously unchanged or suspiciously bad**, don't trust it and move on —
check first that your component isn't silently failing and falling back to its Null path. That's
a real bug that happened during the skeleton build: a broken component still produced a plausible,
non-zero, wrong number instead of an obvious crash.

**Never run `make eval-holdout` while developing.** It's the only honest signal the team has for
whether the private 800-session set will disagree with the public 200 — spend it carefully, at
agreed checkpoints, not as a personal sanity check.

## Five-person plan

Ownership matches the folders the skeleton already set up. Each list is roughly sequential — do
item 1 before item 2.

### R1 — Retrieval · `src/retrieval/`

- [ ] Embed all 50k products (`title + categories + store + top features` — no `brand` field
      exists, use `store`). Cache the vectors to disk.
- [ ] Add a dense-search function returning the same `list[Candidate]` shape BM25 already returns.
      Plain numpy cosine similarity is enough at 50k rows — skip FAISS, it's not needed at this scale.
- [ ] Fuse BM25 + dense with Reciprocal Rank Fusion (one formula; the `k` constant goes in
      `config.yaml`, not the code).
- [ ] Add category filtering as a **hard** filter (the one field that's always populated); treat
      price and store as **soft** boosts only.
- [ ] Add relaxation: if a hard filter empties the pool, drop it and retry.
- [ ] Report Recall@100 (is the gold item even in your top 100?) and re-check it after every
      retrieval change — it's the ceiling everything downstream is capped by.

### R2 — Ranking · `src/ranking/`

- [ ] Install a cross-encoder (e.g. `ms-marco-MiniLM-L-6-v2`) and run one real query through it —
      only its *availability* has been confirmed so far, not a working install.
- [ ] Wire it in as the real reranker behind `NullReranker`'s exact signature. Leave
      `NullReranker` itself alone — it's the permanent fallback, not scaffolding.
- [ ] Test the standard way: unit test on fake candidates → `make test` → `make eval-fast`.
- [ ] Only once that's solid: an LLM listwise reranker, behind a `config.yaml` flag and a
      key-presence check, **off by default** — there are zero API keys anywhere on the dev machine.

### R3 — Dialog · `src/dialog/`

- [ ] Don't read `data/public_set.jsonl` looking for example conversations — it has none. Run
      15–20 sessions through the evaluator with a logging agent instead, to see what the simulated
      customer actually says.
- [ ] Build a simple buy-vs-browse classifier from the opening message.
- [ ] Build slot extraction from words that actually appear in the catalog (regex/keyword list),
      not a general-purpose parser.
- [ ] Build the `ask_attribute` picker. **This is the single highest-leverage thing anyone on the
      team builds** — the simulator only ever reacts to this field. Pick whichever attribute would
      most narrow the candidate pool.
- [ ] Build override detection with a marker-word list (`actually`, `instead`, `never mind`,
      `change of plans`) — the override message follows a fixed template, so this is easy to catch.
- [ ] Always attach the current best 10 recommendations alongside any question — never send a
      question with an empty list.

### R4 — Agent & Eval · `src/agent.py`, `eval/`

- [ ] As each teammate hands off a working component, swap their `primary` import in — this should
      be a one-line change, since every interface is already frozen.
- [ ] After every swap: `make test`, then `make eval-fast`, log the result.
- [ ] Build a simple ablation table (BM25 / +dense / +cross-encoder / +dialog) so the team can see
      which piece is actually earning its score, not just guess.
- [ ] Own `config.yaml` — if someone wants to hard-code a threshold "just for now," the answer is
      it goes in the config file instead.
- [ ] Gatekeep the holdout split. Run `make eval-holdout` at agreed checkpoints only, and announce
      the result to the team rather than letting people check it ad hoc.

### R5 — Memory & Docs · `src/memory/`, `docs/`, demo

- [x] Build intra-session distillation only — compress this session's rejected items and confirmed
      constraints. There's no user ID anywhere in the contract, so cross-session memory has
      nothing to attach to; don't build it.
- [x] Feed the distilled state into retrieval's `soft_prefs` as a boost.
- [x] Write the architecture diagram, README reproduction steps, and limitations section — none of
      this blocks anyone else, so it can happen in parallel from day one.
- [x] Restore the catalog and run the full checks; current full score is `0.113952` versus the recorded `0.114147` baseline.
- [x] Improve the integrated system enough to beat the recorded baseline (`0.540232` full-dev score).
- [ ] Record the demo once the team beats baseline end to end; capture the moment where the agent
      asks a question, narrows down, and hits.

## Ground rules for everyone

- A component never imports another component. Only `src/agent.py` is allowed to know about all
  of them.
- Every tunable number lives in `config.yaml`. If it affects behavior, it doesn't belong in code.
- Don't delete a `null_*.py` file after replacing it — it's the permanent fallback, not a draft.
- Don't edit `evaluator/local_evaluator.py` or `tests/test_evaluator.py`.
