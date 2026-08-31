# Config J/K candidate-selection gate

## Status: frozen before reportable J or K evaluation

This protocol is committed before either candidate is run through the
reportable evaluator. At freeze time, `results.jsonl` contains no row for `J`
or `K`. The rules below may not be relaxed after a candidate result is seen.

The implementation base is `fedd07e83f6f12e7754a4da852dfe865642ec082`.
Reportable evaluation must use the later clean commit that contains this gate
and config `K`; that full SHA is called the **candidate-evaluation SHA** below.

## Narrow research questions

- `J` asks whether disclosure-derived evidence should choose Top-10 membership
  from a 50-product window while popularity and profile priors remain confined
  to reordering that frozen Top-10.
- `K` asks whether a product already shown in an unsuccessful turn should be
  withheld on later turns. It is exactly `T` plus `exclude_shown=True`.

The candidates are not combined. A combination would be a third, unregistered
hypothesis chosen after seeing component results, not evidence for either
question above. `Y` is not rerun: its commit records a non-canonical dev
diagnostic in which widening every reranker's window lost score versus `T`;
`J` is the scoped correction to that diagnosed failure mode. `W` and `X`
remain separate unmeasured hypotheses rather than consuming this pre-freeze
gate.

## Configuration invariants

Before evaluation, tests must prove that `K` differs from `T` only in `name`
and `exclude_shown`. The result row must additionally record all of these K
flags:

| Flag | Required value |
|---|---:|
| `exclude_shown` | `true` |
| `dense_text_recipe` | `full` |
| `negative_preference` | `false` |
| `rerank_window` | `0` |
| `rerank_window_scope` | `all` |

`SUBMISSION_CONFIG_NAME` remains `T`; registering a candidate must not change
what the unnamed competition entry point runs.

For `J`, the row must record `rerank_window=50` and
`rerank_window_scope="evidence"`. Replacing those two fields and its name with
T's values must recover `T` exactly.

## Environment and current-SHA T parity gate

Run from CPython 3.12 on Linux x86-64 with the locked dense requirements, the
official catalog (`da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`),
the official public set (`857259f7a438e6188ac63e18995b6ff4489bfcfc4a716a798b9a2aa0ee8f7579`),
the verified official model, and a clean tree. The effective retriever must be
`hybrid`; every reportability-reason list and all agent, evaluator, and
invalid-response exception counts must be empty or zero.

First run `T` on dev at the candidate-evaluation SHA:

```bash
python3 -m src.eval.runner --config T --split dev
```

The objective-bearing portion of `scores` must exactly reproduce the canonical
T dev row: sample count, every aggregate metric, and each scenario's sample
count, HitRate@10, MRR, and MTTC. Latency and resource measurements are not
part of score parity.

| Metric | Required T value |
|---|---:|
| sample count | `120` |
| TechnicalScore | `0.866774` |
| HitRate@10 | `0.941667` |
| MRR | `0.795913` |
| MTTC | `3.141667` |
| efficiency | `0.785833` |

| Scenario | n | HitRate@10 | MRR | MTTC | Derived TechnicalScore |
|---|---:|---:|---:|---:|---:|
| `boundary` | 6 | `1.000000` | `0.916667` | `4.166667` | `0.911667` |
| `browsing` | 48 | `0.979167` | `0.777282` | `2.645833` | `0.889851` |
| `buying` | 48 | `0.916667` | `0.795833` | `2.895833` | `0.859167` |
| `intent_override` | 18 | `0.888889` | `0.805556` | `4.777778` | `0.810556` |

Scenario TechnicalScore is derived with the official objective,
`0.50 * HR + 0.30 * MRR + 0.20 * (11 - MTTC) / 10`, and rounded to six
decimal places for the gate tables.

The T, J, and K dev rows must all carry the same full
candidate-evaluation SHA. The canonical results log is the evaluator's sole
permitted ignored path, so its appended rows do not justify an implementation
commit between runs.

If current-SHA T fails any scored parity check, stop. Do not compare J or K to
historical T and do not tune around the discrepancy. Shared code has changed
since T's canonical run, so a mismatch invalidates attribution until its cause
is understood.

## Dev runs and qualification gate

After T parity passes, run each candidate exactly once, without code or
parameter changes between runs:

```bash
python3 -m src.eval.runner --config J --split dev
python3 -m src.eval.runner --config K --split dev
```

A candidate qualifies only if every criterion passes:

| Criterion | Dev threshold |
|---|---:|
| reportable, effective hybrid, zero exceptions/invalid responses | required |
| TechnicalScore | **strictly greater than** `0.866774` |
| HitRate@10 | at least `0.941667` |
| scenario `boundary` TechnicalScore | at least `0.891667` |
| scenario `browsing` TechnicalScore | at least `0.869851` |
| scenario `buying` TechnicalScore | at least `0.839167` |
| scenario `intent_override` TechnicalScore | at least `0.790556` |

The scenario floors are T minus `0.02`. They prevent a small aggregate gain
from concealing a material regression in one shopper mode. A tie on overall
score fails: an experiment must add measured value to justify promotion.

Every completed invocation that reveals candidate scores counts as that
candidate's run, including a non-reportable result. A startup refusal before
evaluation produces no scores and does not count. This distinction permits an
environment typo to be corrected without permitting repeated scored trials.

## Winner selection

If neither candidate qualifies, retain `T` and do not open holdout. If one
qualifies, it is the provisional winner. If both qualify, choose once by this
pre-declared lexicographic order:

1. higher dev TechnicalScore;
2. higher dev HitRate@10;
3. higher dev MRR;
4. lower dev MTTC; then
5. `K`, because it is a one-field delta using an already-tested runtime seam,
   whereas J expands the number of candidates processed by evidence reranking.

Do not tune, combine candidates, or substitute a different candidate after
this selection.

## Single exploratory holdout opening

Only the provisional winner may be run on holdout, exactly once and without an
intervening implementation change:

```bash
python3 -m src.eval.runner --config <WINNER> --split holdout
```

It is promoted over T only if every criterion passes:

| Criterion | Holdout threshold |
|---|---:|
| reportable, effective hybrid, zero exceptions/invalid responses | required |
| TechnicalScore | **strictly greater than** `0.891630` |
| HitRate@10 | at least `0.975000` |
| scenario `boundary` TechnicalScore | at least `0.570714` |
| scenario `browsing` TechnicalScore | at least `0.917500` |
| scenario `buying` TechnicalScore | at least `0.878047` |
| scenario `intent_override` TechnicalScore | at least `0.832500` |

These scenario floors are again T minus `0.02`. The holdout is explicitly
**exploratory**, because the public set has already informed earlier research
and both candidates retain T's popularity prior. Every published mention of
the result must carry that label.

If the winner fails, retain `T`. The runner-up may not then be opened on
holdout: doing so would adapt candidate choice to holdout evidence. A
score-bearing non-reportable holdout invocation also fails the gate and may
not be repeated. A startup refusal that exposes no scores may be corrected.

## Outcome record

All reportable rows, including negative dev results, remain in
`results.jsonl`. Append an outcome section to this document only after the
protocol is complete, recording the candidate-evaluation SHA, exact config
flags, each gate decision, and matched-split aggregate deltas against the
current-SHA T row. Changing thresholds or omitting an unfavourable result is
not permitted.
