# Adversarial input robustness

363 test cases — 332 single messages across 30 characteristics, plus 31
multi-turn dialogues — asserting what the agent *should* do with language a
person would actually type.

```bash
python3 -m tools.robustness              # pass rate by characteristic
python3 -m tools.robustness --failures   # the bug list
python3 -m tools.robustness --tag negation --failures
python3 -m tools.robustness --dialogues  # multi-turn only
```

## Why this exists

The public evaluator never feeds the agent a human sentence. Its customer says
`"For that, what matters is: leather; 100% Leather."` — catalog text, quoted
verbatim. Nothing in the scored loop exercises negation, idiom, units,
politeness, or a number that isn't a price. **A failure here is invisible to
the TechnicalScore**, which is exactly what makes it dangerous: the private set
is where it shows up.

Cases assert intended behaviour, not current behaviour, so the corpus is a bug
list rather than a snapshot. Cases marked `soft` are genuinely debatable
(non-USD currency, spelled-out numbers, idioms) and never gate CI.

**Current: 272/283 hard (96.1%), 31/80 soft (38.8%).**

## Findings, worst first

### 1. Negation leaves the rejected value ACTIVE — 8 failures

`NEGATION_CUES` covers `not, no, without, avoid, anything but, except, don't
want`. It misses `nothing`, `hate`, `never`, `can't`:

| input | result |
|---|---|
| `nothing in pink` | `color=pink` **active** |
| `I hate polyester` | `material=polyester` **active** |
| `never black` | `color=black` **active** |
| `I can't wear nylon` | `material=nylon` **active** |
| `can't stand synthetic fabrics` | `material=synthetic` **active** |
| `nothing sleeveless` | `style=sleeveless` **active** |

This is worse than extracting nothing. The reranker adds a bonus per active
value, so the agent actively hunts for the thing the customer just rejected,
and the stronger Lane A's extraction gets the harder it chases. Severity is
high because rejection is common in real dialogue and absent from the simulator.

*Fix: extend `NEGATION_CUES` in `src/lexicons/__init__.py` (Lane A).* Note
`_NEG_WINDOW` is 24 characters, so `"nothing waterproof, I don't need it"` also
needs the cue to precede the value closely enough.

### 2. `no wait` is not an override cue — 1 failure

`show me sneakers → actually boots → no wait, sandals` ends holding **both**
`boots` and `sandals`. `OVERRIDE_CUES` lacks `no wait`, `hold on`, `sorry, I
meant`. Contradictory constraints ranked together is the exact failure the
intent-override scenario is designed to punish.

*Fix: extend `OVERRIDE_CUES` (Lane A).*

### 3. A bare number after a model name reads as a price — 1 failure

`air max 90` → budget `$90`. This is the class you reported as `10 to 20
litres`; Lane A's hardened `parse_budget` fixed the unit case, but a number
trailing a product/model name still slips through. 47 of 48 numeric-not-budget
cases now pass, so the guard is close.

*Fix: require currency or budget wording near a bare trailing number (Lane A).*

### 4. Colour idioms always false-positive — 15 soft failures, 0/15

`feeling blue`, `black friday`, `out of the blue`, `green with envy`, `silver
lining`, `caught red handed`, `a navy veteran` all extract a colour. Same for
`watch out` → `category=watch`, `top of the line` → `category=top`, `boot up
the computer` → `category=boots`.

Marked soft because the fix is genuinely hard — real disambiguation, not a word
list — and the cost is one spurious soft slot value rather than a wrong hard
constraint. Worth knowing before someone raises slot weight further: **every
point of extra slot weight amplifies these too.**

### 5. Smaller gaps

- `for my morning run` misses `use_case=running` — the lexicon has `running`
  and `jogging` but not the bare verb `run`.
- Spelled-out amounts (`under fifty dollars`) never parse: 0/5 soft.
- Non-USD (`under 50 euros`, `£60`, `50 quid`) is 4/7 soft. Defensible for a
  US catalog; listed so the decision is deliberate.
- Recipient inference (`a gift for my wife` → `women`) is 1/6 soft. Probably
  worth having: it is a free category narrowing on a very common opener.

## What already holds up

Worth stating, because it is most of the corpus: budget parsing 32/32 including
every range form; 47/48 numeric-not-budget; multi-slot sentences 10/10 (`black
cotton long sleeve shirt under $30 for the gym` extracts all five plus the
budget); rambling paragraphs 5/5; question forms 8/8; politeness-wrapped
requests 7/7; degenerate input 10/10 with **no input in the corpus raising an
exception**, which the evaluator would score as an outright miss.

The multi-turn dialogues are strong: override 5/6, override scope 2/2 (changing
colour does not drop material or use_case), restatement 2/2 (re-asserting a held
value is emphasis, not a change — the expensive mistake to get wrong),
boundary 4/4, accumulation 4/4, long 8-turn sessions 3/3.

## CI

`tests/test_robustness.py` gates two things without freezing the bugs:

1. **No input may ever raise** — absolute, all 363 cases, soft included.
2. **No new hard failure** — `KNOWN_FAILURES` is an allowlist of the 11 above.
   A failure outside it fails the build; a fixed one must be deleted from the
   list or a second test fails. The list may only get shorter.
