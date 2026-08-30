# Lane C — demonstrated sessions

Two sessions from the public set, reproduced verbatim. Both are scenarios the
brief calls out as punishing a naive design: the first replaces a preference
mid-conversation, the second refuses to answer.

Reproduce either with:

```bash
python3 -m tools.build_index
python3 -m tools.bench --only public_0003 --output ""
```

---

## 1. Intent override — `public_0003`

Target `B09YMTWDXJ`, Casio Men's Wrist Watch AQ-800E-7A. **Hit on turn 3 at
rank 2.** The customer replaces a preference on turn 3.

> **[1] Customer:** I'm looking for Watches Wrist Watches. Stainless Steel Band
>
> **Agent:** Got it — watch and stainless steel. Here are ten to look at — my top
> pick because it matches your stainless steel preference. It would help to know
> anything else that matters to you, a brand you like, or what you will mostly
> use it for.
> `ask_attribute=other`

The opener names the durable category and a replaceable material preference.
Only `other` is scored; the two concrete topics ride along in the prose.

> **[2] Customer:** For that, what matters is: Water Resistant; 3 Year Battery.
>
> **Agent:** Got it — waterproof. Here are ten to look at — my top pick because
> it matches a specific requirement you mentioned. Could you tell me anything
> else that matters to you, a brand you like, or what you will mostly use it for?
> `ask_attribute=other`

Two constraints in one answer — the open-ended question earning its keep.

> **[3] Customer:** Actually, ignore my earlier preference. What I need is: Water
> Resistant.
>
> **Agent:** Understood — I have dropped stainless steel and I am going by
> waterproof now. Here are ten to look at — my top pick because it matches a
> specific requirement you mentioned. It would help to know anything else that
> matters to you, a brand you like, or what you will mostly use it for.
> `ask_attribute=other` — **target returned at rank 2**

The override is cross-slot: the opener's stainless-steel preference is retired
even though the new requirement is a feature. The category and waterproof
constraint learned on turn 2 remain live. The recommendation exclusion set is
cleared at the same transition, so products considered under the old intent may
be reconsidered immediately.

Final state: `category=[watch]`, `material=[]` with `stainless steel` retracted,
and `feature=[waterproof]`.

---

## 2. Boundary — `public_0041`

Target `B09MSY8926`, LETDIOSTO Women's Plus Size Tunic Tops. **Hit on turn 5 at
rank 1.** The customer declines the first question outright.

> **[1] Customer:** I'm looking for Tees & Blouses Tunics, but I'm still exploring.
>
> **Agent:** Got it — shirt and top. Here are ten to look at — my top pick because
> it matches your shirt preference. It would help to know anything else that
> matters to you, a brand you like, or any features that matter.
> `ask_attribute=other`

A browsing opener with no constraint at all, and recommendations go out anyway —
a question with no recommendations throws away a free chance at a hit.

> **[2] Customer:** I don't have a preference for other; please use your judgment.
>
> **Agent:** Understood — I will ask something more specific instead. Here are
> ten to look at — my top pick because it matches your shirt preference. Could
> you tell me a material you prefer, what you will mostly use it for, or a
> particular style or cut?
> `ask_attribute=material`

The open-ended question pauses immediately after the refusal and the policy
switches to a concrete material question.

> **[3] Customer:** For that, what matters is: polyester.
>
> **Agent:** Got it — polyester. Here are ten to look at — my top pick because it
> matches your shirt preference. It would help to know a colour you lean
> towards, the size or fit you need, or a budget you want to stay under.
> `ask_attribute=color`

> **[4] Customer:** I don't have an additional preference for color.
>
> **Agent:** That is fine, I will make the call on the colour. Here are ten to
> look at — my top pick because it matches your shirt preference. Could you tell
> me anything else that matters to you?
> `ask_attribute=other`

> **[5] Customer:** For that, what matters is: 3X women tops care: Hand wash or
> machine wash in gentle; Hang to dry; No ironing; No shrinkage; 进口.
>
> **Agent:** Got it — women and machine washable. Here are ten to look at — my
> top pick because it matches a specific requirement you mentioned. It would
> help to know anything else that matters to you.
> `ask_attribute=other` — **target returned at rank 1**

The policy respects the first refusal, gathers the next preference with a
specific question, and only returns to the open-ended action after intervening
concrete questions.
