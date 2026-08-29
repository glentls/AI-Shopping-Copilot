"""What the customer actually reads.

LANE C OWNS THIS FILE. Working placeholder.

This text does not affect the score at all -- only ask_attribute and
recommendations do. Make it good anyway: it is what human judges see in the
demo, and one of the final deliverables is a demonstrated multi-turn session.

If you want a small local LLM here, put it behind an environment variable with
these templates as the fallback, wrap it in try/except, and never call it
anywhere else in the pipeline. The evaluator counts an exception as a MISS, and
800 sessions x 10 turns of model calls will blow any latency budget.
"""

from __future__ import annotations

from typing import Optional

from src.contracts import Candidate, ConversationState

PHRASING = {
    "category": "what kind of item you have in mind",
    "material": "a material you prefer",
    "color": "a colour you prefer",
    "size": "the size or fit you need",
    "style": "a particular style or cut",
    "brand": "a brand you like",
    "budget": "a budget you want to stay under",
    "feature": "any features that matter",
    "use_case": "what you will mostly use it for",
    "other": "anything else that matters to you",
}


def _acknowledge(state: ConversationState) -> str:
    filled = [v for slot in ("use_case", "feature", "material", "color", "style") for v in state.active(slot)]
    if not filled:
        return "Let me help you narrow this down."
    return "Got it — looking for something " + ", ".join(filled[:3]) + "."


def compose_message(
    state: ConversationState,
    cands: list[Candidate],
    ask_attribute: Optional[str],
    extra_topics: list[str],
) -> str:
    parts = [_acknowledge(state)]

    if cands:
        why = cands[0].why
        parts.append(f"Here are ten that fit best{' — ' + why if why else ''}.")

    topics = [t for t in ([ask_attribute] if ask_attribute else []) + list(extra_topics) if t in PHRASING]
    if topics:
        phrased = [PHRASING[t] for t in topics[:3]]
        if len(phrased) == 1:
            question = f"Could you tell me {phrased[0]}?"
        else:
            question = f"Could you tell me {', '.join(phrased[:-1])}, or {phrased[-1]}?"
        parts.append(question)

    return " ".join(parts)
