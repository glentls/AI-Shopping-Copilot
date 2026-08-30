"""LLM-based follow-up message generation (experimental, opt-in).

Reads the *actual* recommended products (title text, not just the
``parent_asin`` IDs) alongside the same situational signals the hardcoded
system uses (:class:`~src.output.followup.FollowUpContext`), and asks a
local LLM -- the same Docker Model Runner infrastructure already used by
``src.message_parser.llm_parser.LLMMessageParser`` -- to write a clarifying
question or interaction message grounded in what the search algorithm
actually found for this turn.

Opt-in via ``FOLLOWUP_MODE=llm`` (see ``src/agent.py``); the hardcoded
bundled-missing-attributes system (``src.output.followup``) remains the
default and is completely unaffected when this isn't enabled. Never raises:
any failure (package missing, env vars unset, connection/timeout error,
empty or malformed response) falls back to the hardcoded
``build_all_missing_ask_message``, so a turn can never be lost to this
being experimental.

Latency is explicitly out of scope for this experiment -- a live call every
clarify turn, no caching or precompute optimisations.
"""

from __future__ import annotations

import logging
import os

from src.output.followup import FollowUpContext, build_all_missing_ask_message

logger = logging.getLogger(__name__)

# Caps how many candidate products are described in the prompt -- keeps the
# request bounded regardless of top_k. Not a latency optimisation (out of
# scope here), just keeps the prompt from growing unboundedly.
_MAX_PRODUCTS_IN_PROMPT = 10

_SYSTEM_PROMPT = """\
You are a friendly shopping assistant helping a customer narrow down a \
product search. You will be given: the conversation scenario, which \
product attributes are still missing (or none, if everything is covered), \
and the products the search has found so far.

Write ONE short, natural, conversational follow-up message (1-2 sentences):
- If attributes are missing, ask about them naturally. You may reference a \
specific product from the list if it helps (e.g. "I see some running shoes \
and some hiking boots in the results -- which are you after?").
- If nothing is missing, ask a light closing question, e.g. whether the \
customer prefers higher-rated options or more popular picks.
- Do not use bullet points, headers, or a robotic attribute-list format.
- Do not invent product details that aren't in the list you were given.

Respond with the message text only -- no preamble, no quotes, no markdown.
"""


def _get_client():
    """Build the Docker Model Runner OpenAI-compatible client. Raises on any
    setup problem -- callers must catch, per this module's never-raise
    contract for ``build_llm_ask_message``."""
    base_url = os.environ.get("DOCKER_MODEL_BASE_URL")
    api_key = os.environ.get("DOCKER_MODEL_API_KEY")
    model = os.environ.get("DOCKER_MODEL_NAME")
    missing = [
        name for name, val in [
            ("DOCKER_MODEL_BASE_URL", base_url),
            ("DOCKER_MODEL_API_KEY", api_key),
            ("DOCKER_MODEL_NAME", model),
        ] if not val
    ]
    if missing:
        raise RuntimeError(
            "build_llm_ask_message requires the following environment variables: "
            + ", ".join(missing)
            + "\nSee src/message_parser/README.md for Docker Model Runner setup "
            "(the same infra this reuses)."
        )
    from openai import OpenAI  # noqa: PLC0415 -- optional dependency, imported lazily

    return OpenAI(base_url=base_url, api_key=api_key), model


def _describe_products(products: list[dict]) -> str:
    if not products:
        return "(no products found yet)"
    lines = [f"- {p.get('title', 'unknown product')}" for p in products[:_MAX_PRODUCTS_IN_PROMPT]]
    return "\n".join(lines)


def build_llm_ask_message(
    context: FollowUpContext,
    recommended_products: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 120,
) -> str:
    """LLM-generated follow-up message, grounded in the actual recommended
    products for this turn. Falls back to the hardcoded
    ``build_all_missing_ask_message`` on any failure.

    ``recommended_products`` is a list of ``{"parent_asin": ..., "title": ...}``
    dicts for the products already chosen by search/rerank this turn --
    read-only display data, never influences retrieval or ranking.
    """
    try:
        client, model = _get_client()
        missing_desc = (
            ", ".join(context.missing_attrs)
            if context.missing_attrs
            else "nothing -- every attribute is already covered"
        )
        user_prompt = (
            f"Scenario: {context.scenario}\n"
            f"Turn: {context.turn}\n"
            f"Attributes still missing: {missing_desc}\n"
            f"Products found so far:\n{_describe_products(recommended_products)}\n\n"
            "Write the follow-up message now."
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("empty LLM response")
        return text
    except Exception as exc:  # noqa: BLE001 -- must never raise, always fall back
        logger.warning("LLM follow-up generation failed, falling back: %s", exc)
        return build_all_missing_ask_message(context)
