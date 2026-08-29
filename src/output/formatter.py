"""OutputFormatter: assemble the Agent API turn_response dict.

Reads a :class:`~src.confidence.payload.ConfidencePayload` (the decision) plus a
ranked list of ``parent_asin`` strings and emits the frozen contract dict. It
never decides *whether* to ask -- that is the confidence component's job; it
only shapes the response and phrases the clarifying question.
"""

from __future__ import annotations

from src.confidence.payload import ConfidencePayload

# Natural-language phrasing per allowed ask_attribute (contract enum).
_QUESTION_BY_ATTRIBUTE = {
    "category": "What type of item are you looking for?",
    "material": "Do you have a material preference?",
    "color": "Any color in mind?",
    "size": "What size do you need?",
    "style": "What style are you going for?",
    "brand": "Any brand you prefer?",
    "budget": "What's your budget?",
    "feature": "Are there any features that matter most to you?",
    "use_case": "What will you mainly use it for?",
    "other": "Anything else that would help me narrow this down?",
}

_RECOMMEND_MESSAGE = "Here are the closest matches I found."
_DEFAULT_ASK_MESSAGE = "Could you tell me a bit more about what you're after?"


class OutputFormatter:
    """Shape pipeline results into the Agent API ``turn_response`` contract."""

    def format(
        self,
        payload: ConfidencePayload,
        recommendations: list[str],
        usage: dict | None = None,
    ) -> dict:
        """Build the contract dict from a decision payload and recommendations.

        Recommendations are always attached (every turn returns a top-10). When
        ``payload.clarify`` is set, a clarifying ``message`` + ``ask_attribute``
        are included; otherwise ``ask_attribute`` is ``None``.
        """
        recs = [{"parent_asin": asin} for asin in recommendations[:10]]

        if payload.clarify and payload.ask_attribute:
            message = _QUESTION_BY_ATTRIBUTE.get(
                payload.ask_attribute, _DEFAULT_ASK_MESSAGE
            )
            ask_attribute = payload.ask_attribute
        else:
            message = _RECOMMEND_MESSAGE
            ask_attribute = None

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recs,
            "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0},
        }
