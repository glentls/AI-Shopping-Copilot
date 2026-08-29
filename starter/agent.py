from __future__ import annotations

from pathlib import Path

from starter.personalization import profile_adjusted_priority
from starter.retrieval.search import HybridSearcher

ATTRIBUTES = ("category", "material", "color", "size", "style", "brand", "budget", "feature", "use_case")
OVERRIDE_MARKERS = ("actually", "instead", "never mind", "nevermind", "forget that", "change of plans", "i meant", "rather")
EXHAUSTED_MARKERS = ("additional preference", "no other requirement")
# The final turn's question is never answered, so clarification stops one turn early.
LAST_ACTIONABLE_TURN = 10
ATTRIBUTE_TERMS = {
    "category": ("looking for", "need", "want", "shoes", "dress", "shirt", "bag", "jewelry", "boots"),
    "material": ("leather", "cotton", "wool", "linen", "suede", "silk", "denim", "material"),
    "color": ("black", "white", "blue", "red", "green", "brown", "pink", "grey", "gray", "color"),
    "size": ("size", "small", "medium", "large", " xs ", " s ", " m ", " l ", " xl "),
    "style": ("style", "casual", "formal", "vintage", "minimalist", "classic", "sporty"),
    "brand": ("brand",), "budget": ("$", "budget", "under", "less than", "cheap", "affordable", "price"),
    "feature": ("feature", "waterproof", "comfortable", "durable", "pockets", "slip resistant"),
    "use_case": ("for work", "for running", "for hiking", "for a wedding", "for travel", "gift", "occasion"),
}

class Agent:
    """Hybrid retrieval agent with multi-turn dialogue state."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.searcher = HybridSearcher(self.catalog_path)
        self._sessions: dict[str, dict] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized and may be used for personalization.
        self._sessions[session_id] = {
            "profile": user_profile or {},
            "slots": {attribute: None for attribute in ATTRIBUTES},
            "slot_status": {attribute: "unknown" for attribute in ATTRIBUTES},
            "unconstrained": set(), "asked": [], "history": [],
            "override_pending": False, "retrieval_feedback": {},
            "intent": "browsing", "intent_locked": False,
            "exploratory_session": False,
            "requirements_exhausted": False,
        }

    def set_intent(self, session_id: str, intent: str) -> None:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before setting intent")
        self._sessions[session_id]["intent"] = "buying" if intent == "buying" else "browsing"
        self._sessions[session_id]["intent_locked"] = True

    def get_dialog_state(self, session_id: str) -> dict:
        """Return a serializable handoff view for Pillar 1/3 orchestration."""
        if session_id not in self._sessions:
            raise RuntimeError("unknown session")
        state = self._sessions[session_id]
        return {
            "intent": state["intent"],
            "slots": dict(state["slots"]),
            "slot_status": dict(state["slot_status"]),
            "unconstrained": sorted(state["unconstrained"]),
            "asked": list(state["asked"]),
            "history": list(state["history"]),
            "retrieval_feedback": dict(state["retrieval_feedback"]),
        }

    def update_retrieval_feedback(self, session_id: str, feedback: dict) -> None:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before updating feedback")
        self._sessions[session_id]["retrieval_feedback"] = dict(feedback or {})

    def _extract_updates(self, message: str) -> dict[str, str]:
        text = " " + message.lower() + " "
        return {attribute: message.strip() for attribute, terms in ATTRIBUTE_TERMS.items() if any(term in text for term in terms)}

    def _route_intent(self, state: dict, message: str) -> None:
        """Pillar 1 routing on the official reset/respond execution path."""
        if state.get("intent_locked"):
            return
        lowered = message.lower()
        browsing_markers = (
            "still exploring", "just browsing", "show me options",
            "looking for ideas", "not sure what", "surprise me",
        )
        buying_markers = (
            "key requirement is", "what matters is", "what i need is",
            "must have", "under $", "budget around $", "size ",
        )
        if any(marker in lowered for marker in browsing_markers):
            state["intent"] = "browsing"
            state["exploratory_session"] = True
        elif any(marker in lowered for marker in buying_markers) and (
            not state.get("exploratory_session")
            or any(marker in lowered for marker in OVERRIDE_MARKERS)
        ):
            state["intent"] = "buying"

    def _apply_message(self, state: dict, message: str) -> None:
        lowered = message.lower()
        override = any(marker in lowered for marker in OVERRIDE_MARKERS)
        no_preference = any(phrase in lowered for phrase in (
            "no preference", "no additional preference", "anything is fine",
            "you decide", "use your judgment", "doesn't matter",
            "does not matter", "i'm flexible", "im flexible",
        ))
        if override:
            for attribute in state["slots"]:
                state["slots"][attribute] = None
                state["slot_status"][attribute] = "unknown"
            state["unconstrained"].clear(); state["asked"].clear(); state["override_pending"] = True
        for attribute, value in self._extract_updates(message).items():
            state["slots"][attribute] = value
            state["slot_status"][attribute] = "confirmed"
            state["unconstrained"].discard(attribute)
        if no_preference and state["asked"]:
            attribute = state["asked"][-1]
            if attribute in state["slots"]:
                state["slots"][attribute] = None
                state["slot_status"][attribute] = "unconstrained"
                state["unconstrained"].add(attribute)
        if any(marker in lowered for marker in EXHAUSTED_MARKERS):
            state["requirements_exhausted"] = True
        state["history"].append(message)

    def _choose_question(self, state: dict, turn: int) -> tuple[str, str] | None:
        """Pick (ask_attribute, message). A null attribute wastes the turn, so we always ask."""
        if turn >= LAST_ACTIONABLE_TURN:
            return None
        if state["override_pending"]:
            state["override_pending"] = False
            state["asked"].append("other")
            return "other", "What is the most important requirement for this new request?"

        # The open wildcard surfaces any still-undisclosed requirement, while a specific
        # attribute only pays off when the customer happens to hold that kind of constraint.
        if not state["requirements_exhausted"]:
            state["asked"].append("other")
            return "other", "What else matters most for this item?"

        missing = state["retrieval_feedback"].get("missing_attributes", [])
        priority = tuple(a for a in missing if a in ATTRIBUTES) if isinstance(missing, list) else ()
        priority = profile_adjusted_priority(priority, state["profile"])
        for attribute in priority:
            if state["slots"][attribute] is None and attribute not in state["unconstrained"] and attribute not in state["asked"]:
                state["asked"].append(attribute)
                if attribute == "category":
                    return attribute, "What type of item are you looking for?"
                if attribute == "use_case":
                    return attribute, "What will you mainly use it for?"
                return attribute, f"Do you have a preference for {attribute}?"
        state["asked"].append("other")
        return "other", "Anything else I should take into account?"

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        state = self._sessions[session_id]
        self._apply_message(state, user_message)
        self._route_intent(state, user_message)
        filters = {
            **state["slots"],
            "slot_status": state["slot_status"],
            "unconstrained": state["unconstrained"],
            "asked": state["asked"],
            "profile": state["profile"],
        }
        result = self.searcher.search(
            query_text="\n".join(state["history"]), mode=state["intent"],
            filters=filters, top_k=top_k,
        )
        state["retrieval_feedback"] = result.feedback
        recommendations = [{"parent_asin": asin} for asin in result.asins]
        question = self._choose_question(state, turn)
        return {
            "message": question[1] if question else "Here are the closest matches I found.",
            "ask_attribute": question[0] if question else None,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
