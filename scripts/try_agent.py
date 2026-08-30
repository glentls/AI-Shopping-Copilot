"""Interactive REPL to try the full Agent pipeline on your own input.

Unlike src/message_parser/try_it.py (parser only), this runs a real session
end to end: Intent Router -> Ledger -> Retrieval/Rerank -> Confidence ->
Output -- exactly the path evaluator/local_evaluator.py drives, just with you
typing the customer's side instead of the simulator.

Run from the repo root:
    python3 scripts/try_agent.py

Turn numbers auto-increment per session, starting at 1. Commands:
    reset   start a new session (fresh turn counter, fresh ledger state)
    quit    exit (Ctrl+D also works)
Anything else is sent as the customer's message for the current turn.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import Agent  # noqa: E402

# A representative user_profile shape (see docs/agent_api_contract.json) --
# doesn't need to be realistic, the pipeline only reads it for the
# rating_style tie-break in the reranker.
_DEFAULT_PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.2,
    "rating_style": "usually positive",
    "preference_tags": ["comfort", "fit"],
    "summary": "Prior purchases emphasize comfort and fit.",
}


def _new_session(agent: Agent) -> tuple[str, int]:
    session_id = f"try_{uuid.uuid4().hex[:8]}"
    agent.reset(session_id, dict(_DEFAULT_PROFILE))
    print(f"\n--- new session: {session_id} ---\n")
    return session_id, 1


def main() -> None:
    print("Building the agent (FTS5 index + catalog load)...")
    agent = Agent()
    print("Ready.\n")
    print("Type a customer message each turn (or 'reset' / 'quit'):\n")

    session_id, turn = _new_session(agent)

    while True:
        try:
            text = input(f"[turn {turn}] > ").strip()
        except EOFError:
            break

        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            break
        if text.lower() == "reset":
            session_id, turn = _new_session(agent)
            continue

        response = agent.respond(session_id, text, turn, top_k=10)
        print(json.dumps(response, indent=2))

        n_hits = len(response["recommendations"])
        print(f"  ({n_hits} recommendation(s) revealed, ask_attribute={response['ask_attribute']!r})\n")

        turn += 1
        if turn > 10:
            print("Session hit the 10-turn cap -- type 'reset' to start a new one.\n")


if __name__ == "__main__":
    sys.exit(main())
