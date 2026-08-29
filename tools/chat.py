"""Talk to the agent yourself.

    python3 -m tools.chat                          # interactive
    python3 -m tools.chat --target B071X54486      # say where the target ranks
    python3 -m tools.chat --script "I need hiking boots" "waterproof, under $120"

The evaluator only ever feeds the agent the simulator's stilted phrasing. This
is the way to see what it does with a sentence a person would actually type --
which is the case the public score cannot measure.

In-session commands: /state /why /reset /profile a,b /target ASIN /help /quit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.policy.question import other_value, recommendation_window
from starter.agent import Agent

BANNER = """
  TechJam shopping agent -- interactive session
  type a message and press enter.  /help for commands, /quit to leave
"""

HELP = """
  /state          what the agent believes right now
  /why            score components behind the current top pick
  /reset          start a fresh session (turn counter back to 1)
  /profile a,b    set preference tags, e.g. /profile fit,comfort
  /target ASIN    track a product and report its rank each turn
  /help  /quit
"""


def load_titles(catalog_path: Path) -> dict[str, str]:
    titles: dict[str, str] = {}
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            titles[str(product["parent_asin"])] = str(product.get("title") or "")
    return titles


class Session:
    def __init__(self, agent: Agent, titles: dict[str, str], top_k: int,
                 profile: dict, target: str | None) -> None:
        self.agent = agent
        self.titles = titles
        self.top_k = top_k
        self.profile = profile
        self.target = target
        self.turn = 0
        self.session_id = "chat-1"
        self.last = None
        self.reset()

    def reset(self) -> None:
        self.turn = 0
        self.session_id = f"chat-{int(time.time() * 1000)}"
        self.agent.reset(self.session_id, self.profile)
        self.last = None

    @property
    def state(self):
        return self.agent._states.get(self.session_id)

    def say(self, message: str) -> None:
        self.turn += 1
        started = time.perf_counter()
        response = self.agent.respond(self.session_id, message, self.turn, self.top_k)
        elapsed = (time.perf_counter() - started) * 1000
        self.last = response

        print(f"\n  agent  {response['message']}")
        print(f"         [ask_attribute={response['ask_attribute']}  "
              f"turn={self.turn}  {elapsed:.0f}ms]")

        asins = [item["parent_asin"] for item in response["recommendations"]]
        offset = recommendation_window(self.state, self.top_k) if self.state else 0
        window = f"  (showing ranks {offset + 1}-{offset + len(asins)})" if offset else ""
        print(f"\n  recommendations{window}")
        for position, asin in enumerate(asins, start=1):
            mark = "  <-- TARGET" if asin == self.target else ""
            print(f"    {position:2}. {asin}  {self.titles.get(asin, '?')[:74]}{mark}")

        if self.target:
            self._report_target(asins)

    def _report_target(self, shown: list[str]) -> None:
        if self.target in shown:
            print(f"\n  HIT -- target at rank {shown.index(self.target) + 1}")
            return
        depth = self._target_depth()
        where = f"rank {depth} of the full ranking" if depth else "not retrieved at all"
        print(f"\n  miss -- target is {where}")

    def _target_depth(self) -> int | None:
        if self.state is None:
            return None
        try:
            cands = self.agent.retriever.search(self.state, top_n=300)
            cands = self.agent.retriever.rerank(cands, self.state)
        except Exception:
            return None
        for position, candidate in enumerate(cands, start=1):
            if candidate.parent_asin == self.target:
                return position
        return None

    def show_state(self) -> None:
        state = self.state
        if state is None:
            print("  no state yet")
            return
        print("\n  believed:")
        for slot, values in sorted(state.slots.items()):
            live = [v.value for v in values if v.polarity]
            dead = [v.value for v in values if not v.polarity]
            confidence = {v.value: round(v.confidence, 2) for v in values if v.polarity}
            if live or dead:
                line = f"    {slot:10} {live}"
                if dead:
                    line += f"   retracted={dead}"
                print(f"{line}   conf={confidence}")
        if not state.slots:
            print("    (nothing extracted yet)")
        print(f"\n    budget_max   {state.budget_max}")
        print(f"    asked        {state.asked}")
        print(f"    unanswerable {sorted(state.unanswerable)}")
        print(f"    wildcard value {other_value(state):.2f}"
              f"   page offset {recommendation_window(state, self.top_k)}")

    def show_why(self) -> None:
        state = self.state
        if state is None:
            print("  nothing yet")
            return
        try:
            cands = self.agent.retriever.rerank(self.agent.retriever.search(state), state)
        except Exception as error:
            print(f"  could not score: {error}")
            return
        print("\n  top 5 with score components:")
        for position, candidate in enumerate(cands[:5], start=1):
            parts = "  ".join(f"{k}={v:.3f}" for k, v in sorted(candidate.components.items()))
            print(f"    {position}. {candidate.parent_asin}  score={candidate.score:.3f}  {parts}")
            print(f"       {self.titles.get(candidate.parent_asin, '?')[:76]}")
            if candidate.why:
                print(f"       why: {candidate.why}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive session with the agent")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--artifacts", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--profile-tags", default="",
                        help="comma-separated preference tags for the user profile")
    parser.add_argument("--target", default=None,
                        help="parent_asin to track; reports its rank every turn")
    parser.add_argument("--script", nargs="+", metavar="MSG",
                        help="run these messages non-interactively and exit")
    args = parser.parse_args()

    catalog = Path(args.catalog)
    if not catalog.exists():
        sys.exit(f"catalog not found: {catalog}  (see README for the download step)")

    print("  loading catalog and indexes ...", file=sys.stderr)
    started = time.perf_counter()
    titles = load_titles(catalog)
    agent = Agent(catalog, args.artifacts) if args.artifacts else Agent(catalog)
    print(f"  ready in {time.perf_counter() - started:.1f}s "
          f"({len(titles):,} products, retrieval mode "
          f"{getattr(agent.retriever, 'mode', 'bm25')})", file=sys.stderr)

    tags = [tag.strip() for tag in args.profile_tags.split(",") if tag.strip()]
    session = Session(agent, titles, args.top_k,
                      {"preference_tags": tags} if tags else {}, args.target)

    if args.script:
        for message in args.script:
            print(f"\n  you    {message}")
            session.say(message)
        return

    print(BANNER)
    while True:
        try:
            message = input("\n  you    ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            continue
        command, _, argument = message.partition(" ")
        if command in ("/quit", "/exit", "/q"):
            return
        if command == "/help":
            print(HELP)
        elif command == "/state":
            session.show_state()
        elif command == "/why":
            session.show_why()
        elif command == "/reset":
            session.reset()
            print("  new session")
        elif command == "/profile":
            tags = [t.strip() for t in argument.split(",") if t.strip()]
            session.profile = {"preference_tags": tags} if tags else {}
            session.reset()
            print(f"  profile set to {tags}; session reset")
        elif command == "/target":
            session.target = argument.strip() or None
            print(f"  tracking {session.target}")
        elif command.startswith("/"):
            print(f"  unknown command {command}; /help for the list")
        else:
            session.say(message)


if __name__ == "__main__":
    main()
