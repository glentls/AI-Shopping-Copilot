"""Required submission entry point (docs/submission_rules.md, docs/agent_api_contract.json).

`evaluator/local_evaluator.py` imports `Agent` from this exact module and constructs it as
`Agent(catalog_path)`. The real implementation lives in `src/agent.py` per
docs/plan/architecture.md's directory-ownership rules (retrieval/, ranking/, dialog/, memory/,
agent.py under src/, owned one-per-team-member); this file is kept only as the thin, contractually
required shim so the evaluator's import path never has to change. Do not put logic here.
"""

from __future__ import annotations

from src.agent import Agent

__all__ = ["Agent"]
