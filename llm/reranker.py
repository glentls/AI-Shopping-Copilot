"""LLM reranking of a retrieval candidate pool down to top_k, with a hard fallback to
the pre-rerank order. Forces a tool call (`tool_choice`) so the response is guaranteed
structured JSON rather than free text that needs fragile parsing.

Model choice: claude-opus-5, per this project's default policy (not downgraded for
cost). Configurable via configs/dialog.json -> llm_rerank.model so cost/latency can be
traded off and disclosed explicitly (CLAUDE.md requires disclosing model choice,
estimated cost, token usage, and latency).
"""

from __future__ import annotations

import time

from llm.cache import LLMCache, cache_key

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_TIMEOUT_SECONDS = 20.0

RERANK_TOOL = {
    "name": "rank_candidates",
    "description": (
        "Return the given candidate parent_asin values ordered best-to-worst match for "
        "what the shopper has said they want. Include every candidate exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ranked_asins": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Every candidate parent_asin, most relevant to the stated need first.",
            }
        },
        "required": ["ranked_asins"],
        "additionalProperties": False,
    },
}


def _build_prompt(query: str, candidates: list[dict]) -> str:
    lines = [f"- {c['parent_asin']}: {c.get('title', '')}" for c in candidates]
    return (
        f"A shopper is looking for: {query}\n\n"
        f"Candidate products:\n" + "\n".join(lines) + "\n\n"
        "Call rank_candidates with every parent_asin above, ordered best match first."
    )


class LLMReranker:
    """Stateless wrapper: construct once (loads the client lazily on first real call),
    call rerank() every turn. Never raises -- returns None on any failure so the caller
    falls back to the order it already had."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        cache_dir: str = "data/llm_cache",
    ):
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.cache = LLMCache(cache_dir)
        self._client = None  # constructed lazily; None means "not yet attempted"
        self._unavailable = False  # set once we know we can't call the API at all

    def _get_client(self):
        if self._unavailable:
            return None
        if self._client is not None:
            return self._client
        import os

        if not os.environ.get("ANTHROPIC_API_KEY"):
            self._unavailable = True
            return None
        try:
            import anthropic

            self._client = anthropic.Anthropic().with_options(timeout=self.timeout_seconds, max_retries=1)
        except Exception:
            self._unavailable = True
            return None
        return self._client

    def rerank(self, query: str, candidates: list[dict]) -> tuple[list[str] | None, dict]:
        """Returns (ranked_asins_or_None, usage). usage is {"prompt_tokens": int,
        "completion_tokens": int} -- zero on any fallback path, real on a live call, and
        the original call's usage on a cache hit (see llm/cache.py: this represents the
        equivalent LLM cost of the decision, not marginal spend on this particular run)."""
        zero_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        if not candidates:
            return None, zero_usage

        candidate_ids = [c["parent_asin"] for c in candidates]
        key = cache_key(self.model, query, candidate_ids)
        cached = self.cache.get(key)
        if cached is not None:
            return cached["ranked_asins"], cached["usage"]

        client = self._get_client()
        if client is None:
            return None, zero_usage

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=2048,
                tools=[RERANK_TOOL],
                tool_choice={"type": "tool", "name": "rank_candidates"},
                messages=[{"role": "user", "content": _build_prompt(query, candidates)}],
            )
        except Exception:
            return None, zero_usage

        ranked_asins = None
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "rank_candidates":
                raw = block.input.get("ranked_asins")
                if isinstance(raw, list) and all(isinstance(a, str) for a in raw):
                    valid = set(candidate_ids)
                    ranked_asins = [a for a in raw if a in valid]
                    ranked_asins += [a for a in candidate_ids if a not in ranked_asins]
                break

        if ranked_asins is None:
            return None, zero_usage

        usage = {
            "prompt_tokens": int(getattr(response.usage, "input_tokens", 0) or 0),
            "completion_tokens": int(getattr(response.usage, "output_tokens", 0) or 0),
        }
        self.cache.put(key, {"ranked_asins": ranked_asins, "usage": usage})
        return ranked_asins, usage
