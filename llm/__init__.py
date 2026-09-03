"""LLM reranking, isolated behind a hard fallback. This is the only package that
imports `anthropic` -- every call is wrapped so a missing key, network error, or
timeout falls back to the pre-rerank candidate order silently, never raising into the
Agent contract (CLAUDE.md: "the agent must fall back to pure hybrid retrieval if the
LLM errors, times out, or no API key is present")."""
