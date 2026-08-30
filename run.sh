#!/bin/bash
set -e

# The evaluation pipeline is fully deterministic and uses NO LLM on the hot
# path -- bucket lookup + verbatim-constraint scoring + popularity. The optional
# LLMMessageParser (chat.completions) is a bolt-on reachable only via `try`, and
# it needs a CHAT model, not an embedding one. The former run.sh pulled
# ai/embeddinggemma (an embedding model) and exported it as DOCKER_MODEL_NAME,
# a pairing that would never have worked against chat.completions.create. It is
# dropped here so nothing implies a dependency the evaluation does not have.

case "${1:-eval}" in
    try)
        # Optional: exercise the LLM parser. Requires a chat model to be served
        # locally and the three DOCKER_MODEL_* vars set (see llm_parser.py).
        python3 -m src.message_parser.try_it
        ;;
    eval)
        python3 -m evaluator.local_evaluator --output results_ours.json
        ;;
esac
