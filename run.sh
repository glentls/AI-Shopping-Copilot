#!/bin/bash
set -e

export DOCKER_MODEL_BASE_URL="http://localhost:12434/engines/llama.cpp/v1"
export DOCKER_MODEL_API_KEY="none"
export DOCKER_MODEL_NAME="ai/embeddinggemma"

# Check Docker is running
if ! docker info &>/dev/null; then
    echo "Docker is not running. Please open Docker Desktop and try again."
    open -a Docker 2>/dev/null || true
    exit 1
fi

# Pull model if not present
if ! docker model list | grep -q "gemma"; then
    echo "Pulling ai/embeddinggemma..."
    docker model pull ai/embeddinggemma
fi

case "${1:-try}" in
    try)
        python3 -m src.message_parser.try_it
        ;;
    eval)
        python3 -m evaluator.local_evaluator --output results_ours.json
        ;;
esac
