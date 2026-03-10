#!/bin/bash
set -e

# Install private packages at container startup using GITHUB_TOKEN from .env
# Always pulls latest HEAD from each repo
if [ -n "${GITHUB_TOKEN}" ]; then
    echo "Installing private packages..."
    pip install --quiet git+https://x-access-token:${GITHUB_TOKEN}@github.com/ECU-School-of-Engineering/escalation_scoring.git
    pip install --quiet git+https://x-access-token:${GITHUB_TOKEN}@github.com/ECU-School-of-Engineering/llm_escalation_evaluator.git
fi

exec "$@"
