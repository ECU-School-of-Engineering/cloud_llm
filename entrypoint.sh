#!/bin/bash
set -e

# Install private package at container startup using GITHUB_TOKEN from .env
if [ -n "${GITHUB_TOKEN}" ]; then
    echo "Installing escalation_scoring..."
    pip install --quiet git+https://x-access-token:${GITHUB_TOKEN}@github.com/ECU-School-of-Engineering/escalation_scoring.git
fi

exec "$@"
