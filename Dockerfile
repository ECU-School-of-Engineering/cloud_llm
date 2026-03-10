FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python alias
RUN ln -s /usr/bin/python3 /usr/bin/python

# Python deps (public packages only)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

WORKDIR /app

# Entrypoint installs private package at container start using GITHUB_TOKEN from .env
# Written inline to guarantee LF line endings on both Windows and Linux hosts
RUN printf '#!/bin/bash\nset -e\nif [ -n "${GITHUB_TOKEN}" ]; then\n    echo "Installing private packages..."\n    pip install --quiet git+https://x-access-token:${GITHUB_TOKEN}@github.com/ECU-School-of-Engineering/escalation_scoring.git\n    pip install --quiet git+https://x-access-token:${GITHUB_TOKEN}@github.com/ECU-School-of-Engineering/llm_escalation_evaluator.git\nfi\nexec "$@"\n' > /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
