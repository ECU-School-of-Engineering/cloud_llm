#!/bin/bash
# IVADE Setup Script for Linux (Barry / Ubuntu)
# Run from the project directory: ./install.sh
set -e

echo "=== IVADE Setup ==="

# Check Docker
if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker not found. Install with: curl -fsSL https://get.docker.com | sh"
    exit 1
fi
if ! docker info &>/dev/null; then
    echo "ERROR: Docker is not running or you need sudo. Try: sudo usermod -aG docker \$USER && newgrp docker"
    exit 1
fi
echo "OK  Docker is running"

# Check docker compose
if ! docker compose version &>/dev/null; then
    echo "ERROR: docker compose not found. Update Docker to a recent version."
    exit 1
fi
echo "OK  docker compose available"

# Check NVIDIA Container Toolkit
if ! docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
    echo "WARN: GPU not accessible to Docker. Install nvidia-container-toolkit:"
    echo "      https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
else
    echo "OK  NVIDIA GPU accessible"
fi

# Check .env
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found. Create it with HF_TOKEN, GITHUB_TOKEN, NGROK_AUTHTOKEN, HUME_API_KEY, OPENAI_API_KEY."
    exit 1
fi
echo "OK  .env found"

# Check required config files
for f in "grader_config.yaml" "config/fuzzy_system.yaml" "keys/ivade 1.pem"; do
    if [ ! -f "$f" ]; then
        echo "WARN: Missing $f — some services may fail to start."
    else
        echo "OK  $f found"
    fi
done

echo ""
echo "Starting containers..."
docker compose up -d --build

echo ""
echo "=== Done ==="
echo "Check status : docker compose ps"
echo "LLM logs     : docker compose logs -f llm"
echo "CLM logs     : docker compose logs -f clm"
