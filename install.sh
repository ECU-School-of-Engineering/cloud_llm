#!/bin/bash
# One-time setup on the GPU machine (Linux/Barry)
# Run from the repo directory: ./install.sh
set -e

echo "=== IVADE Install ==="
echo "Building images and starting all containers..."

docker compose up -d --build

echo ""
echo "=== Done ==="
echo "Check status with: docker compose ps"
echo "Monitor logs with: docker compose logs -f clm"
