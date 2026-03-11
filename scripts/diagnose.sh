#!/bin/bash
# IVADE Diagnostics Script for Linux (Barry)
# Run from the project directory: ./scripts/diagnose.sh

echo "=== IVADE Diagnostics ==="
echo ""

# Docker
echo "--- Docker ---"
if docker info &>/dev/null; then
    echo "OK  Docker is running"
else
    echo "FAIL Docker is not running"
fi

# Container status
echo ""
echo "--- Containers ---"
docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

# Health endpoints
echo ""
echo "--- Health Checks ---"
for check in "LLM|http://localhost:8001/health" "CLM|http://localhost:8080/health"; do
    name=$(echo $check | cut -d'|' -f1)
    url=$(echo $check | cut -d'|' -f2)
    resp=$(curl -sf --max-time 5 "$url" 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "OK  $name: $resp"
    else
        echo "FAIL $name: not responding at $url"
    fi
done

# ngrok
echo ""
echo "--- ngrok Tunnel ---"
ngrok_resp=$(curl -sf --max-time 5 http://localhost:4040/api/tunnels 2>/dev/null)
if [ $? -eq 0 ]; then
    url=$(echo "$ngrok_resp" | python3 -c "import sys,json; t=json.load(sys.stdin)['tunnels']; print(t[0]['public_url'] if t else 'no tunnels')")
    echo "OK  Public URL: $url"
else
    echo "FAIL ngrok not responding on :4040"
fi

# GPU
echo ""
echo "--- GPU ---"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
else
    echo "WARN nvidia-smi not found"
fi

# GPU inside Docker
echo ""
echo "--- GPU in Docker ---"
if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L &>/dev/null; then
    docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L
    echo "OK  GPU accessible to containers"
else
    echo "FAIL GPU not accessible to containers - check nvidia-container-toolkit"
fi

# Disk
echo ""
echo "--- Disk ---"
if [ -d "models" ]; then
    size=$(du -sh models/ 2>/dev/null | cut -f1)
    echo "OK  models/ folder: $size"
else
    echo "WARN models/ folder not found"
fi
df -h . | tail -1 | awk '{print "OK  Disk: " $4 " free of " $2}'

# Recent errors
echo ""
echo "--- Recent Errors (last 20 lines each) ---"
for svc in llm clm; do
    errors=$(docker compose logs $svc --tail=20 2>&1 | grep -E "ERROR|Traceback|Exception" | tail -3)
    if [ -n "$errors" ]; then
        echo "WARN $svc has recent errors:"
        echo "$errors" | sed 's/^/     /'
    else
        echo "OK  $svc - no recent errors"
    fi
done

echo ""
echo "=== Done ==="
