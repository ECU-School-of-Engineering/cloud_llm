# IVADE Diagnostics Script for Windows
# Run from the project directory: .\scripts\diagnose.ps1

Write-Host "=== IVADE Diagnostics ===" -ForegroundColor Cyan
Write-Host ""

# Docker running
Write-Host "--- Docker ---" -ForegroundColor White
try {
    docker info 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw }
    Write-Host "OK  Docker is running" -ForegroundColor Green
} catch {
    Write-Host "FAIL Docker is not running" -ForegroundColor Red
}

# Container status
Write-Host ""
Write-Host "--- Containers ---" -ForegroundColor White
docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

# Health endpoints
Write-Host ""
Write-Host "--- Health Checks ---" -ForegroundColor White
foreach ($check in @(
    @{name="LLM"; url="http://localhost:8001/health"},
    @{name="CLM"; url="http://localhost:8080/health"}
)) {
    try {
        $resp = Invoke-WebRequest -Uri $check.url -TimeoutSec 5 -UseBasicParsing
        $body = $resp.Content | ConvertFrom-Json
        Write-Host ("OK  {0}: {1}" -f $check.name, ($body | ConvertTo-Json -Compress)) -ForegroundColor Green
    } catch {
        Write-Host ("FAIL {0}: not responding at {1}" -f $check.name, $check.url) -ForegroundColor Red
    }
}

# ngrok tunnel
Write-Host ""
Write-Host "--- ngrok Tunnel ---" -ForegroundColor White
try {
    $ngrok = Invoke-WebRequest -Uri "http://localhost:4040/api/tunnels" -TimeoutSec 5 -UseBasicParsing
    $tunnels = ($ngrok.Content | ConvertFrom-Json).tunnels
    if ($tunnels.Count -gt 0) {
        Write-Host ("OK  Public URL: {0}" -f $tunnels[0].public_url) -ForegroundColor Green
    } else {
        Write-Host "WARN ngrok running but no tunnels active" -ForegroundColor Yellow
    }
} catch {
    Write-Host "FAIL ngrok not responding on :4040" -ForegroundColor Red
}

# GPU
Write-Host ""
Write-Host "--- GPU ---" -ForegroundColor White
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
} else {
    Write-Host "WARN nvidia-smi not found on host" -ForegroundColor Yellow
}

# Disk space (models folder)
Write-Host ""
Write-Host "--- Disk ---" -ForegroundColor White
if (Test-Path "models") {
    $size = (Get-ChildItem -Recurse "models" -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB
    Write-Host ("OK  models/ folder: {0:F1} GB" -f $size) -ForegroundColor Green
} else {
    Write-Host "WARN models/ folder not found" -ForegroundColor Yellow
}

# Recent errors in logs
Write-Host ""
Write-Host "--- Recent Errors (last 20 lines each) ---" -ForegroundColor White
foreach ($svc in @("llm", "clm")) {
    $errors = docker compose logs $svc --tail=20 2>&1 | Select-String -Pattern "ERROR|Traceback|Exception" | Select-Object -Last 3
    if ($errors) {
        Write-Host "WARN $svc has recent errors:" -ForegroundColor Yellow
        $errors | ForEach-Object { Write-Host "     $_" -ForegroundColor Yellow }
    } else {
        Write-Host "OK  $svc - no recent errors" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Cyan
