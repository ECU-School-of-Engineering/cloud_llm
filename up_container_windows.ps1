Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "         Starting IVADE containers..." -ForegroundColor Yellow
Write-Host "============================================================`n" -ForegroundColor Yellow

# Start all containers
docker compose up -d
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to start containers" -ForegroundColor Red
    exit 1
}

Write-Host "Waiting for containers to be ready..." -ForegroundColor Yellow

# Wait for llm container to be running
Write-Host "  Checking llm container..." -ForegroundColor Yellow
while ((docker inspect -f '{{.State.Running}}' cloud_llm-llm-1 2>$null) -ne "true") {
    Write-Host "  - llm is starting..." -ForegroundColor Yellow
    Start-Sleep -Seconds 2
}
Write-Host "  + llm is running" -ForegroundColor Green

# Wait for clm container to be running
Write-Host "  Checking clm container..." -ForegroundColor Yellow
while ((docker inspect -f '{{.State.Running}}' cloud_llm-clm-1 2>$null) -ne "true") {
    Write-Host "  - clm is starting..." -ForegroundColor Yellow
    Start-Sleep -Seconds 2
}
Write-Host "  + clm is running" -ForegroundColor Green

# Wait for LLM service health (model load takes a while)
Write-Host "  Waiting for LLM service to be ready (model loading...)..." -ForegroundColor Yellow
$retryCount = 0
while ($retryCount -lt 120) {
    try {
        $null = Invoke-WebRequest -Uri "http://localhost:8001/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        Write-Host "  + LLM service is ready" -ForegroundColor Green
        break
    } catch {
        $retryCount++
        Write-Host "  - LLM loading (attempt $retryCount/120)..." -ForegroundColor Yellow
        Start-Sleep -Seconds 5
    }
}

# Wait for CLM FastAPI
Write-Host "  Waiting for CLM service to be ready..." -ForegroundColor Yellow
$retryCount = 0
while ($retryCount -lt 60) {
    try {
        $null = Invoke-WebRequest -Uri "http://localhost:8080/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        Write-Host "  + CLM service is ready" -ForegroundColor Green
        break
    } catch {
        $retryCount++
        Write-Host "  - CLM starting (attempt $retryCount/60)..." -ForegroundColor Yellow
        Start-Sleep -Seconds 2
    }
}

if ($retryCount -eq 60) {
    Write-Host "CLM service failed to start" -ForegroundColor Red
    exit 1
}

# Wait for ngrok tunnel
Write-Host "  Waiting for ngrok tunnel..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

$ngrokUrl = $null
$retryCount = 0
while ($retryCount -lt 30) {
    try {
        $tunnels = Invoke-RestMethod -Uri "http://localhost:4040/api/tunnels" -ErrorAction Stop
        $ngrokUrl = $tunnels.tunnels[0].public_url
        if ($ngrokUrl) {
            Write-Host "  + ngrok tunnel established" -ForegroundColor Green
            break
        }
    } catch {}
    $retryCount++
    Write-Host "  - Waiting for ngrok (attempt $retryCount/30)..." -ForegroundColor Yellow
    Start-Sleep -Seconds 2
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "              IVADE is Ready!" -ForegroundColor Green
Write-Host "============================================================`n" -ForegroundColor Green

Write-Host "Local URLs:" -ForegroundColor Yellow
Write-Host "  CLM:  http://localhost:8080"
Write-Host "  LLM:  http://localhost:8001"
Write-Host "  ngrok dashboard: http://localhost:4040`n"

if ($ngrokUrl) {
    Write-Host "Public ngrok URL:" -ForegroundColor Yellow
    Write-Host "  $ngrokUrl`n" -ForegroundColor Green
} else {
    Write-Host "ngrok URL (check dashboard):" -ForegroundColor Yellow
    Write-Host "  http://localhost:4040`n"
    Start-Process "http://localhost:4040"
}

Write-Host "Test Commands:" -ForegroundColor Yellow
Write-Host "  curl.exe http://localhost:8080/health"
Write-Host "  curl.exe http://localhost:8001/health"
Write-Host "  curl.exe $ngrokUrl/chat/sessions`n"

Write-Host "Monitor logs:" -ForegroundColor Yellow
Write-Host "  docker compose logs -f clm"
Write-Host "  docker compose logs -f llm`n"

Write-Host "To stop:" -ForegroundColor Yellow
Write-Host "  docker compose down`n"
