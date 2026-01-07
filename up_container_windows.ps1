Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "         Starting Barry Cloud LLM containers..." -ForegroundColor Yellow
Write-Host "============================================================`n" -ForegroundColor Yellow

# Start containers
docker-compose up -d
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to start containers" -ForegroundColor Red
    exit 1
}

Write-Host "Waiting for containers to be ready..." -ForegroundColor Yellow

# Wait for host_clm
Write-Host "  Checking host_clm container..." -ForegroundColor Yellow
while ((docker inspect -f '{{.State.Running}}' cloud_llm-host_clm-1 2>$null) -ne "true") {
    Write-Host "  - host_clm is starting..." -ForegroundColor Yellow
    Start-Sleep -Seconds 2
}
Write-Host "  + host_clm is running" -ForegroundColor Green

# Wait for ngrok
Write-Host "  Checking ngrok container..." -ForegroundColor Yellow
while ((docker inspect -f '{{.State.Running}}' cloud_llm-ngrok-1 2>$null) -ne "true") {
    Write-Host "  - ngrok is starting..." -ForegroundColor Yellow
    Start-Sleep -Seconds 2
}
Write-Host "  + ngrok is running" -ForegroundColor Green

# Wait for FastAPI
Write-Host "  Waiting for FastAPI to be ready..." -ForegroundColor Yellow
$retryCount = 0
while ($retryCount -lt 60) {
    try {
        $null = Invoke-WebRequest -Uri "http://localhost:8080/chat/sessions" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        Write-Host "  + FastAPI is ready" -ForegroundColor Green
        break
    } catch {
        $retryCount++
        Write-Host "  - FastAPI is starting (attempt $retryCount/60)..." -ForegroundColor Yellow
        Start-Sleep -Seconds 2
    }
}

if ($retryCount -eq 60) {
    Write-Host "FastAPI failed to start" -ForegroundColor Red
    exit 1
}

# Wait for ngrok
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
    Write-Host "  - Waiting for ngrok tunnel (attempt $retryCount/30)..." -ForegroundColor Yellow
    Start-Sleep -Seconds 2
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "              Barry LLM is Ready!" -ForegroundColor Green
Write-Host "============================================================`n" -ForegroundColor Green

Write-Host "Local URL:" -ForegroundColor Yellow
Write-Host "  http://localhost:8080`n"

if ($ngrokUrl) {
    Write-Host "Public ngrok URL:" -ForegroundColor Yellow
    Write-Host "  $ngrokUrl`n" -ForegroundColor Green
} else {
    Write-Host "ngrok URL (check dashboard):" -ForegroundColor Yellow
    Write-Host "  http://localhost:4040`n"
    Start-Process "http://localhost:4040"
}

Write-Host "Test Commands:" -ForegroundColor Yellow
Write-Host "  curl.exe $ngrokUrl/chat/new_session`n"

Write-Host "To stop:" -ForegroundColor Yellow
Write-Host "  docker-compose down`n"