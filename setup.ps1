# IVADE Setup Script for Windows
# Run from the project directory: .\setup.ps1

Write-Host "=== IVADE Setup ===" -ForegroundColor Cyan

# Check Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Docker not found. Install Docker Desktop from https://www.docker.com/products/docker-desktop" -ForegroundColor Red
    exit 1
}

# Check Docker is running
try {
    docker info 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw }
} catch {
    Write-Host "ERROR: Docker is not running. Start Docker Desktop and try again." -ForegroundColor Red
    exit 1
}
Write-Host "OK  Docker is running" -ForegroundColor Green

# Check docker compose
if (-not (docker compose version 2>&1 | Select-String "Docker Compose")) {
    Write-Host "ERROR: docker compose not found. Update Docker Desktop to a recent version." -ForegroundColor Red
    exit 1
}
Write-Host "OK  docker compose available" -ForegroundColor Green

# Check .env exists
if (-not (Test-Path ".env")) {
    Write-Host "ERROR: .env file not found. Create it with HF_TOKEN, GITHUB_TOKEN, NGROK_AUTHTOKEN, HUME_API_KEY, OPENAI_API_KEY." -ForegroundColor Red
    exit 1
}
Write-Host "OK  .env found" -ForegroundColor Green

# Check required config files
$required = @("grader_config.yaml", "config/fuzzy_system.yaml", "keys/ivade 1.pem")
foreach ($f in $required) {
    if (-not (Test-Path $f)) {
        Write-Host "WARN: Missing $f — some services may fail to start." -ForegroundColor Yellow
    } else {
        Write-Host "OK  $f found" -ForegroundColor Green
    }
}

# Check NVIDIA GPU (optional)
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "OK  NVIDIA GPU detected" -ForegroundColor Green
} else {
    Write-Host "WARN: nvidia-smi not found — GPU may not be available to containers." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting containers..." -ForegroundColor Cyan
docker compose up -d --build

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=== Done ===" -ForegroundColor Green
    Write-Host "Check status : docker compose ps"
    Write-Host "LLM logs     : docker compose logs -f llm"
    Write-Host "CLM logs     : docker compose logs -f clm"
} else {
    Write-Host "ERROR: docker compose up failed." -ForegroundColor Red
    exit 1
}
