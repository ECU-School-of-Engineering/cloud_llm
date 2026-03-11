# Open 4 console windows, each tailing logs for one Docker service
# Usage: .\scripts\monitor.ps1
# Run from the repo root (where docker-compose.yml is)

$repoRoot = Split-Path -Parent $PSScriptRoot

$services = @("llm", "clm", "ngrok", "tunnel")

foreach ($svc in $services) {
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "cd '$repoRoot'; `$host.UI.RawUI.WindowTitle = 'IVADE: $svc'; docker compose logs -f $svc"
    )
}

Write-Host "Opened 4 log windows (llm, clm, ngrok, tunnel)." -ForegroundColor Cyan
