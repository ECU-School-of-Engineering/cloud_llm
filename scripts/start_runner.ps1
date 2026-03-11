# Start the GitHub Actions runner in a dedicated console window
# Usage: .\scripts\start_runner.ps1

$RunnerDir = "C:\actions-runner"

if (-not (Test-Path "$RunnerDir\run.cmd")) {
    Write-Host "FAIL Runner not found at $RunnerDir. Run setup_runner.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "==> Starting GitHub Actions runner..." -ForegroundColor Cyan

Start-Process cmd -ArgumentList "/k", "title GitHub Actions Runner - Barry && cd /d $RunnerDir && run.cmd"

Write-Host "OK  Runner window opened. It will show 'Listening for Jobs' when ready." -ForegroundColor Green
Write-Host "    Keep that window open while deploying."
Write-Host ""
Write-Host "To run as a background service instead (survives reboots):"
Write-Host "  cd $RunnerDir"
Write-Host "  .\svc.ps1 install"
Write-Host "  .\svc.ps1 start"
