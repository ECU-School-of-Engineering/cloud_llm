# Set up a GitHub Actions self-hosted runner on this Windows machine (one-time setup)
# Run as Administrator in PowerShell
#
# Usage: .\scripts\setup_runner.ps1 -RepoUrl <url> -Token <token>
#
# Get the token from:
#   GitHub repo -> Settings -> Actions -> Runners -> New self-hosted runner
#   Select Windows, copy the --token value from the Configure step
#
# Example:
#   .\scripts\setup_runner.ps1 -RepoUrl https://github.com/YOUR_ORG/cloud_llm -Token AABBCC...

param(
    [Parameter(Mandatory)][string]$RepoUrl,
    [Parameter(Mandatory)][string]$Token
)

$RunnerDir = "C:\actions-runner"
$RunnerVersion = "2.321.0"

Write-Host "=== GitHub Actions Runner Setup ===" -ForegroundColor Cyan

# Create runner directory
if (-not (Test-Path $RunnerDir)) {
    New-Item -ItemType Directory -Path $RunnerDir | Out-Null
}
Set-Location $RunnerDir

# Download runner if not already present
if (-not (Test-Path ".\run.cmd")) {
    Write-Host "==> Downloading runner v$RunnerVersion..." -ForegroundColor White
    $url = "https://github.com/actions/runner/releases/download/v${RunnerVersion}/actions-runner-win-x64-${RunnerVersion}.zip"
    Invoke-WebRequest -Uri $url -OutFile "runner.zip" -UseBasicParsing
    Expand-Archive -Path "runner.zip" -DestinationPath $RunnerDir -Force
    Remove-Item "runner.zip"
    Write-Host "OK  Runner extracted" -ForegroundColor Green
}

# Configure runner
Write-Host "==> Configuring runner..." -ForegroundColor White
.\config.cmd `
    --url $RepoUrl `
    --token $Token `
    --name "barry" `
    --labels "self-hosted,barry,windows,x64" `
    --work "C:\actions-runner\_work" `
    --unattended `
    --replace

# Install as a Windows service so it starts on boot
Write-Host "==> Installing runner as Windows service..." -ForegroundColor White
.\svc.ps1 install
.\svc.ps1 start

Write-Host ""
Write-Host "OK  Runner installed and started." -ForegroundColor Green
Write-Host "    Check status:  .\svc.ps1 status"
Write-Host "    View logs:     Get-EventLog -LogName Application -Source 'actions.runner.*' -Newest 20"
Write-Host ""
Write-Host "Verify on GitHub: Settings -> Actions -> Runners"
Write-Host "Barry should appear as 'Idle' (green dot)."
