# Deploy to Barry via GitHub Actions
# Usage: .\deploy.ps1 ["optional commit message"]

param(
    [string]$Msg = "deploy $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
)

Write-Host "==> Committing and pushing..." -ForegroundColor Cyan
git add -A
git commit -m $Msg 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "(nothing to commit)" -ForegroundColor Gray }
git push origin dev_docker

$repo = git remote get-url origin | ForEach-Object { $_ -replace '.*github\.com[:/]', '' -replace '\.git$', '' }
Write-Host ""
Write-Host "==> Push done. GitHub Actions will now:" -ForegroundColor Green
Write-Host "    1. Run deploy.yml on Barry's self-hosted runner"
Write-Host "    2. git pull + docker compose restart clm"
Write-Host ""
Write-Host "Monitor at: https://github.com/$repo/actions"
