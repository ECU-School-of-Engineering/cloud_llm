# Generate GitHub SSH key and configure it on this Windows machine
# Usage: .\scripts\setup_github_ssh.ps1

Write-Host "=== GitHub SSH Setup ===" -ForegroundColor Cyan

$sshDir = "$env:USERPROFILE\.ssh"
$keyPath = "$sshDir\github_ivade"

# Create ~/.ssh
if (-not (Test-Path $sshDir)) {
    New-Item -ItemType Directory -Path $sshDir | Out-Null
}

# Generate key
if (Test-Path $keyPath) {
    Write-Host "OK  Key already exists at $keyPath" -ForegroundColor Green
} else {
    ssh-keygen -t ed25519 -C "ivade-deploy" -f $keyPath -N '""'
    Write-Host "OK  Key generated at $keyPath" -ForegroundColor Green
}

# Print public key immediately
Write-Host ""
Write-Host "================================================================" -ForegroundColor Yellow
Write-Host "ACTION REQUIRED: Add this public key to GitHub" -ForegroundColor Yellow
Write-Host "  GitHub -> Settings -> SSH and GPG keys -> New SSH key" -ForegroundColor Yellow
Write-Host ""
Get-Content "$keyPath.pub"
Write-Host "================================================================" -ForegroundColor Yellow
Write-Host ""

# Detect available SSH port by testing actual SSH banner (TCP test is unreliable with firewalls)
Write-Host "--- Detecting available SSH port ---" -ForegroundColor White
$sshHost = $null
$sshPort = $null

function Test-SshPort {
    param($host_, $port)
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $connect = $tcp.BeginConnect($host_, $port, $null, $null)
        $wait = $connect.AsyncWaitHandle.WaitOne(5000, $false)
        if (-not $wait) { $tcp.Close(); return $false }
        $tcp.EndConnect($connect)
        # Read SSH banner to confirm real SSH (not just a firewall ACK)
        $stream = $tcp.GetStream()
        $stream.ReadTimeout = 5000
        $buf = New-Object byte[] 64
        $read = $stream.Read($buf, 0, $buf.Length)
        $tcp.Close()
        $banner = [System.Text.Encoding]::ASCII.GetString($buf, 0, $read)
        return $banner -match "^SSH-"
    } catch {
        return $false
    }
}

Write-Host "    Testing port 22 on github.com..." -ForegroundColor Gray
if (Test-SshPort "github.com" 22) {
    Write-Host "OK  Port 22 works - using standard SSH" -ForegroundColor Green
    $sshHost = "github.com"
    $sshPort = 22
} else {
    Write-Host "WARN Port 22 blocked or no SSH banner - trying ssh.github.com:443" -ForegroundColor Yellow
    Write-Host "    Testing port 443 on ssh.github.com..." -ForegroundColor Gray
    if (Test-SshPort "ssh.github.com" 443) {
        Write-Host "OK  Port 443 works - using ssh.github.com:443" -ForegroundColor Green
        $sshHost = "ssh.github.com"
        $sshPort = 443
    } else {
        Write-Host "FAIL Neither port 22 nor 443 returned an SSH banner. Check your network/firewall." -ForegroundColor Red
        exit 1
    }
}

# Add SSH config entry for GitHub
$configPath = "$sshDir\config"
$configEntry = @"

Host github.com
    HostName $sshHost
    Port $sshPort
    User git
    IdentityFile $keyPath
    IdentitiesOnly yes
"@

if (Test-Path $configPath) {
    $existing = Get-Content $configPath -Raw
    if ($existing -match "Host github.com") {
        # Remove existing github.com block and rewrite it
        $updated = $existing -replace "(?s)Host github\.com.*?(?=\nHost |\z)", ""
        Set-Content $configPath $updated.TrimEnd()
        Add-Content $configPath $configEntry
        Write-Host "OK  Updated github.com entry in $configPath (HostName=$sshHost Port=$sshPort)" -ForegroundColor Green
    } else {
        Add-Content $configPath $configEntry
        Write-Host "OK  Added github.com to $configPath (HostName=$sshHost Port=$sshPort)" -ForegroundColor Green
    }
} else {
    Set-Content $configPath $configEntry
    Write-Host "OK  Created $configPath (HostName=$sshHost Port=$sshPort)" -ForegroundColor Green
}

# Add GitHub to known_hosts
Write-Host "==> Adding github.com to known_hosts..."
ssh-keyscan -p $sshPort $sshHost 2>$null >> "$sshDir\known_hosts"
Write-Host "OK  Done" -ForegroundColor Green

# Test
Write-Host ""
Write-Host "==> Testing GitHub connection (may fail until key is added to GitHub)..."
ssh -T git@github.com 2>&1
Write-Host ""
Write-Host "If you see 'Hi <username>!' above, you are done."
Write-Host "If not, add the public key above to GitHub first, then run:"
Write-Host "  ssh -T git@github.com"
