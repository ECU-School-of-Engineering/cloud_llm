#!/bin/bash
# Generate GitHub SSH key and configure it on this machine
# Usage: ./scripts/setup_github_ssh.sh

echo "=== GitHub SSH Setup ==="

# Create ~/.ssh if needed
mkdir -p ~/.ssh
chmod 700 ~/.ssh

KEY_PATH="$HOME/.ssh/github_ivade"

# Generate key if it doesn't exist
if [ -f "$KEY_PATH" ]; then
    echo "OK  Key already exists at $KEY_PATH"
else
    ssh-keygen -t ed25519 -C "ivade-deploy" -f "$KEY_PATH" -N ""
    echo "OK  Key generated at $KEY_PATH"
fi

# Detect available SSH port by checking for real SSH banner (nc TCP test is unreliable with firewalls)
echo ""
echo "--- Detecting available SSH port ---"
SSH_HOST=""
SSH_PORT=""

test_ssh_banner() {
    local host=$1
    local port=$2
    # Connect and read banner; grep for SSH-
    banner=$(timeout 6 bash -c "exec 3<>/dev/tcp/$host/$port && cat <&3" 2>/dev/null | head -1)
    echo "$banner" | grep -q "^SSH-"
}

echo "    Testing port 22 on github.com..."
if test_ssh_banner "github.com" 22; then
    echo "OK  Port 22 works - using standard SSH"
    SSH_HOST="github.com"
    SSH_PORT=22
else
    echo "WARN Port 22 blocked or no SSH banner - trying ssh.github.com:443"
    echo "    Testing port 443 on ssh.github.com..."
    if test_ssh_banner "ssh.github.com" 443; then
        echo "OK  Port 443 works - using ssh.github.com:443"
        SSH_HOST="ssh.github.com"
        SSH_PORT=443
    else
        echo "FAIL Neither port 22 nor 443 returned an SSH banner. Check your network/firewall."
        exit 1
    fi
fi

# Add SSH config entry for GitHub
if grep -q "Host github.com" ~/.ssh/config 2>/dev/null; then
    echo "OK  ~/.ssh/config already has github.com entry"
else
    cat >> ~/.ssh/config <<EOF

Host github.com
    HostName $SSH_HOST
    Port $SSH_PORT
    User git
    IdentityFile $KEY_PATH
    IdentitiesOnly yes
EOF
    chmod 600 ~/.ssh/config
    echo "OK  Added github.com to ~/.ssh/config (HostName=$SSH_HOST Port=$SSH_PORT)"
fi

# Add GitHub to known_hosts
ssh-keyscan -p $SSH_PORT $SSH_HOST >> ~/.ssh/known_hosts 2>/dev/null
chmod 644 ~/.ssh/known_hosts
echo "OK  $SSH_HOST added to known_hosts"

echo ""
echo "================================================================"
echo "ACTION REQUIRED: Add this public key to GitHub"
echo "  GitHub -> Settings -> SSH and GPG keys -> New SSH key"
echo ""
cat "$KEY_PATH.pub"
echo "================================================================"
echo ""

# Test (will fail until key is added to GitHub)
echo "==> Testing GitHub connection (may fail until key is added)..."
ssh -T git@github.com 2>&1
echo ""
echo "If you see 'Hi <username>!' above, you are done."
echo "If not, add the public key above to GitHub first, then run:"
echo "  ssh -T git@github.com"
