#!/bin/bash
# Set up a GitHub Actions self-hosted runner on Barry (one-time setup)
# Usage: ./scripts/setup_runner.sh <GITHUB_REPO_URL> <RUNNER_TOKEN>
#
# Get the runner token from:
#   GitHub repo -> Settings -> Actions -> Runners -> New self-hosted runner
#   Copy the token from the "Configure" step (starts with A...)
#
# Example:
#   ./scripts/setup_runner.sh https://github.com/YOUR_ORG/cloud_llm AABBCC...

set -e

REPO_URL=${1:?"Usage: $0 <github-repo-url> <runner-token>"}
TOKEN=${2:?"Usage: $0 <github-repo-url> <runner-token>"}
RUNNER_DIR="/opt/actions-runner"
RUNNER_VERSION="2.321.0"

echo "=== GitHub Actions Runner Setup ==="

# Create runner directory
sudo mkdir -p $RUNNER_DIR
sudo chown $USER:$USER $RUNNER_DIR
cd $RUNNER_DIR

# Download runner if not already present
if [ ! -f "./run.sh" ]; then
    echo "==> Downloading runner v$RUNNER_VERSION..."
    curl -sL "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
        | tar xz
    echo "OK  Runner extracted"
fi

# Configure runner
echo "==> Configuring runner..."
./config.sh \
    --url "$REPO_URL" \
    --token "$TOKEN" \
    --name "barry" \
    --labels "self-hosted,barry,linux,x64" \
    --work "/opt/actions-runner/_work" \
    --unattended \
    --replace

# Install as a systemd service so it starts on boot
echo "==> Installing runner as systemd service..."
sudo ./svc.sh install $USER
sudo ./svc.sh start

echo ""
echo "OK  Runner installed and started."
echo "    Check status: sudo /opt/actions-runner/svc.sh status"
echo "    View logs:    journalctl -u actions.runner.* -f"
echo ""
echo "Verify on GitHub: Settings -> Actions -> Runners"
echo "Barry should appear as 'Idle' (green dot)."
