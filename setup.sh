#!/bin/bash

# Install Docker
sudo apt update
sudo apt install -y docker.io
sudo usermod -aG docker $USER
newgrp docker

# Install NVIDIA Container Toolkit
distribution="ubuntu22.04"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list | \
  sed 's|^deb |deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] |' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker


#Git
echo "🔐 Git..."
sudo apt install -y git
git clone https://github.com/ECU-School-of-Engineering/cloud_llm.git

# Docker Compose: build and run
echo "🐳 Init Docker Compose..."
cd cloud_llm
docker compose --compatibility up --build -d