#!/bin/bash

# Docker
sudo apt update
sudo apt install -y docker.io
sudo usermod -aG docker $USER
newgrp docker  # Cambia de grupo dentro del mismo shell

# NVIDIA Container Toolkit
distribution="ubuntu22.04"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list | \
  sed 's|^deb |deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] |' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

# Toolkit setup Docker with  NVIDIA support
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Docker compose
sudo apt update
sudo apt install docker-compose-plugin

# Git
echo "🔐 Git..."
sudo apt install -y git
git clone https://github.com/ECU-School-of-Engineering/cloud_llm.git

# Docker Compose
echo "🐳 Init Docker Compose..."
cd cloud_llm
docker compose up --build -d
