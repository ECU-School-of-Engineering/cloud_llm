#!/bin/bash

sudo apt update && sudo apt upgrade -y

sudo apt install -y build-essential dkms linux-headers-$(uname -r)

# drivers NVIDIA 
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
echo "✅ NVIDIA drives installed. Reboot"

# 🔁 Reinicio obligatorio para que nvidia-smi funcione
read -p "¿Reboot now (y/n): " confirm
if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
    sudo reboot
else
    echo "Do manual reboot."
fi
