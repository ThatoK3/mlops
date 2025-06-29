#!/bin/bash

set -e

echo "Starting Docker installation..."

# Step 1: Remove old versions
sudo apt remove -y docker docker-engine docker.io containerd runc || true

# Step 2: Update the package index
sudo apt update

# Step 3: Install dependencies
sudo apt install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Step 4: Add Docker’s official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Step 5: Set up the Docker stable repository
echo \
  "deb [arch=$(dpkg --print-architecture) \
  signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Step 6: Update the package index again
sudo apt update

# Step 7: Install Docker Engine and related tools
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Step 8: Enable and start Docker
sudo systemctl enable docker
sudo systemctl start docker

# Step 9: Add current user to docker group (optional but recommended)
sudo usermod -aG docker "$USER"

echo -e "\n✅ Docker installation complete!"
echo "You may need to log out and back in to use Docker without sudo."
docker --version

