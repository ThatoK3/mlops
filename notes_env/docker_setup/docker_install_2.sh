#!/bin/bash
set -e

echo "=== Installing Docker ==="

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Cannot detect OS type."
    exit 1
fi

echo "Detected OS: $OS"

# Remove old Docker versions
sudo apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true
sudo yum remove -y docker docker-client docker-client-latest docker-common docker-latest docker-latest-logrotate docker-logrotate docker-engine 2>/dev/null || true

# Install dependencies and Docker based on OS
if [[ "$OS" == "ubuntu" || "$OS" == "debian" ]]; then
    echo "Installing Docker on Debian/Ubuntu..."
    sudo apt-get update -y
    sudo apt-get install -y ca-certificates curl gnupg lsb-release

    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/$OS/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/$OS \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

elif [[ "$OS" == "centos" || "$OS" == "rhel" || "$OS" == "fedora" ]]; then
    echo "Installing Docker on CentOS/RHEL/Fedora..."
    sudo yum remove -y podman runc || true
    sudo yum install -y yum-utils
    sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
    sudo yum install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

elif [[ "$OS" == "amzn" || "$OS" == "amazon" ]]; then
    echo "Installing Docker on Amazon Linux..."
    sudo yum update -y
    sudo amazon-linux-extras install docker -y
else
    echo "Unsupported OS: $OS"
    exit 1
fi

# Enable and start Docker
sudo systemctl enable docker
sudo systemctl start docker

# Add current user to docker group
if ! groups $USER | grep -q docker; then
    echo "Adding user $USER to docker group..."
    sudo usermod -aG docker $USER
fi

# Verify installation
echo "=== Docker Installation Complete ==="
docker --version
docker compose version

echo "To use Docker without sudo, log out and back in or run: newgrp docker"
