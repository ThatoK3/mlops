#!/bin/bash

set -e

echo "Starting Jenkins installation..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install Java (required for Jenkins)
echo "Installing Java (OpenJDK 17)..."
sudo apt install -y openjdk-17-jdk

# Add Jenkins repository and key
echo "Adding Jenkins repository..."
curl -fsSL https://pkg.jenkins.io/debian-stable/jenkins.io.key | sudo tee \
  /usr/share/keyrings/jenkins-keyring.asc > /dev/null

echo "deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] \
  https://pkg.jenkins.io/debian-stable binary/" | sudo tee \
  /etc/apt/sources.list.d/jenkins.list > /dev/null

# Update again and install Jenkins
sudo apt update
sudo apt install -y jenkins

# Enable and start Jenkins
sudo systemctl enable jenkins
sudo systemctl start jenkins

# Show status
echo "Jenkins status:"
sudo systemctl status jenkins --no-pager

echo -e "\n✅ Jenkins installation complete!"
echo "Access Jenkins at: http://localhost:8080"
echo "Initial Admin Password:"
sudo cat /var/lib/jenkins/secrets/initialAdminPassword
