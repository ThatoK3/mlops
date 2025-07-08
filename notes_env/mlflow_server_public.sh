#!/bin/bash

# Load environment variables
set -a
source .env
set +a

# Generate credentials file if it doesn't exist
if [ ! -f "$(pwd)/notebook_experiments/.mlflow_auth" ]; then
  echo "Generating secure credentials..."
  MLFLOW_USER="admin_$(openssl rand -hex 3)"
  MLFLOW_PASS="$(openssl rand -hex 16)"
  echo "$MLFLOW_USER:$MLFLOW_PASS" > "$(pwd)/notebook_experiments/.mlflow_auth"
  echo "Credentials saved to: $(pwd)/notebook_experiments/.mlflow_auth"
fi

# Stop existing container
echo "Reconfiguring mlflow-server for public access..."
docker stop mlflow-server 2>/dev/null
docker rm mlflow-server 2>/dev/null

# Restart with public access and security
docker run -d \
  --name mlflow-server \
  -p 5000:5000 \
  -v "$(pwd)/notebook_experiments/mlruns:/mlruns" \
  -v "$(pwd)/notebook_experiments/.mlflow_auth:/.mlflow_auth" \
  thatojoe/mlflow-mysql \
  bash -c 'mkdir -p /mlflow && \
          apt-get update && \
          apt-get install -y apache2-utils && \
          htpasswd -bc /mlflow/.htpasswd $(head -n1 /.mlflow_auth | cut -d: -f1) $(head -n1 /.mlflow_auth | cut -d: -f2) && \
          mlflow server \
            --backend-store-uri "mysql+pymysql://'"${MYSQL_USER}"':'"${MYSQL_PASSWORD}"'@'"${MYSQL_HOST}"':'"${MYSQL_PORT}"'/'"${MYSQL_DB}"'" \
            --default-artifact-root "file:///mlruns" \
            --host 0.0.0.0 \
            --port 5000 \
            --app-name basic-auth'

# Configure firewall
echo "Configuring firewall..."
sudo ufw allow 5000/tcp 2>/dev/null
sudo ufw enable 2>/dev/null

# Get public IP
PUBLIC_IP=$(curl -s ifconfig.me)

echo "✅ MLflow server now publicly available with BASIC AUTH:"
echo "   - URL: http://$PUBLIC_IP:5000"
echo "   - Username: $(cut -d: -f1 "$(pwd)/notebook_experiments/.mlflow_auth")"
echo "   - Password: $(cut -d: -f2 "$(pwd)/notebook_experiments/.mlflow_auth")"
