#!/bin/bash

# Load environment variables from .env
set -a
source .env
set +a

# Stop and remove existing container if it exists
echo "Stopping and removing existing mlflow-server container if it exists..."
docker stop mlflow-server 2>/dev/null
docker rm mlflow-server 2>/dev/null

# Run MLflow server container
echo "Starting mlflow-server on port 5000..."
docker run -d \
  --name mlflow-server \
  --network="host" \
  -v "$(pwd)/notebook_experiments/mlruns:/mlruns" \
  thatojoe/mlflow-mysql \
  mlflow server \
    --backend-store-uri "mysql+pymysql://${MYSQL_USER}:${MYSQL_PASSWORD}@${MYSQL_HOST}:${MYSQL_PORT}/${MYSQL_DB}" \
    --default-artifact-root "file:///mlruns" \
    --host 0.0.0.0 \
    --port 5000

# Confirm status
if [ $? -eq 0 ]; then
  echo "✅ MLflow server started successfully at http://localhost:5000"
else
  echo "❌ Failed to start MLflow server"
fi
