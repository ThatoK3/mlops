#!/bin/bash

set -a
source .env
set +a

docker stop mlflow-server
docker rm mlflow-server

docker run -d \
  --name mlflow-server \
  --network="host" \
  -v $(pwd)/notebook_experiments/mlruns:/mlflow/mlruns \
  thatojoe/mlflow-mysql \
  mlflow server \
    --backend-store-uri mysql+pymysql://$MYSQL_USER:$MYSQL_PASSWORD@$MYSQL_HOST:$MYSQL_PORT/$MYSQL_DB \
    --default-artifact-root file:/mlflow/mlruns \
    --host 0.0.0.0 \
    --port 5000

