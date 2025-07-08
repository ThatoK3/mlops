#!/bin/bash

# Load environment variables from .env
set -a
source .env
set +a

# Stop and remove existing container if it exist
docker stop mlflow-mysql
docker rm mlflow-mysql


docker run -d \
  --name mlflow-mysql \
  -e MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD} \
  -e MYSQL_DATABASE=${MYSQL_DB} \
  -e MYSQL_USER=${MYSQL_USER} \
  -e MYSQL_PASSWORD=${MYSQL_PASSWORD} \
  -p 3306:${MYSQL_PORT} \
  thatojoe/mysql
