docker run -d \
  --name mlflow-server \
  --network="host" \
  -v $(pwd)/mlruns:/mlflow/mlruns \
  mlflow-mysql \
  mlflow server \
    --backend-store-uri mysql+pymysql://mlflow:mlflowpass@45.151.153.107:3306/mlflow_db \
    --default-artifact-root /mlflow/mlruns \
    --host 0.0.0.0 \
    --port 5000
