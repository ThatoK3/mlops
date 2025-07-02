docker exec -d mlflow-server \
  mlflow server \
    --backend-store-uri file:/mlflow/mlruns \
    --default-artifact-root file:/mlflow/mlruns \
    --host 0.0.0.0 \
    --port 5001
