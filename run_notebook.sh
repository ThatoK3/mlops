docker run -d -it --rm \
  --name jupyter-nb-exps \
  --user root -e GRANT_SUDO=yes\
  -p 8888:8888 \
  --network=host \
  -v $(pwd):/home/jovyan/work \
  -v $(pwd)/notebook_experiments/mlruns:/mlflow/mlruns \
  thatojoe/jupyter-mlops-exps  
