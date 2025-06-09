docker run -d -it --rm \
  --name jupyter-nb-exps \
  -p 8888:8888 \
  --network=host \
  -v $(pwd):/home/jovyan/work \
  jupyter-mlops-exps  
