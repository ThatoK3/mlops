docker rm mlflow-server
docker rm mlflow-mysql
docker rm jupyter-nb-exps
sudo kill $(sudo lsof -t -i:3306)
sudo kill $(sudo lsof -t -i:5000)
sudo kill $(sudo lsof -t -i:8888)
