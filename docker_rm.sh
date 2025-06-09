docker rm mlflow-server
docker rm mlflow-mysql
sudo kill $(sudo lsof -t -i:3306)
sudo kill $(sudo lsof -t -i:5000)

