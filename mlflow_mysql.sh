docker run -d \
  --name mlflow-mysql \
  -e MYSQL_ROOT_PASSWORD=rootpass \
  -e MYSQL_DATABASE=mlflow_db \
  -e MYSQL_USER=mlflow \
  -e MYSQL_PASSWORD=mlflowpass \
  -p 3306:3306 \
  thatojoe/mysql
