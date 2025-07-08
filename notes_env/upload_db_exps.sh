mysql -u root -prootpass -h 103.6.171.147 -e "DROP DATABASE mysql_db;"
mysql -u root -prootpass -h 103.6.171.147 -e "CREATE DATABASE mysql_db;"
mysql -u root -prootpass -h 103.6.171.147 mlflow_db < backup.sql
