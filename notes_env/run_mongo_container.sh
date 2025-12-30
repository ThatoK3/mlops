docker run -d \
  --name mongodb \
  --network stroke-predict-spark-feast_stroke-network \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=password123 \
  -e MONGO_INITDB_DATABASE=stroke_prediction \
  -v mongodb_data:/data/db \
  mongo:latest
