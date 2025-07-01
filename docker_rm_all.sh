echo "Stopping all running Docker containers..."
docker stop $(docker ps -aq) 2>/dev/null || echo "No containers running."

echo "Removing all Docker containers..."
docker rm $(docker ps -aq) 2>/dev/null || echo "No containers to remove."

echo -e "\n✅ Docker cleanup complete."
