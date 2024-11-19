build_option=$1
docker compose -f docker/docker-compose.yml -p $USER up -d $build_option
