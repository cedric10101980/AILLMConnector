#!/bin/bash

docker compose down
# Remove the old Docker image
docker rmi -f outboundacrcicd.azurecr.io/outbound/aillmprocessor:main
docker rmi -f aillmmodel-app:latest

# Run the Docker container
docker compose up -d