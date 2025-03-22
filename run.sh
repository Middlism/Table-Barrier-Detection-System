#!/bin/bash

# This script helps with running the Table Barrier Detection System

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    echo "Visit https://docs.docker.com/get-docker/ for installation instructions."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit https://docs.docker.com/compose/install/ for installation instructions."
    exit 1
fi

# If on Linux, allow Docker to use X server
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Setting up X server permissions for Docker..."
    xhost +local:docker
fi

# Build and run the container
echo "Building and starting the Table Barrier Detection System..."
docker-compose up --build

echo "Application stopped."