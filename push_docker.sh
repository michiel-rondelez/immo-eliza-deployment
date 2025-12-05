#!/bin/bash
set -e

# Configuration
DOCKER_USERNAME="supermichie3000"  # Change this to your Docker Hub username

echo "🚀 Pushing Docker containers to Docker Hub"
echo "=========================================="
echo ""

# Login to Docker Hub
echo "📝 Step 1: Logging in to Docker Hub..."
docker login

echo ""
echo "🔨 Step 2: Building and pushing API container (linux/amd64)..."
docker build --platform linux/amd64 -f Dockerfile.api -t ${DOCKER_USERNAME}/immo-eliza-api:latest .
docker push ${DOCKER_USERNAME}/immo-eliza-api:latest

echo ""
echo "🔨 Step 3: Building and pushing Streamlit container (linux/amd64)..."
docker build --platform linux/amd64 -f Dockerfile.streamlit -t ${DOCKER_USERNAME}/immo-eliza-streamlit:latest .
docker push ${DOCKER_USERNAME}/immo-eliza-streamlit:latest

echo ""
echo "✅ Success! Both containers pushed to Docker Hub"
echo ""
echo "📦 Container URLs:"
echo "   API:      https://hub.docker.com/r/${DOCKER_USERNAME}/immo-eliza-api"
echo "   Streamlit: https://hub.docker.com/r/${DOCKER_USERNAME}/immo-eliza-streamlit"
echo ""
echo "💡 To pull and run:"
echo "   docker pull ${DOCKER_USERNAME}/immo-eliza-api:latest"
echo "   docker pull ${DOCKER_USERNAME}/immo-eliza-streamlit:latest"
