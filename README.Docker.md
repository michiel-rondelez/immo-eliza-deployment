# Docker Setup Guide

This guide explains how to run the Immo Eliza ML project using Docker.

## Prerequisites

- Docker Desktop installed and running
- Docker Compose (usually included with Docker Desktop)

## Quick Start

### 1. Build and Start Services

```bash
docker-compose up --build
```

This will:
- Build both the FastAPI backend and Streamlit frontend images
- Start both services
- Make them available at:
  - **FastAPI**: http://localhost:8000
  - **Streamlit**: http://localhost:8501

### 2. Access the Applications

- **Streamlit UI**: Open http://localhost:8501 in your browser
- **FastAPI Docs**: Open http://localhost:8000/docs for interactive API documentation

### 3. Stop Services

```bash
docker-compose down
```

## Development Workflow

### Rebuild After Code Changes

If you modify code, rebuild the images:

```bash
docker-compose up --build
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f streamlit
```

### Run Individual Services

```bash
# Start only the API
docker-compose up api

# Start only Streamlit
docker-compose up streamlit
```

## Volume Mounts

The `docker-compose.yml` mounts:
- `./models` → `/app/models` (for trained models and preprocessors)
- `./data` → `/app/data` (for data files)

This means models and data persist on your host machine and are accessible to both containers.

## Troubleshooting

### Port Already in Use

If ports 8000 or 8501 are already in use, modify the port mappings in `docker-compose.yml`:

```yaml
ports:
  - "8001:8000"  # Change host port from 8000 to 8001
```

### Rebuild from Scratch

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Check Container Status

```bash
docker-compose ps
```

### Execute Commands Inside Container

```bash
# API container
docker-compose exec api bash

# Streamlit container
docker-compose exec streamlit bash
```

## Production Considerations

For production deployment:

1. **Use specific image tags** instead of `latest`
2. **Set environment variables** via `.env` file (not included in repo)
3. **Use secrets management** for sensitive data
4. **Configure reverse proxy** (nginx/traefik) for SSL termination
5. **Set resource limits** in docker-compose.yml:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
```

