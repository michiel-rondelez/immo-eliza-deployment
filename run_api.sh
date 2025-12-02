#!/bin/bash
# Start FastAPI server

echo "Starting FastAPI server..."
cd src
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
