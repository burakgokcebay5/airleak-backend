"""
Vercel Serverless Function Handler for FastAPI
"""
import sys
import os
from pathlib import Path

# Add parent directory to path to import main_perfect
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the FastAPI app
from main_perfect import app

# Create handler for Vercel
from fastapi import FastAPI
from mangum import Mangum

# Mangum is an adapter for running ASGI applications in AWS Lambda or Vercel
handler = Mangum(app)

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)