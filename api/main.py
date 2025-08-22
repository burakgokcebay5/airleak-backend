"""
Vercel Serverless Function Handler
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import FastAPI app from main_perfect
try:
    from main_perfect import app
except ImportError:
    # If main_perfect fails, create a simple app
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI()
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        return {"message": "Airleak Backend API on Vercel", "status": "operational"}

# Export handler for Vercel
handler = app