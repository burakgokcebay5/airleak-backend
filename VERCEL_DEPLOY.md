# Vercel Deployment Guide

Backend repository has been pushed to GitHub with minimal FastAPI setup optimized for Vercel.

## Deploy to Vercel (Web Interface)

1. Go to https://vercel.com
2. Log in with your GitHub account
3. Click "New Project"
4. Import the repository: `burakgokcebay5/airleak-backend`
5. Configure:
   - Framework Preset: Other
   - Root Directory: ./
   - Build Command: (leave empty)
   - Output Directory: (leave empty)
6. Click "Deploy"

## Files Updated

- `api/index.py` - Minimal FastAPI app without heavy dependencies
- `vercel.json` - Simple rewrite configuration
- `requirements.txt` - Only FastAPI and uvicorn

## Test Endpoints

Once deployed, test these endpoints:
- `/` - Root endpoint
- `/api/health` - Health check
- `/api/test` - Test endpoint

## Next Steps

After successful deployment:
1. Copy the Vercel URL (e.g., https://airleak-backend.vercel.app)
2. Update frontend `.env` file with new URL
3. Rebuild and redeploy frontend