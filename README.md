# Airleak Backend API

Professional LOB (Leak-Off-Brake) Test Analysis Platform Backend

## Features

- Excel file upload and processing
- Advanced data analysis and pattern recognition
- Machine learning insights
- Pressure range optimization
- Test result predictions
- Channel health monitoring
- Firebase integration

## Tech Stack

- **FastAPI** - Modern web framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning
- **Python 3.11**

## API Endpoints

- `POST /api/upload-excel` - Upload and analyze Excel files
- `GET /api/discovered-patterns` - Get pattern analysis
- `GET /api/channel-health` - Channel health metrics
- `POST /api/optimize` - Run optimization algorithms
- `GET /api/ml-insights` - Machine learning insights
- `POST /api/predict` - Predict test outcomes
- `GET /api/statistics` - Statistical analysis

## Deployment

This backend is designed to be deployed on:
- Render.com
- Google Cloud Run
- Railway.app
- Any Docker-compatible platform

## Environment Variables

- `PORT` - Server port (default: 8000)
- `CORS_ORIGINS` - Allowed CORS origins

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn main_perfect:app --reload --port 8000
```

## Docker

```bash
# Build image
docker build -t airleak-backend .

# Run container
docker run -p 8000:8000 airleak-backend
```

## Frontend

Frontend is deployed at: https://airleak.web.app

## License

Private - All rights reserved