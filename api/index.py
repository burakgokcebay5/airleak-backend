from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Airleak Backend API v2", "status": "operational", "version": "2.0"}

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "platform": "vercel"}

@app.get("/api/test")
def test_endpoint():
    return {"test": "success", "message": "API is working"}