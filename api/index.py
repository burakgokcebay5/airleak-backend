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
    return {"message": "Airleak Backend API", "status": "operational"}

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "platform": "vercel"}

@app.get("/api/test")
def test_endpoint():
    return {"test": "success", "message": "API is working"}