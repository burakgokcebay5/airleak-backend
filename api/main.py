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

@app.get("/api/test")
async def test():
    return {"test": "success", "platform": "vercel"}

# Handler for Vercel
handler = app