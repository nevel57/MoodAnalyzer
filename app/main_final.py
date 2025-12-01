from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(
    title="Russian Sentiment API",
    version="3.0.0"
)

# Простые модели для теста
class TextRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {
        "message": "API работает",
        "endpoints": ["/analyze", "/train", "/health"]
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/train")
async def train():
    return {
        "success": True,
        "message": "Training endpoint работает",
        "model_path": "models/test.joblib"
    }

@app.post("/analyze")
async def analyze(request: TextRequest):
    return {
        "text": request.text,
        "sentiment": "positive",
        "confidence": 0.9
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 API запущен с /train endpoint")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
