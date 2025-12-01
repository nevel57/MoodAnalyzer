import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app.services.hybrid_sentiment_rusentiment import HybridSentimentServiceRuSentiment

app = FastAPI(
    title="RuSentiment Hybrid Sentiment Analysis API",
    description="API trained on RuSentiment dataset with hybrid approach",
    version="3.0.0"
)

service = HybridSentimentServiceRuSentiment()


class TextRequest(BaseModel):
    text: str
    force_model: Optional[str] = None


class BatchRequest(BaseModel):
    texts: List[str]
    force_model: Optional[str] = None


class SentimentResponse(BaseModel):
    success: bool
    text: str
    sentiment: str
    confidence: float
    probabilities: dict
    model_used: str
    routing_decision: Optional[dict] = None
    performance: Optional[dict] = None
    text_info: Optional[dict] = None


class BatchResponse(BaseModel):
    results: List[SentimentResponse]
    statistics: dict


@app.get("/")
async def root():
    return {
        "message": "RuSentiment Hybrid Sentiment Analysis API",
        "version": "3.0.0",
        "description": "Trained on RuSentiment dataset. Uses hybrid approach.",
        "endpoints": {
            "analyze": "POST /analyze",
            "batch_analyze": "POST /batch_analyze",
            "health": "GET /health",
            "info": "GET /info",
            "train": "POST /train"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "rusentiment_hybrid_analysis",
        "dataset": "RuSentiment",
        "components": {
            "fast_model": "RuSentimentPredictor",
            "accurate_model": "stub",
            "router": "active"
        }
    }


@app.get("/info")
async def service_info():
    return {
        "service": "RuSentiment Hybrid Analysis",
        "dataset": "RuSentiment (Russian Sentiment Analysis Dataset)",
        "approach": "TF-IDF + Logistic Regression trained on real data",
        "classes": ["negative", "positive", "other"],
        "routing": "Automatic model selection based on text complexity"
    }


@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: TextRequest):
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if request.force_model:
        if request.force_model not in ['fast', 'accurate']:
            raise HTTPException(
                status_code=400,
                detail="force_model must be 'fast' or 'accurate'"
            )

    result = service.analyze(request.text)

    if not result['success']:
        raise HTTPException(status_code=422, detail=result['error'])

    return result


@app.post("/batch_analyze", response_model=BatchResponse)
async def batch_analyze_sentiment(request: BatchRequest):
    if not request.texts or len(request.texts) == 0:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")

    if len(request.texts) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 texts per batch"
        )

    for i, text in enumerate(request.texts):
        if not text or len(text.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Text at position {i} is empty"
            )

    results = service.batch_analyze(request.texts)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    if successful:
        fast_count = sum(1 for r in successful if r['routing_decision']['use_fast'])
        accurate_count = len(successful) - fast_count

        avg_confidence = sum(r['confidence'] for r in successful) / len(successful)
        avg_time = sum(r['performance']['total_time'] for r in successful) / len(successful)
    else:
        fast_count = accurate_count = avg_confidence = avg_time = 0

    statistics = {
        "total_texts": len(request.texts),
        "successful": len(successful),
        "failed": len(failed),
        "routing": {
            "fast_model": fast_count,
            "accurate_model": accurate_count
        },
        "average_confidence": round(avg_confidence, 3) if successful else 0,
        "average_processing_time": round(avg_time, 4) if successful else 0
    }

    return {
        "results": results,
        "statistics": statistics
    }


@app.post("/train")
async def train_model():
    try:
        from app.ml.rusentiment_predictor import RuSentimentPredictor

        predictor = RuSentimentPredictor(
            model_path="models/rusentiment_fresh.joblib",
            data_path="data/rusentiment.csv"
        )

        predictor._train_model()

        return {
            "success": True,
            "message": "Model trained successfully on RuSentiment data",
            "model_path": "models/rusentiment_fresh.joblib"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Starting RuSentiment Hybrid Sentiment Analysis API")
    print("Version: 3.0.0 (Trained on RuSentiment dataset)")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 60)

    uvicorn.run(
        "app.main_rusentiment:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
