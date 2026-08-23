from typing import Optional, List
from pydantic import BaseModel, Field


class PredictBase64Request(BaseModel):
    """Body for POST /predict/base64"""
    image_base64: str = Field(..., description="Base64-encoded image bytes (no data: prefix needed)")


class PredictionResult(BaseModel):
    predicted_class: str
    confidence: float
    top_k: Optional[List[dict]] = None  # e.g. [{"class": "cat", "confidence": 0.87}, ...]


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_stage: str
    model_version: Optional[str] = None
