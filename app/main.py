import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.logging_config import get_logger

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "model.joblib"
FEATURE_ORDER = [
    "flu_cases_reported",
    "temp_c",
    "nearby_clinics",
    "discount_pct",
    "stock_lines",
    "distance_to_hospital_km",
    "shop_type",
    "locality",
]

logger = get_logger()


class PredictRequest(BaseModel):
    flu_cases_reported: int = Field(..., ge=0, description="flu cases reported in the ward")
    temp_c: float = Field(..., description="minimum temperature in deg C")
    nearby_clinics: int = Field(..., ge=0, description="clinics within 1 km")
    discount_pct: float = Field(..., ge=0, le=100, description="discount offered, percent")
    stock_lines: int = Field(..., ge=0, le=5000, description="different products stocked (max 5000)")
    distance_to_hospital_km: float = Field(..., ge=0, description="distance to nearest hospital, km (not negative)")
    shop_type: Literal["chain", "independent", "hospital_store"]
    locality: Literal["Old Town", "New Town", "Cantonment"]


class PredictResponse(BaseModel):
    prediction: float


@lru_cache(maxsize=1)
def get_model():
    """Load the champion model once; works with or without app startup events."""
    path = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    model = joblib.load(path)
    logger.info("model loaded", extra={"extra_fields": {"event": "model_loaded", "path": path.name}})
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()  # fail fast at start-up if the model file is broken
    yield


app = FastAPI(title="MP046 Pharmacy Daily Sales API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start = time.perf_counter()
    try:
        model = get_model()
        row = pd.DataFrame([request.model_dump()])[FEATURE_ORDER]
        value = float(model.predict(row)[0])
    except Exception:
        logger.exception("prediction failed", extra={"extra_fields": {"event": "prediction_error"}})
        raise HTTPException(status_code=500, detail="prediction failed")

    logger.info(
        "prediction served",
        extra={
            "extra_fields": {
                "event": "prediction",
                "features": request.model_dump(),
                "prediction": round(value, 2),
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            }
        },
    )
    return PredictResponse(prediction=round(value, 2))
