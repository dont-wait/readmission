import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


DEFAULT_MODEL_PATH = "models/improved_best_model.joblib"


class PatientFeatures(BaseModel):
    age: float = Field(..., examples=[70])
    bmi: float = Field(..., examples=[28.1])
    bnp: float = Field(..., examples=[456])
    sodium: float = Field(..., examples=[137.5])
    creatinine: float = Field(..., examples=[1.2])
    systolic_bp: float = Field(..., examples=[130])
    heart_rate: float = Field(..., examples=[82])
    adherence_score: float = Field(..., examples=[0.62])
    distance_to_hospital_km: float = Field(..., examples=[24.5])


class BatchPredictionRequest(BaseModel):
    items: list[PatientFeatures]


class PredictionResponse(BaseModel):
    readmission_probability: float
    predicted_label: int
    threshold: float
    model_path: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


def get_model_path() -> Path:
    return Path(os.getenv("READMISSION_MODEL_PATH", DEFAULT_MODEL_PATH))


@lru_cache(maxsize=1)
def load_model_bundle() -> dict[str, Any]:
    model_path = get_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Train it with: python -m src.improve_models --config configs/xgboost_basic.yaml"
        )

    bundle = joblib.load(model_path)
    required_keys = {"model", "feature_columns"}
    missing_keys = required_keys - set(bundle.keys())
    if missing_keys:
        raise ValueError(f"Invalid model bundle. Missing keys: {sorted(missing_keys)}")

    return bundle


def get_threshold(bundle: dict[str, Any], threshold: float | None) -> float:
    if threshold is not None:
        return threshold
    return float(bundle.get("selected_threshold", 0.5))


def build_feature_frame(records: list[PatientFeatures], feature_columns: list[str]) -> pd.DataFrame:
    data = pd.DataFrame([record.model_dump() for record in records])
    missing_columns = [column for column in feature_columns if column not in data.columns]
    if missing_columns:
        raise HTTPException(status_code=422, detail=f"Missing required feature columns: {missing_columns}")

    return data[feature_columns]


def predict_records(
    records: list[PatientFeatures],
    threshold: float | None,
) -> list[PredictionResponse]:
    try:
        bundle = load_model_bundle()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    model = bundle["model"]
    feature_columns = list(bundle["feature_columns"])
    selected_threshold = get_threshold(bundle, threshold)
    features = build_feature_frame(records, feature_columns)
    probabilities = model.predict_proba(features)[:, 1]
    model_path = str(get_model_path())

    return [
        PredictionResponse(
            readmission_probability=float(probability),
            predicted_label=int(probability >= selected_threshold),
            threshold=selected_threshold,
            model_path=model_path,
        )
        for probability in probabilities
    ]


app = FastAPI(
    title="Readmission Prediction API",
    description="Predict 30-day hospital readmission risk with the trained XGBoost model.",
    version="1.0.0",
)

cors_origins = [origin.strip() for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        bundle = load_model_bundle()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return {
        "status": "ok",
        "model_path": str(get_model_path()),
        "threshold": float(bundle.get("selected_threshold", 0.5)),
        "feature_columns": list(bundle["feature_columns"]),
    }


@app.get("/features")
def features() -> dict[str, list[str]]:
    try:
        bundle = load_model_bundle()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return {"feature_columns": list(bundle["feature_columns"])}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    patient: PatientFeatures,
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
) -> PredictionResponse:
    return predict_records([patient], threshold)[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    request: BatchPredictionRequest,
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
) -> BatchPredictionResponse:
    if not request.items:
        raise HTTPException(status_code=422, detail="items must contain at least one record.")

    return BatchPredictionResponse(predictions=predict_records(request.items, threshold))
