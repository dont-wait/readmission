import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field


MODEL_REGISTRY = {
    "improved_xgboost": Path("models/improved_best_model.joblib"),
    "logistic": Path("models/logistic_regression_best_model.joblib"),
    "base_xgboost": Path("models/xgboost_basic.joblib"),
}
MODEL_METRICS_REGISTRY = {
    "improved_xgboost": Path("reports/improved/test_metrics.json"),
    "logistic": Path("reports/logistic/test_metrics.json"),
    "base_xgboost": Path("reports/base/test_metrics.json"),
}
DEFAULT_MODEL_ID = os.getenv("READMISSION_MODEL_ID", "improved_xgboost")
CUSTOM_MODEL_PATH_ENV = "READMISSION_MODEL_PATH"


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(..., examples=[1.4994980475525317])
    bmi: float = Field(..., examples=[-0.16957741141794483])
    bnp: float = Field(..., examples=[0.2813581310178311])
    sodium: float = Field(..., examples=[0.3407817314775273])
    creatinine: float = Field(..., examples=[1.3621636626344145])
    systolic_bp: float = Field(..., examples=[-0.0403355143463284])
    heart_rate: float = Field(..., examples=[0.43393264135275605])
    ace_inhibitor: float = Field(..., examples=[0.9247012835332911])
    beta_blocker: float = Field(..., examples=[0.9712465382469481])
    diuretic: float = Field(..., examples=[-0.9783591080694793])
    adherence_score: float = Field(..., examples=[0.9212000066229001])
    distance_to_hospital_km: float = Field(..., examples=[0.9304731191402703])


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[PatientFeatures] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    readmission_probability: float
    predicted_label: int
    threshold: float
    model_id: str
    model_path: str
    model_metrics: dict[str, Any] | None = None
    model_metrics_path: str | None = None


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class ModelInfo(BaseModel):
    model_id: str
    model_path: str
    exists: bool
    selected_threshold: float | None = None
    feature_columns: list[str] | None = None


def resolve_model_path(model_id: str | None = None) -> tuple[str, Path]:
    custom_model_path = os.getenv(CUSTOM_MODEL_PATH_ENV)
    if custom_model_path and model_id is None:
        return ("custom", Path(custom_model_path))

    selected_model_id = model_id or DEFAULT_MODEL_ID
    if selected_model_id not in MODEL_REGISTRY:
        valid_models = sorted(MODEL_REGISTRY)
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model '{selected_model_id}'. Valid models: {valid_models}",
        )

    return selected_model_id, MODEL_REGISTRY[selected_model_id]


@lru_cache(maxsize=8)
def load_model_bundle_from_path(model_path_string: str) -> dict[str, Any]:
    model_path = Path(model_path_string)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Train the model before starting the API."
        )

    bundle = joblib.load(model_path)
    required_keys = {"model", "feature_columns"}
    missing_keys = required_keys - set(bundle.keys())
    if missing_keys:
        raise ValueError(f"Invalid model bundle. Missing keys: {sorted(missing_keys)}")

    return bundle


def load_model_bundle(model_id: str | None = None) -> tuple[str, Path, dict[str, Any]]:
    resolved_model_id, model_path = resolve_model_path(model_id)
    try:
        bundle = load_model_bundle_from_path(str(model_path))
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return resolved_model_id, model_path, bundle


@lru_cache(maxsize=8)
def load_model_metrics_from_path(metrics_path_string: str) -> dict[str, Any]:
    metrics_path = Path(metrics_path_string)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with metrics_path.open(encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def load_model_metrics(model_id: str) -> tuple[dict[str, Any] | None, str | None]:
    metrics_path = MODEL_METRICS_REGISTRY.get(model_id)
    if metrics_path is None:
        return None, None

    try:
        return load_model_metrics_from_path(str(metrics_path)), str(metrics_path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None, str(metrics_path)


def get_threshold(bundle: dict[str, Any], threshold: float | None) -> float:
    if threshold is not None:
        return threshold
    return float(bundle.get("selected_threshold", 0.5))


def build_feature_frame(records: list[PatientFeatures], feature_columns: list[str]) -> pd.DataFrame:
    data = pd.DataFrame([record.model_dump() for record in records])
    missing_columns = [column for column in feature_columns if column not in data.columns]
    if missing_columns:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required feature columns for selected model: {missing_columns}",
        )

    return data[feature_columns]


def predict_records(
    records: list[PatientFeatures],
    threshold: float | None,
    model_id: str | None,
) -> list[PredictionResponse]:
    selected_model_id, model_path, bundle = load_model_bundle(model_id)
    model_metrics, model_metrics_path = load_model_metrics(selected_model_id)

    model = bundle["model"]
    feature_columns = list(bundle["feature_columns"])
    selected_threshold = get_threshold(bundle, threshold)
    features = build_feature_frame(records, feature_columns)

    try:
        probabilities = model.predict_proba(features)[:, 1]
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}") from error

    return [
        PredictionResponse(
            readmission_probability=float(probability),
            predicted_label=int(probability >= selected_threshold),
            threshold=selected_threshold,
            model_id=selected_model_id,
            model_path=str(model_path),
            model_metrics=model_metrics,
            model_metrics_path=model_metrics_path,
        )
        for probability in probabilities
    ]


def get_model_info(model_id: str, model_path: Path) -> ModelInfo:
    if not model_path.exists():
        return ModelInfo(model_id=model_id, model_path=str(model_path), exists=False)

    try:
        bundle = load_model_bundle_from_path(str(model_path))
    except (FileNotFoundError, ValueError):
        return ModelInfo(model_id=model_id, model_path=str(model_path), exists=False)

    return ModelInfo(
        model_id=model_id,
        model_path=str(model_path),
        exists=True,
        selected_threshold=float(bundle.get("selected_threshold", 0.5)),
        feature_columns=list(bundle["feature_columns"]),
    )


app = FastAPI(
    title="Readmission Prediction API",
    description=(
        "Predict 30-day hospital readmission risk using models trained on the new "
        "12-feature preprocessed dataset."
    ),
    version="2.0.0",
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
def health(model: str | None = Query(default=None)) -> dict[str, Any]:
    selected_model_id, model_path, bundle = load_model_bundle(model)
    return {
        "status": "ok",
        "active_model_id": selected_model_id,
        "active_model_path": str(model_path),
        "threshold": float(bundle.get("selected_threshold", 0.5)),
        "feature_columns": list(bundle["feature_columns"]),
        "available_models": sorted(MODEL_REGISTRY),
        "expects_preprocessed_features": True,
    }


@app.get("/models", response_model=list[ModelInfo])
def models() -> list[ModelInfo]:
    return [get_model_info(model_id, path) for model_id, path in MODEL_REGISTRY.items()]


@app.get("/features")
def features(model: str | None = Query(default=None)) -> dict[str, Any]:
    selected_model_id, _, bundle = load_model_bundle(model)
    return {
        "model_id": selected_model_id,
        "feature_columns": list(bundle["feature_columns"]),
        "expects_preprocessed_features": True,
        "example": PatientFeatures.model_json_schema()["properties"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    patient: PatientFeatures,
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    model: str | None = Query(default=None, description="One of: improved_xgboost, logistic, base_xgboost"),
) -> PredictionResponse:
    return predict_records([patient], threshold, model)[0]


@app.post("/predict/xgboost-improved", response_model=PredictionResponse)
def predict_xgboost_improved(
    patient: PatientFeatures,
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
) -> PredictionResponse:
    return predict_records([patient], threshold, "improved_xgboost")[0]


@app.post("/predict/logistic", response_model=PredictionResponse)
def predict_logistic(
    patient: PatientFeatures,
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
) -> PredictionResponse:
    return predict_records([patient], threshold, "logistic")[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    request: BatchPredictionRequest,
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    model: str | None = Query(default=None, description="One of: improved_xgboost, logistic, base_xgboost"),
) -> BatchPredictionResponse:
    return BatchPredictionResponse(predictions=predict_records(request.items, threshold, model))


@app.post("/predict/xgboost-improved/batch", response_model=BatchPredictionResponse)
def predict_xgboost_improved_batch(
    request: BatchPredictionRequest,
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
) -> BatchPredictionResponse:
    return BatchPredictionResponse(predictions=predict_records(request.items, threshold, "improved_xgboost"))


@app.post("/predict/logistic/batch", response_model=BatchPredictionResponse)
def predict_logistic_batch(
    request: BatchPredictionRequest,
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
) -> BatchPredictionResponse:
    return BatchPredictionResponse(predictions=predict_records(request.items, threshold, "logistic"))
