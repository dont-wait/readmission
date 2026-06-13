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
MODEL_QUERY_DESCRIPTION = "One of: improved_xgboost, logistic, base_xgboost"


class PatientFeatures(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "age": 70,
                "bmi": 28.1,
                "bnp": 456,
                "sodium": 137.5,
                "creatinine": 1.2,
                "systolic_bp": 130,
                "heart_rate": 82,
                "ace_inhibitor": 1,
                "beta_blocker": 1,
                "diuretic": 0,
                "adherence_score": 0.62,
                "distance_to_hospital_km": 24.5,
            }
        },
    )

    age: float = Field(..., description="Patient age in years.", examples=[70])
    bmi: float = Field(..., description="Body mass index in kg/m2.", examples=[28.1])
    bnp: float = Field(..., description="BNP level in pg/mL.", examples=[456])
    sodium: float = Field(..., description="Serum sodium in mEq/L.", examples=[137.5])
    creatinine: float = Field(..., description="Serum creatinine in mg/dL.", examples=[1.2])
    systolic_bp: float = Field(
        ...,
        description="Systolic blood pressure in mmHg.",
        examples=[130],
    )
    heart_rate: float = Field(..., description="Heart rate in beats per minute.", examples=[82])
    ace_inhibitor: float = Field(
        ...,
        description="Whether the patient uses an ACE inhibitor, 0 or 1.",
        examples=[1],
    )
    beta_blocker: float = Field(
        ...,
        description="Whether the patient uses a beta blocker, 0 or 1.",
        examples=[1],
    )
    diuretic: float = Field(..., description="Whether the patient uses a diuretic, 0 or 1.", examples=[0])
    adherence_score: float = Field(
        ...,
        description="Medication adherence score from 0 to 1.",
        examples=[0.62],
    )
    distance_to_hospital_km: float = Field(
        ...,
        description="Distance to hospital in kilometers.",
        examples=[24.5],
    )


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[PatientFeatures] = Field(
        ...,
        min_length=1,
        description="One or more patient feature rows to score.",
    )


class ModelMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    threshold: float | None = Field(default=None, description="Threshold used when the saved metrics were computed.")
    accuracy: float | None = Field(default=None, description="Accuracy on the saved evaluation split.")
    precision: float | None = Field(default=None, description="Positive-class precision on the saved evaluation split.")
    recall: float | None = Field(default=None, description="Positive-class recall on the saved evaluation split.")
    f1: float | None = Field(default=None, description="Positive-class F1 score on the saved evaluation split.")
    roc_auc: float | None = Field(default=None, description="ROC AUC on the saved evaluation split.")
    average_precision: float | None = Field(
        default=None,
        description="Average precision score on the saved evaluation split.",
    )
    tn: int | None = Field(default=None, description="True negatives.")
    fp: int | None = Field(default=None, description="False positives.")
    fn: int | None = Field(default=None, description="False negatives.")
    tp: int | None = Field(default=None, description="True positives.")
    predicted_positive: int | None = Field(default=None, description="Number of positive predictions.")
    confusion_matrix: list[list[int]] | None = Field(default=None, description="Confusion matrix if available.")
    classification_report: dict[str, Any] | None = Field(default=None, description="Classification report if available.")


class PredictionResponse(BaseModel):
    readmission_probability: float = Field(
        ...,
        description="Predicted probability for 30-day readmission, from 0 to 1.",
    )
    predicted_label: int = Field(..., description="Binary prediction after thresholding. 1 means high risk, 0 means low risk.")
    threshold: float = Field(..., description="Threshold used to convert probability into predicted_label.")
    model_id: str = Field(..., description="Model used for this prediction.")
    model_path: str = Field(..., description="Path to the loaded model bundle.")
    model_metrics: ModelMetrics | None = Field(
        default=None,
        description="Saved model-level evaluation metrics. These are not computed from a single patient request.",
    )
    model_metrics_path: str | None = Field(default=None, description="Path to the saved metrics JSON file.")


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse] = Field(..., description="Prediction result for each submitted item.")


class ModelInfo(BaseModel):
    model_id: str = Field(..., description="Registered model id.")
    model_path: str = Field(..., description="Path to the model bundle.")
    exists: bool = Field(..., description="Whether the model bundle exists and can be loaded.")
    selected_threshold: float | None = Field(default=None, description="Saved threshold from training, if available.")
    feature_columns: list[str] | None = Field(default=None, description="Feature columns expected by this model.")


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    active_model_id: str
    active_model_path: str
    threshold: float
    feature_columns: list[str]
    available_models: list[str]
    expects_preprocessed_features: bool


class FeaturesResponse(BaseModel):
    model_id: str
    feature_columns: list[str]
    expects_preprocessed_features: bool
    example: dict[str, Any]


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


def get_preprocessing(bundle: dict[str, Any]) -> dict[str, Any] | None:
    preprocessing = bundle.get("preprocessing")
    if isinstance(preprocessing, dict) and preprocessing.get("input_type") == "raw":
        return preprocessing
    return None


def expects_preprocessed_features(bundle: dict[str, Any]) -> bool:
    return get_preprocessing(bundle) is None


def get_input_feature_columns(bundle: dict[str, Any]) -> list[str]:
    preprocessing = get_preprocessing(bundle)
    if preprocessing is not None:
        return list(preprocessing.get("raw_feature_columns", bundle["feature_columns"]))
    return list(bundle["feature_columns"])


def build_feature_frame(records: list[PatientFeatures], bundle: dict[str, Any]) -> pd.DataFrame:
    feature_columns = list(bundle["feature_columns"])
    input_columns = get_input_feature_columns(bundle)
    data = pd.DataFrame([record.model_dump() for record in records])
    missing_columns = [column for column in input_columns if column not in data.columns]
    if missing_columns:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required feature columns for selected model: {missing_columns}",
        )

    preprocessing = get_preprocessing(bundle)
    if preprocessing is None:
        return data[feature_columns]

    imputer = preprocessing.get("imputer")
    scaler = preprocessing.get("scaler")
    if imputer is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model bundle declares raw preprocessing but is missing imputer or scaler.",
        )

    raw_features = data[input_columns].copy()
    if "creatinine" in raw_features.columns:
        raw_features.loc[raw_features["creatinine"] < 0, "creatinine"] = pd.NA

    numeric_columns = list(preprocessing.get("numeric_columns", input_columns))
    raw_features[numeric_columns] = imputer.transform(raw_features[numeric_columns])
    transformed = scaler.transform(raw_features[input_columns])
    return pd.DataFrame(transformed, columns=feature_columns)


def predict_records(
    records: list[PatientFeatures],
    threshold: float | None,
    model_id: str | None,
) -> list[PredictionResponse]:
    selected_model_id, model_path, bundle = load_model_bundle(model_id)
    model_metrics, model_metrics_path = load_model_metrics(selected_model_id)

    model = bundle["model"]
    selected_threshold = get_threshold(bundle, threshold)
    features = build_feature_frame(records, bundle)

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
        feature_columns=get_input_feature_columns(bundle),
    )


app = FastAPI(
    title="Readmission Prediction API",
    description=(
        "Predict 30-day hospital readmission risk using models trained on the new "
        "12-feature dataset. Models that include a saved preprocessing bundle accept "
        "raw clinical values and scale them before prediction. Prediction responses include patient-level "
        "risk output and saved model-level evaluation metrics."
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


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Metadata"],
    summary="Check API and active model status",
)
def health(model: str | None = Query(default=None, description=MODEL_QUERY_DESCRIPTION)) -> HealthResponse:
    selected_model_id, model_path, bundle = load_model_bundle(model)
    return HealthResponse(
        status="ok",
        active_model_id=selected_model_id,
        active_model_path=str(model_path),
        threshold=float(bundle.get("selected_threshold", 0.5)),
        feature_columns=get_input_feature_columns(bundle),
        available_models=sorted(MODEL_REGISTRY),
        expects_preprocessed_features=expects_preprocessed_features(bundle),
    )


@app.get(
    "/models",
    response_model=list[ModelInfo],
    tags=["Metadata"],
    summary="List registered models",
)
def models() -> list[ModelInfo]:
    return [get_model_info(model_id, path) for model_id, path in MODEL_REGISTRY.items()]


@app.get(
    "/features",
    response_model=FeaturesResponse,
    tags=["Metadata"],
    summary="Get expected feature columns",
)
def features(model: str | None = Query(default=None, description=MODEL_QUERY_DESCRIPTION)) -> FeaturesResponse:
    selected_model_id, _, bundle = load_model_bundle(model)
    return FeaturesResponse(
        model_id=selected_model_id,
        feature_columns=get_input_feature_columns(bundle),
        expects_preprocessed_features=expects_preprocessed_features(bundle),
        example=PatientFeatures.model_json_schema()["properties"],
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict readmission risk with a selectable model",
)
def predict(
    patient: PatientFeatures,
    threshold: float | None = Query(default=None, ge=0.0, le=1.0, description="Optional classification threshold."),
    model: str | None = Query(default=None, description=MODEL_QUERY_DESCRIPTION),
) -> PredictionResponse:
    return predict_records([patient], threshold, model)[0]


@app.post(
    "/predict/xgboost-improved",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict readmission risk with the improved XGBoost model",
)
def predict_xgboost_improved(
    patient: PatientFeatures,
    threshold: float | None = Query(default=None, ge=0.0, le=1.0, description="Optional classification threshold."),
) -> PredictionResponse:
    return predict_records([patient], threshold, "improved_xgboost")[0]


@app.post(
    "/predict/logistic",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict readmission risk with the logistic regression model",
)
def predict_logistic(
    patient: PatientFeatures,
    threshold: float | None = Query(default=None, ge=0.0, le=1.0, description="Optional classification threshold."),
) -> PredictionResponse:
    return predict_records([patient], threshold, "logistic")[0]


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Predict readmission risk for multiple patients with a selectable model",
)
def predict_batch(
    request: BatchPredictionRequest,
    threshold: float | None = Query(default=None, ge=0.0, le=1.0, description="Optional classification threshold."),
    model: str | None = Query(default=None, description=MODEL_QUERY_DESCRIPTION),
) -> BatchPredictionResponse:
    return BatchPredictionResponse(predictions=predict_records(request.items, threshold, model))


@app.post(
    "/predict/xgboost-improved/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Predict readmission risk for multiple patients with the improved XGBoost model",
)
def predict_xgboost_improved_batch(
    request: BatchPredictionRequest,
    threshold: float | None = Query(default=None, ge=0.0, le=1.0, description="Optional classification threshold."),
) -> BatchPredictionResponse:
    return BatchPredictionResponse(predictions=predict_records(request.items, threshold, "improved_xgboost"))


@app.post(
    "/predict/logistic/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Predict readmission risk for multiple patients with the logistic regression model",
)
def predict_logistic_batch(
    request: BatchPredictionRequest,
    threshold: float | None = Query(default=None, ge=0.0, le=1.0, description="Optional classification threshold."),
) -> BatchPredictionResponse:
    return BatchPredictionResponse(predictions=predict_records(request.items, threshold, "logistic"))
