
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "best_model.joblib"))
)
METADATA_PATH = Path(
    os.getenv("METADATA_PATH", str(BASE_DIR / "models" / "model_metadata.json"))
)

app = FastAPI(
    title="AirSense API",
    description=(
        "Air quality prediction API based on pollution measurements, "
        "weather data, and historical observations."
    ),
    version="1.0.0",
)


class PredictionInput(BaseModel):
    """Input schema containing the 100 features required by the model."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, float] = Field(
        ...,
        description=(
            "Dictionary containing exactly the model's required features "
            "and their numerical values."
        ),
    )


def load_artifacts() -> tuple[Any, dict]:
    """Load the trained model and its metadata."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Model metadata file not found: {METADATA_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    with METADATA_PATH.open(encoding="utf-8") as file:
        metadata = json.load(file)

    return model, metadata


try:
    model, metadata = load_artifacts()
    FEATURE_COLUMNS = metadata["feature_columns"]
    TARGET_COLUMNS = metadata["target_columns"]
except Exception as exc:
    model = None
    metadata = {}
    FEATURE_COLUMNS = []
    TARGET_COLUMNS = []
    LOAD_ERROR = str(exc)
else:
    LOAD_ERROR = None


@app.get("/health", tags=["Health"])
def health() -> dict:
    """Check whether the API and the model are available."""
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "detail": LOAD_ERROR,
    }


@app.get("/model-info", tags=["Model"])
def model_info() -> dict:
    """Return information about the trained model and its evaluation."""
    if model is None:
        raise HTTPException(status_code=503, detail=LOAD_ERROR)

    return {
        "model_name": metadata.get("model_name", type(model).__name__),
        "feature_count": len(FEATURE_COLUMNS),
        "target_columns": TARGET_COLUMNS,
        "evaluation": metadata.get("evaluation", []),
    }


@app.post("/predict", tags=["Prediction"])
def predict(payload: PredictionInput) -> dict:
    """Predict the six pollutant concentrations for the next time step."""
    if model is None:
        raise HTTPException(status_code=503, detail=LOAD_ERROR)

    supplied = payload.features
    expected = set(FEATURE_COLUMNS)
    received = set(supplied)

    missing = sorted(expected - received)
    unexpected = sorted(received - expected)

    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "The provided features do not match the model's "
                    "expected input features."
                ),
                "missing_features": missing,
                "unexpected_features": unexpected,
            },
        )

    values = np.array(
        [supplied[column] for column in FEATURE_COLUMNS],
        dtype=float,
    )

    if not np.isfinite(values).all():
        raise HTTPException(
            status_code=422,
            detail="All feature values must be finite numbers.",
        )

    input_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    prediction = np.asarray(model.predict(input_df))

    if prediction.ndim != 2 or prediction.shape[1] != len(TARGET_COLUMNS):
        raise HTTPException(
            status_code=500,
            detail=(
                "The prediction output shape does not match "
                "the expected target columns."
            ),
        )

    return {
        "predictions": {
            target: float(value)
            for target, value in zip(TARGET_COLUMNS, prediction[0])
        },
        "horizon": "t+1",
        "model_name": metadata.get("model_name", type(model).__name__),
    }
