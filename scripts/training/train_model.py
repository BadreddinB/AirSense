"""Training data for the AirSense ML pipeline."""

from __future__ import annotations

import os
import logging

import numpy as np
import pandas as pd
import json
import joblib

from datetime import datetime, timezone
from pathlib import Path

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

import mlflow
import mlflow.sklearn

POLLUTANTS = (

    "pm2_5",
    "pm10",
    "ozone",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "carbon_monoxide",
)

WEATHER_FEATURES = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "dew_point_2m",
    "surface_pressure",
    "cloud_cover",
    "shortwave_radiation",
    "boundary_layer_height",
)

TIME_FEATURES = (
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
)

TARGET_COLUMNS = tuple(
    f"target_{pollutant}_t1" for pollutant in POLLUTANTS
)

BASE_FEATURES = (*POLLUTANTS, *WEATHER_FEATURES, *TIME_FEATURES)

FEATURE_COLUMNS = (
    *BASE_FEATURES,
    *(
        f"{column}_lag_{lag}"
        for column in (*POLLUTANTS, *WEATHER_FEATURES)
        for lag in range(1, 6)
    ),
)

def build_training_dataset(
    gold_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the Gold schema and return features X and targets y."""

    required_columns = {

        "city",
        "timestamp",
        *FEATURE_COLUMNS,
        *TARGET_COLUMNS,
    }

    missing_columns = sorted(required_columns - set(gold_df.columns))

    if missing_columns:
        missing_targets = sorted(set(TARGET_COLUMNS) & set(missing_columns))

        if missing_targets:
            raise ValueError(f"Missing target columns: {missing_targets}")
        raise ValueError(f"Missing required columns: {missing_columns}")

    if gold_df.empty:
        raise ValueError("Gold dataset must not be empty")

    X = gold_df.loc[:, FEATURE_COLUMNS].copy()
    y = gold_df.loc[:, TARGET_COLUMNS].copy()

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("All training features must be numeric")

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in y.dtypes):
        raise ValueError("All target columns must be numeric")

    return X, y


def build_time_splits(
    gold_df: pd.DataFrame,
    n_splits: int = 3,

) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create chronological splits while keeping cities together per timestamp."""

    if "timestamp" not in gold_df.columns:
        raise ValueError("Gold dataset must contain a timestamp column")

    if gold_df["timestamp"].isna().any():
        raise ValueError("Timestamps must not contain missing values")

    unique_timestamps = pd.Index(
        gold_df["timestamp"].drop_duplicates()
    ).sort_values()

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    if len(unique_timestamps) <= n_splits:
        raise ValueError("Number of unique timestamps must be greater than n_splits")

    splitter = TimeSeriesSplit(n_splits=n_splits)
    splits = []

    timestamp_values = gold_df["timestamp"]

    for train_time_idx, validation_time_idx in splitter.split(unique_timestamps):
        train_timestamps = unique_timestamps[train_time_idx]
        validation_timestamps = unique_timestamps[validation_time_idx]

        train_idx = np.flatnonzero(timestamp_values.isin(train_timestamps))
        validation_idx = np.flatnonzero(timestamp_values.isin(validation_timestamps))

        splits.append((train_idx, validation_idx))

    return splits


def predict_persistence(gold_df: pd.DataFrame) -> pd.DataFrame:
    """Predict each next-hour pollutant value using its current value."""

    missing_pollutants = [
        pollutant
        for pollutant in POLLUTANTS
        if pollutant not in gold_df.columns
    ]

    if missing_pollutants:
        raise ValueError(
            f"Missing pollutant columns for persistence baseline: "
            f"{missing_pollutants}"
        )

    if gold_df.empty:
        raise ValueError("Gold dataset must not be empty")

    predictions = gold_df.loc[:, POLLUTANTS].copy()
    predictions.columns = TARGET_COLUMNS

    return predictions


def evaluate_predictions(

    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,

) -> pd.DataFrame:
    """Calculate regression metrics for each pollutant target."""

    missing_targets = [
        target for target in TARGET_COLUMNS
        if target not in y_true.columns or target not in y_pred.columns
    ]

    if missing_targets:
        raise ValueError(f"Missing target columns for evaluation: {missing_targets}")

    if y_true.empty or y_pred.empty:
        raise ValueError("Evaluation data must not be empty")

    if len(y_true) != len(y_pred):
        raise ValueError("True and predicted values must have the same number of rows")

    results = []

    for target in TARGET_COLUMNS:
        actual = y_true[target]
        predicted = y_pred[target]

        results.append(
            {
                "target": target,
                "mae": mean_absolute_error(actual, predicted),
                "rmse": np.sqrt(mean_squared_error(actual, predicted)),
                "r2": r2_score(actual, predicted),
            }
        )

    return pd.DataFrame(results)


def split_train_test_by_time(

    gold_df: pd.DataFrame,
    test_fraction: float = 0.15,

) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split Gold data chronologically, keeping each timestamp together."""

    if "timestamp" not in gold_df.columns:
        raise ValueError("Gold dataset must contain a timestamp column")

    if gold_df.empty:
        raise ValueError("Gold dataset must not be empty")

    if gold_df["timestamp"].isna().any():
        raise ValueError("Timestamps must not contain missing values")

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    unique_timestamps = pd.Index(
        gold_df["timestamp"].drop_duplicates()).sort_values()

    if len(unique_timestamps) < 2:
        raise ValueError("At least two unique timestamps are required")

    split_position = int(len(unique_timestamps) * (1 - test_fraction))
    split_position = min(max(split_position, 1), len(unique_timestamps) - 1)
    cutoff_timestamp = unique_timestamps[split_position]

    development_df = gold_df.loc[
        gold_df["timestamp"] < cutoff_timestamp].copy()

    test_df = gold_df.loc[
        gold_df["timestamp"] >= cutoff_timestamp].copy()

    return development_df, test_df

def build_models() -> dict[str, object]:
    """Create the regression models to compare."""

    return {

        "linear_regression": LinearRegression(),

        "random_forest": RandomForestRegressor(
            n_estimators=40,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=2,
        ),

        "xgboost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=40,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=2,
        ),
    }


def cross_validate_models(
    development_df: pd.DataFrame,
    n_splits: int = 3,

) -> pd.DataFrame:
    """Evaluate regression models using chronological cross-validation."""

    X, y = build_training_dataset(development_df)

    splits = build_time_splits(development_df, n_splits=n_splits)
    models = build_models()

    results = []

    for model_name, model in models.items():

        for fold, (train_idx, validation_idx) in enumerate(splits, start=1):

            estimator = clone(model)

            X_train = X.iloc[train_idx]

            X_validation = X.iloc[validation_idx]

            y_train = y.iloc[train_idx]

            y_validation = y.iloc[validation_idx]

            estimator.fit(X_train, y_train)

            predictions = estimator.predict(X_validation)

            fold_metrics = evaluate_predictions(
                y_validation,
                pd.DataFrame(predictions, columns=TARGET_COLUMNS, index=y_validation.index),
            )

            fold_metrics["model"] = model_name
            fold_metrics["fold"] = fold
            results.append(fold_metrics)

    return pd.concat(results, ignore_index=True)


def cross_validate_persistence(
    development_df: pd.DataFrame,
    n_splits: int = 3,

) -> pd.DataFrame:
    """Evaluate the persistence baseline on chronological validation folds."""

    splits = build_time_splits(development_df, n_splits=n_splits)

    results = []

    for fold, (_, validation_idx) in enumerate(splits, start=1):

        validation_df = development_df.iloc[validation_idx]

        y_validation = validation_df.loc[:, TARGET_COLUMNS]

        predictions = predict_persistence(validation_df)

        fold_metrics = evaluate_predictions(y_validation, predictions)
        fold_metrics["model"] = "persistence"
        fold_metrics["fold"] = fold
        results.append(fold_metrics)

    return pd.concat(results, ignore_index=True)


def compare_models(
    development_df: pd.DataFrame,
    n_splits: int = 3,

) -> pd.DataFrame:
    """Compare the persistence baseline and regression models."""

    persistence_results = cross_validate_persistence(
        development_df,
        n_splits=n_splits,
    )

    model_results = cross_validate_models(
        development_df,
        n_splits=n_splits,
    )

    results = pd.concat(
        [persistence_results, model_results],
        ignore_index=True,
    )
    return results


def summarize_model_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize cross-validation metrics by model and pollutant."""

    required_columns = {"model", "fold", "target", "mae", "rmse", "r2"}
    missing_columns = sorted(required_columns - set(results.columns))

    if missing_columns:
        raise ValueError(f"Missing result columns: {missing_columns}")

    if results.empty:
        raise ValueError("Results must not be empty")

    return (

        results.groupby(["model", "target"], as_index=False)[
            ["mae", "rmse", "r2"]
        ]
        .mean()
        .sort_values(["target", "mae"])
        .reset_index(drop=True)
    )


def select_best_model(results: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Select the best trainable model using mean RMSE across targets."""

    required_columns = {"model", "target", "rmse"}
    missing_columns = sorted(required_columns - set(results.columns))

    if missing_columns:
        raise ValueError(
            f"Missing required result columns: {missing_columns}"
        )

    if results.empty:
        raise ValueError("Model comparison results must not be empty")

    # Persistence is a baseline, not a trainable estimator.
    candidates = results.loc[results["model"] != "persistence"].copy()

    if candidates.empty:
        raise ValueError("No trainable models are available for selection")

    if candidates["rmse"].isna().any():
        raise ValueError("RMSE values must not contain missing values")

    summary = (
        candidates.groupby("model", as_index=False)
        .agg(mean_rmse=("rmse", "mean"))
        .sort_values("mean_rmse")
        .reset_index(drop=True)
    )

    best_model_name = str(summary.iloc[0]["model"])

    return best_model_name, summary


def fit_selected_model(

    development_df: pd.DataFrame,
    model_name: str,

) -> object:
    """Fit the selected model on the complete development dataset."""

    X, y = build_training_dataset(development_df)

    models = build_models()

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {sorted(models)}"
        )

    model = clone(models[model_name])
    model.fit(X, y)

    return model


def evaluate_final_model(

    model: object,
    test_df: pd.DataFrame,

) -> pd.DataFrame:
    """Evaluate a fitted model on the untouched chronological test set."""

    X_test, y_test = build_training_dataset(test_df)

    predictions = model.predict(X_test)

    predictions_df = pd.DataFrame(
        predictions,
        columns=TARGET_COLUMNS,
        index=y_test.index,
    )

    return evaluate_predictions(y_test, predictions_df)


def save_model_artifacts(

    model: object,
    model_name: str,
    evaluation_metrics: pd.DataFrame,
    output_dir: str | Path,

) -> dict[str, str]:
    """Persist the fitted model, evaluation metrics, and metadata."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "best_model.joblib"
    metrics_path = output_path / "final_test_metrics.csv"
    metadata_path = output_path / "model_metadata.json"


    joblib.dump(model, model_path)
    evaluation_metrics.to_csv(metrics_path, index=False)


    metadata = {

        "model_name": model_name,
        "target_columns": list(TARGET_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "evaluation": evaluation_metrics.to_dict(orient="records"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, allow_nan=False)


    return {

        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metadata_path": str(metadata_path),
    }

def configure_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> str:
    """Configure MLflow tracking and return the active experiment name."""
    resolved_uri = tracking_uri or os.getenv(
        "MLFLOW_TRACKING_URI", "http://mlflow:5000"
    )
    resolved_experiment = experiment_name or os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "airsense-model-training"
    )

    mlflow.set_tracking_uri(resolved_uri)
    mlflow.set_experiment(resolved_experiment)
    logging.info(
        "Configured MLflow tracking URI=%s, experiment=%s",
        resolved_uri,
        resolved_experiment,
    )
    return resolved_experiment


def log_metrics_by_target(
    metrics: pd.DataFrame,
    prefix: str,
) -> None:
    """Log per-target regression metrics and their unweighted means."""
    metric_columns = [column for column in ("mae", "rmse", "r2") if column in metrics]
    for _, row in metrics.iterrows():
        target = str(row["target"])
        for metric_name in metric_columns:
            value = float(row[metric_name])
            if np.isfinite(value):
                mlflow.log_metric(f"{prefix}_{target}_{metric_name}", value)

    for metric_name in metric_columns:
        values = pd.to_numeric(metrics[metric_name], errors="coerce").dropna()
        if not values.empty:
            mlflow.log_metric(f"{prefix}_mean_{metric_name}", float(values.mean()))


def run_training_workflow(
    gold_df: pd.DataFrame,
    output_dir: str | Path = "models",
    n_splits: int = 3,
    test_fraction: float = 0.15,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> dict[str, object]:
    """Run chronological model selection, final evaluation, and MLflow logging.

    The final test period is held out before cross-validation and is not used
    for model selection. Each candidate's CV results are logged as a nested
    MLflow run; the selected model's final test metrics and artifacts are
    logged in the parent run.
    """
    configure_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)

    development_df, test_df = split_train_test_by_time(
        gold_df, test_fraction=test_fraction
    )

    with mlflow.start_run(run_name="airsense_training_workflow") as parent_run:
        mlflow.log_params(
            {
                "n_splits": n_splits,
                "test_fraction": test_fraction,
                "development_rows": len(development_df),
                "test_rows": len(test_df),
                "feature_count": len(FEATURE_COLUMNS),
                "target_count": len(TARGET_COLUMNS),
                "selection_metric": "mean_cv_rmse",
            }
        )

        cv_results = compare_models(development_df, n_splits=n_splits)
        summary = summarize_model_results(cv_results)

        mlflow.log_text(
            cv_results.to_csv(index=False),
            "cross_validation/cv_results.csv",
        )
        mlflow.log_text(
            summary.to_csv(index=False),
            "cross_validation/model_summary.csv",
        )

        # Track each candidate separately while preserving the full CV table.
        for model_name in sorted(cv_results["model"].unique()):
            candidate_results = cv_results.loc[
                cv_results["model"] == model_name
            ]
            with mlflow.start_run(run_name=f"cv_{model_name}", nested=True):
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("role", "baseline" if model_name == "persistence" else "candidate")
                for metric_name in ("mae", "rmse", "r2"):
                    values = pd.to_numeric(
                        candidate_results[metric_name], errors="coerce"
                    ).dropna()
                    if not values.empty:
                        mlflow.log_metric(
                            f"cv_mean_{metric_name}", float(values.mean())
                        )
                mlflow.log_text(
                    candidate_results.to_csv(index=False),
                    "cross_validation/fold_metrics.csv",
                )

        best_model_name, selection_summary = select_best_model(cv_results)
        mlflow.log_param("selected_model", best_model_name)
        mlflow.log_text(
            selection_summary.to_csv(index=False),
            "cross_validation/selection_summary.csv",
        )

        model = fit_selected_model(development_df, best_model_name)
        final_metrics = evaluate_final_model(model, test_df)

        # Compare the candidate with the current reference on the same test set.
        output_path = Path(output_dir)
        reference_model_path = output_path / "best_model.joblib"
        promoted = not reference_model_path.exists()
        reference_metrics = None

        if reference_model_path.exists():
            import joblib

            reference_model = joblib.load(reference_model_path)
            reference_metrics = evaluate_final_model(reference_model, test_df)
            promoted = should_promote_model(final_metrics, reference_metrics)

        mlflow.log_param("model_promoted", promoted)
        mlflow.log_metric("model_promoted", float(promoted))
        mlflow.log_param(
            "promotion_reason",
            "no_reference_model" if reference_metrics is None
            else "candidate_improved" if promoted
            else "candidate_not_better",
        )

        log_metrics_by_target(final_metrics, prefix="candidate_test")

        # Keep candidate results in MLflow even when promotion is rejected.
        mlflow.log_text(
            final_metrics.to_csv(index=False),
            "candidate_evaluation/final_test_metrics.csv",
        )

        artifact_paths = {}
        if promoted:
            artifact_paths = save_model_artifacts(
                model, best_model_name, final_metrics, output_dir
            )
            mlflow.log_artifact(
                artifact_paths["model_path"], artifact_path="model"
            )
            mlflow.log_artifact(
                artifact_paths["metrics_path"], artifact_path="evaluation"
            )
            mlflow.log_artifact(
                artifact_paths["metadata_path"], artifact_path="evaluation"
            )
            logging.info(
                "Candidate model %s promoted and saved.",
                best_model_name,
            )
        else:
            logging.info(
                "Candidate model %s was not promoted. "
                "The reference model remains unchanged.",
                best_model_name,
            )

        mlflow.log_dict(
            {
                "model_name": best_model_name,
                "model_promoted": promoted,
                "feature_columns": list(FEATURE_COLUMNS),
                "target_columns": list(TARGET_COLUMNS),
                "test_metrics": final_metrics.to_dict(orient="records"),
                "reference_metrics": (
                    reference_metrics.to_dict(orient="records")
                    if reference_metrics is not None
                    else None
                ),
                "parent_run_id": parent_run.info.run_id,
            },
            "model_metadata_mlflow.json",
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="selected_model",
            input_example=development_df.loc[:, FEATURE_COLUMNS].head(5),
        )

        return {
            "run_id": parent_run.info.run_id,
            "model_name": best_model_name,
            "cv_results": cv_results,
            "selection_summary": selection_summary,
            "final_metrics": final_metrics,
            "reference_metrics": reference_metrics,
            "promoted": promoted,
            "artifacts": artifact_paths,
        }


def should_promote_model(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
) -> bool:
    """Return whether a candidate model improves mean RMSE over a reference."""
    for name, metrics in (
        ("candidate", candidate),
        ("reference", reference),
    ):
        if "rmse" not in metrics.columns:
            raise ValueError(f"{name} metrics must contain an 'rmse' column")

        rmse_values = pd.to_numeric(metrics["rmse"], errors="coerce")

        if rmse_values.empty or rmse_values.isna().any():
            raise ValueError(f"{name} RMSE values must be non-empty and numeric")

        if not rmse_values.map(lambda value: value >= 0 and value < float("inf")).all():
            raise ValueError(f"{name} RMSE values must be finite and non-negative")

    candidate_mean_rmse = candidate["rmse"].mean()
    reference_mean_rmse = reference["rmse"].mean()

    return bool(candidate_mean_rmse < reference_mean_rmse)
