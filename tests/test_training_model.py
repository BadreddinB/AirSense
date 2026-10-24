
"""Tests for the AirSense model training pipeline."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from scripts.training.train_model import (
    FEATURE_COLUMNS,
    POLLUTANTS,
    TARGET_COLUMNS,
    build_time_splits,
    build_training_dataset,
    evaluate_predictions,
    predict_persistence,
    save_model_artifacts,
    select_best_model,
    split_train_test_by_time,
    summarize_model_results,
)


@pytest.fixture
def gold_sample() -> pd.DataFrame:
    """Build a synthetic Gold dataset with the expected schema."""
    rows = 12
    timestamps = pd.date_range("2023-01-01", periods=rows, freq="h")

    data = {
        "city": ["Paris"] * rows,
        "timestamp": timestamps,
    }

    for index, column in enumerate(FEATURE_COLUMNS):
        data[column] = np.arange(rows, dtype=float) + index + 1

    for index, target in enumerate(TARGET_COLUMNS):
        data[target] = np.arange(rows, dtype=float) + (index + 10)

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Training data contract
# ---------------------------------------------------------------------------

def test_training_contract_excludes_metadata_and_targets(
    gold_sample: pd.DataFrame,
) -> None:
    """Only declared features belong in X; targets belong in y."""
    X, y = build_training_dataset(gold_sample)

    assert X.shape == (len(gold_sample), 100)
    assert y.shape == (len(gold_sample), len(TARGET_COLUMNS))
    assert list(X.columns) == list(FEATURE_COLUMNS)
    assert list(y.columns) == list(TARGET_COLUMNS)

    assert "city" not in X.columns
    assert "timestamp" not in X.columns

    assert set(X.columns).isdisjoint(TARGET_COLUMNS)


def test_training_contract_rejects_missing_targets(
    gold_sample: pd.DataFrame,
) -> None:
    """A missing target must produce a clear validation error."""
    invalid_gold = gold_sample.drop(columns=[TARGET_COLUMNS[0]])

    with pytest.raises(ValueError, match="target"):
        build_training_dataset(invalid_gold)


def test_training_contract_rejects_missing_features(
    gold_sample: pd.DataFrame,
) -> None:
    """A missing required feature must be rejected."""
    invalid_gold = gold_sample.drop(columns=[FEATURE_COLUMNS[0]])

    with pytest.raises(ValueError, match="required columns"):
        build_training_dataset(invalid_gold)


def test_training_contract_rejects_empty_dataset(
    gold_sample: pd.DataFrame,
) -> None:
    """An empty Gold dataset must not be accepted."""
    with pytest.raises(ValueError, match="empty"):
        build_training_dataset(gold_sample.iloc[0:0])


def test_training_contract_rejects_non_numeric_features(
    gold_sample: pd.DataFrame,
) -> None:
    """Feature columns must contain numeric data."""
    invalid_gold = gold_sample.copy()
    invalid_gold[FEATURE_COLUMNS[0]] = "not numeric"

    with pytest.raises(ValueError, match="numeric"):
        build_training_dataset(invalid_gold)


# ---------------------------------------------------------------------------
# Chronological splitting
# ---------------------------------------------------------------------------

def test_time_splits_keep_timestamps_together_across_cities() -> None:
    """All cities at a timestamp must stay in the same temporal partition."""
    timestamps = pd.date_range("2023-01-01", periods=100, freq="h")
    gold_df = pd.DataFrame(
        {
            "city": ["Paris"] * 100 + ["Berlin"] * 100,
            "timestamp": list(timestamps) + list(timestamps),
        }
    )

    splits = build_time_splits(gold_df, n_splits=5)

    assert len(splits) == 5

    for train_idx, validation_idx in splits:
        train = gold_df.iloc[train_idx]
        validation = gold_df.iloc[validation_idx]

        assert train["timestamp"].max() < validation["timestamp"].min()
        assert set(train["timestamp"]).isdisjoint(
            set(validation["timestamp"])
        )

        assert train.groupby("timestamp")["city"].nunique().eq(2).all()
        assert validation.groupby("timestamp")["city"].nunique().eq(2).all()


def test_time_splits_reject_missing_timestamps() -> None:
    """Cross-validation must reject missing timestamps."""
    gold_df = pd.DataFrame(
        {"timestamp": [pd.Timestamp("2023-01-01"), pd.NaT]}
    )

    with pytest.raises(ValueError, match="missing values"):
        build_time_splits(gold_df, n_splits=2)


def test_time_splits_reject_too_few_unique_timestamps() -> None:
    """There must be more unique timestamps than splits."""
    gold_df = pd.DataFrame(
        {"timestamp": pd.date_range("2023-01-01", periods=3, freq="h")}
    )

    with pytest.raises(ValueError, match="greater than n_splits"):
        build_time_splits(gold_df, n_splits=3)


def test_train_test_split_is_chronological(
    gold_sample: pd.DataFrame,
) -> None:
    """The final test period must occur after the development period."""
    development, test = split_train_test_by_time(
        gold_sample,
        test_fraction=0.25,
    )

    assert not development.empty
    assert not test.empty
    assert development["timestamp"].max() < test["timestamp"].min()
    assert len(development) + len(test) == len(gold_sample)


def test_train_test_split_keeps_timestamp_groups_together() -> None:
    """Rows sharing a timestamp must not be divided across the split."""
    timestamps = pd.date_range("2023-01-01", periods=20, freq="h")
    gold_df = pd.DataFrame(
        {
            "city": ["Paris"] * 20 + ["Berlin"] * 20,
            "timestamp": list(timestamps) + list(timestamps),
        }
    )

    development, test = split_train_test_by_time(
        gold_df,
        test_fraction=0.2,
    )

    assert development["timestamp"].max() < test["timestamp"].min()
    assert development.groupby("timestamp")["city"].nunique().eq(2).all()
    assert test.groupby("timestamp")["city"].nunique().eq(2).all()


def test_train_test_split_rejects_invalid_fraction(
    gold_sample: pd.DataFrame,
) -> None:
    """The test fraction must be strictly between zero and one."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        split_train_test_by_time(gold_sample, test_fraction=1.0)


# ---------------------------------------------------------------------------
# Persistence baseline and evaluation
# ---------------------------------------------------------------------------

def test_persistence_predicts_current_pollutant_values(
    gold_sample: pd.DataFrame,
) -> None:
    """The persistence baseline uses current values as next-hour predictions."""
    predictions = predict_persistence(gold_sample)

    assert list(predictions.columns) == list(TARGET_COLUMNS)
    assert len(predictions) == len(gold_sample)

    for pollutant, target in zip(POLLUTANTS, TARGET_COLUMNS):
        pd.testing.assert_series_equal(
            predictions[target].reset_index(drop=True),
            gold_sample[pollutant].reset_index(drop=True),
            check_names=False,
        )


def test_persistence_rejects_missing_pollutant(
    gold_sample: pd.DataFrame,
) -> None:
    """The baseline must reject missing pollutant inputs."""
    invalid_gold = gold_sample.drop(columns=[POLLUTANTS[0]])

    with pytest.raises(ValueError, match="Missing pollutant"):
        predict_persistence(invalid_gold)


def test_evaluation_returns_metrics_for_every_target(
    gold_sample: pd.DataFrame,
) -> None:
    """Evaluation returns MAE, RMSE and R² for each pollutant."""
    _, y_true = build_training_dataset(gold_sample)
    y_pred = y_true.copy()

    metrics = evaluate_predictions(y_true, y_pred)

    assert list(metrics["target"]) == list(TARGET_COLUMNS)
    assert {"mae", "rmse", "r2"}.issubset(metrics.columns)
    assert np.allclose(metrics["mae"], 0.0)
    assert np.allclose(metrics["rmse"], 0.0)
    assert np.allclose(metrics["r2"], 1.0)


def test_evaluation_rejects_missing_targets(
    gold_sample: pd.DataFrame,
) -> None:
    """Evaluation must fail when a target is missing."""
    _, y_true = build_training_dataset(gold_sample)
    invalid_predictions = y_true.drop(columns=[TARGET_COLUMNS[0]])

    with pytest.raises(ValueError, match="Missing target"):
        evaluate_predictions(y_true, invalid_predictions)


# ---------------------------------------------------------------------------
# Model selection and result summaries
# ---------------------------------------------------------------------------

def test_summarize_model_results_averages_metrics() -> None:
    """Summary metrics are averaged by model and target."""
    results = pd.DataFrame(
        [
            {
                "model": "linear_regression",
                "fold": 1,
                "target": TARGET_COLUMNS[0],
                "mae": 2.0,
                "rmse": 3.0,
                "r2": 0.5,
            },
            {
                "model": "linear_regression",
                "fold": 2,
                "target": TARGET_COLUMNS[0],
                "mae": 4.0,
                "rmse": 5.0,
                "r2": 0.7,
            },
        ]
    )

    summary = summarize_model_results(results)

    assert len(summary) == 1
    assert summary.iloc[0]["mae"] == pytest.approx(3.0)
    assert summary.iloc[0]["rmse"] == pytest.approx(4.0)
    assert summary.iloc[0]["r2"] == pytest.approx(0.6)


def test_select_best_model_ignores_persistence_baseline() -> None:
    """Selection chooses the lowest mean RMSE among trainable models."""
    results = pd.DataFrame(
        [
            {"model": "persistence", "target": "a", "rmse": 0.1},
            {"model": "linear_regression", "target": "a", "rmse": 2.0},
            {"model": "linear_regression", "target": "b", "rmse": 4.0},
            {"model": "random_forest", "target": "a", "rmse": 1.0},
            {"model": "random_forest", "target": "b", "rmse": 2.0},
        ]
    )

    best_model, summary = select_best_model(results)

    assert best_model == "random_forest"
    assert summary.iloc[0]["model"] == "random_forest"
    assert summary.iloc[0]["mean_rmse"] == pytest.approx(1.5)


def test_select_best_model_rejects_missing_rmse() -> None:
    """Selection must reject missing RMSE values."""
    results = pd.DataFrame(
        [{"model": "linear_regression", "target": "a", "rmse": np.nan}]
    )

    with pytest.raises(ValueError, match="missing values"):
        select_best_model(results)


# ---------------------------------------------------------------------------
# Model artifact persistence
# ---------------------------------------------------------------------------

def test_save_model_artifacts_creates_expected_files(
    tmp_path,
    gold_sample: pd.DataFrame,
) -> None:
    """The fitted model, metrics and metadata are persisted to disk."""
    X, y = build_training_dataset(gold_sample)

    model = LinearRegression()
    model.fit(X, y)

    metrics = pd.DataFrame(
        [
            {"target": target, "mae": 0.1, "rmse": 0.2, "r2": 0.9}
            for target in TARGET_COLUMNS
        ]
    )

    artifacts = save_model_artifacts(
        model=model,
        model_name="linear_regression",
        evaluation_metrics=metrics,
        output_dir=tmp_path / "model_results",
    )

    model_path = tmp_path / "model_results" / "best_model.joblib"
    metrics_path = tmp_path / "model_results" / "final_test_metrics.csv"
    metadata_path = tmp_path / "model_results" / "model_metadata.json"

    assert model_path.is_file()
    assert metrics_path.is_file()
    assert metadata_path.is_file()

    assert artifacts["model_path"] == str(model_path)
    assert artifacts["metrics_path"] == str(metrics_path)
    assert artifacts["metadata_path"] == str(metadata_path)

    loaded_model = joblib.load(model_path)
    assert loaded_model.predict(X).shape == y.shape

    saved_metrics = pd.read_csv(metrics_path)
    assert list(saved_metrics["target"]) == list(TARGET_COLUMNS)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_name"] == "linear_regression"
    assert metadata["feature_columns"] == list(FEATURE_COLUMNS)
    assert metadata["target_columns"] == list(TARGET_COLUMNS)
    assert len(metadata["evaluation"]) == len(TARGET_COLUMNS)
    assert "created_at_utc" in metadata
