"""Unit tests for the Gold transformation pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.transformation.transform_gold import (
    EXPECTED_POLLUTANTS,
    EXPECTED_WEATHER_FEATURES,
    process_gold_features,
)


@pytest.fixture
def sample_silver_df() -> pd.DataFrame:
    """Create synthetic hourly Silver data for testing."""
    timestamps = pd.date_range(
        start="2021-01-01 00:00:00",
        periods=12,
        freq="h",
    )

    data: dict[str, object] = {
        "city": ["Paris"] * len(timestamps),
        "timestamp": timestamps,
    }

    for index, column in enumerate(EXPECTED_POLLUTANTS):
        data[column] = np.arange(len(timestamps), dtype=float) + index * 100

    for index, column in enumerate(EXPECTED_WEATHER_FEATURES):
        data[column] = np.arange(len(timestamps), dtype=float) + index * 10

    return pd.DataFrame(data)


def test_process_gold_features_schema(sample_silver_df: pd.DataFrame) -> None:
    """Verify Gold output columns and row count."""
    gold_df = process_gold_features(sample_silver_df)

    assert isinstance(gold_df, pd.DataFrame)
    assert "city" in gold_df.columns
    assert "timestamp" in gold_df.columns

    for column in EXPECTED_POLLUTANTS + EXPECTED_WEATHER_FEATURES:
        assert column in gold_df.columns

    for column in EXPECTED_POLLUTANTS + EXPECTED_WEATHER_FEATURES:
        for lag in range(1, 6):
            assert f"{column}_lag_{lag}" in gold_df.columns

    for column in (
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
    ):
        assert column in gold_df.columns

    for pollutant in EXPECTED_POLLUTANTS:
        assert f"target_{pollutant}_t1" in gold_df.columns

    # Five historical hours and one future target are required.
    assert len(gold_df) == len(sample_silver_df) - 6


def test_lags_and_future_targets_are_temporally_aligned(
    sample_silver_df: pd.DataFrame,
) -> None:
    """Verify lagged features and t+1 targets use the correct timestamps."""
    gold_df = process_gold_features(sample_silver_df)

    # The first retained row corresponds to source row 5.
    source_index = 5
    gold_row = gold_df.iloc[0]

    assert gold_row["timestamp"] == sample_silver_df.iloc[source_index]["timestamp"]

    for column in EXPECTED_POLLUTANTS + EXPECTED_WEATHER_FEATURES:
        for lag in range(1, 6):
            expected_value = sample_silver_df.iloc[source_index - lag][column]
            assert gold_row[f"{column}_lag_{lag}"] == expected_value

    for pollutant in EXPECTED_POLLUTANTS:
        expected_target = sample_silver_df.iloc[source_index + 1][pollutant]
        assert gold_row[f"target_{pollutant}_t1"] == expected_target


def test_cyclic_features_are_within_expected_range(
    sample_silver_df: pd.DataFrame,
) -> None:
    """Verify cyclic time encodings stay between -1 and 1."""
    gold_df = process_gold_features(sample_silver_df)

    for column in ("hour_sin", "hour_cos", "month_sin", "month_cos"):
        assert gold_df[column].between(-1.0, 1.0).all()