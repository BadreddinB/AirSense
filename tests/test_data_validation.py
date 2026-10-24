
"""
Unit tests for Bronze data validation using Great Expectations.

Tests schema adherence, completeness, physical ranges, timestamp uniqueness,
and hourly continuity for air quality and weather datasets.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.validation.data_validation import (
    EXPECTED_POLLUTANTS,
    EXPECTED_WEATHER_FEATURES,
    validate_air_quality_data,
    validate_weather_data,
)


@pytest.fixture
def valid_air_quality_df() -> pd.DataFrame:
    """Return a minimal valid hourly air quality DataFrame."""
    return pd.DataFrame(
        {
            "time": [
                "2021-01-01T00:00",
                "2021-01-01T01:00",
                "2021-01-01T02:00",
            ],
            "pm2_5": [12.5, 14.2, 10.1],
            "pm10": [22.0, 25.1, 18.4],
            "ozone": [45.0, 42.0, 48.5],
            "nitrogen_dioxide": [18.2, 21.0, 16.5],
            "sulphur_dioxide": [3.1, 2.9, 3.4],
            "carbon_monoxide": [280.0, 310.0, 260.0],
        }
    )


@pytest.fixture
def valid_weather_df() -> pd.DataFrame:
    """Return a minimal valid hourly weather DataFrame."""
    return pd.DataFrame(
        {
            "time": [
                "2024-01-01T00:00",
                "2024-01-01T01:00",
                "2024-01-01T02:00",
            ],
            "temperature_2m": [5.2, 4.8, 4.1],
            "relative_humidity_2m": [82.0, 85.0, 88.0],
            "wind_speed_10m": [14.5, 12.0, 10.2],
            "wind_direction_10m": [220.0, 230.0, 215.0],
            "precipitation": [0.0, 0.2, 0.0],
            "dew_point_2m": [2.1, 2.4, 2.2],
            "surface_pressure": [1013.2, 1012.8, 1012.5],
            "cloud_cover": [75.0, 80.0, 90.0],
            "shortwave_radiation": [0.0, 0.0, 0.0],
            "boundary_layer_height": [350.0, 320.0, 300.0],
        }
    )


def test_constants_completeness() -> None:
    """Ensure expected column lists match the project specification."""
    assert len(EXPECTED_POLLUTANTS) == 6
    assert len(EXPECTED_WEATHER_FEATURES) == 10
    assert "pm2_5" in EXPECTED_POLLUTANTS
    assert "temperature_2m" in EXPECTED_WEATHER_FEATURES


def test_validate_air_quality_success(
    valid_air_quality_df: pd.DataFrame,
) -> None:
    """Valid air quality data must pass validation."""
    success, details = validate_air_quality_data(valid_air_quality_df)

    assert success is True
    assert details["success"] is True


def test_validate_air_quality_negative_values(
    valid_air_quality_df: pd.DataFrame,
) -> None:
    """Negative pollutant concentrations must fail validation."""
    corrupted_df = valid_air_quality_df.copy()
    corrupted_df.loc[0, "pm2_5"] = -10.0

    success, details = validate_air_quality_data(corrupted_df)

    assert success is False
    assert details["success"] is False


def test_validate_air_quality_duplicate_timestamps(
    valid_air_quality_df: pd.DataFrame,
) -> None:
    """Duplicate timestamps must fail validation."""
    corrupted_df = valid_air_quality_df.copy()
    corrupted_df.loc[1, "time"] = corrupted_df.loc[0, "time"]

    success, details = validate_air_quality_data(corrupted_df)

    assert success is False
    assert details["success"] is False


def test_validate_air_quality_missing_column(
    valid_air_quality_df: pd.DataFrame,
) -> None:
    """Missing pollutant columns must fail validation."""
    corrupted_df = valid_air_quality_df.drop(columns=["pm2_5"])

    success, details = validate_air_quality_data(corrupted_df)

    assert success is False
    assert details["success"] is False


def test_validate_air_quality_missing_hour(
    valid_air_quality_df: pd.DataFrame,
) -> None:
    """Missing hourly timestamps must fail validation."""
    corrupted_df = valid_air_quality_df.drop(index=1).reset_index(drop=True)

    success, details = validate_air_quality_data(corrupted_df)

    assert success is False
    assert details["success"] is False


def test_validate_weather_success(
    valid_weather_df: pd.DataFrame,
) -> None:
    """Valid weather data must pass validation."""
    success, details = validate_weather_data(valid_weather_df)

    assert success is True
    assert details["success"] is True


def test_validate_weather_out_of_bounds_humidity(
    valid_weather_df: pd.DataFrame,
) -> None:
    """Relative humidity above 100% must fail validation."""
    corrupted_df = valid_weather_df.copy()
    corrupted_df.loc[0, "relative_humidity_2m"] = 125.0

    success, details = validate_weather_data(corrupted_df)

    assert success is False
    assert details["success"] is False


def test_validate_weather_duplicate_timestamps(
    valid_weather_df: pd.DataFrame,
) -> None:
    """Duplicate timestamps must fail validation."""
    corrupted_df = valid_weather_df.copy()
    corrupted_df.loc[1, "time"] = corrupted_df.loc[0, "time"]

    success, details = validate_weather_data(corrupted_df)

    assert success is False
    assert details["success"] is False


def test_validate_weather_missing_column(
    valid_weather_df: pd.DataFrame,
) -> None:
    """Missing weather columns must fail validation."""
    corrupted_df = valid_weather_df.drop(columns=["temperature_2m"])

    success, details = validate_weather_data(corrupted_df)

    assert success is False
    assert details["success"] is False


def test_validate_weather_missing_hour(
    valid_weather_df: pd.DataFrame,
) -> None:
    """Missing hourly timestamps must fail validation."""
    corrupted_df = valid_weather_df.drop(index=1).reset_index(drop=True)

    success, details = validate_weather_data(corrupted_df)

    assert success is False
    assert details["success"] is False
