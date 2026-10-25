"""Unit tests for the Silver transformation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.transformation.transform_silver import (
    EXPECTED_POLLUTANTS,
    EXPECTED_WEATHER_FEATURES,
    process_city_year,
    save_silver_parquet,
)


@pytest.fixture
def bronze_dir(tmp_path: Path) -> Path:
    """Create valid synthetic hourly Bronze data for one city and year."""
    city = "paris"
    year = 2021
    hours = 8760
    timestamps = pd.date_range(
        f"{year}-01-01 00:00:00",
        periods=hours,
        freq="h",
    ).strftime("%Y-%m-%dT%H:%M:%S").tolist()

    air_quality = {"time": timestamps}
    for pollutant in EXPECTED_POLLUTANTS:
        air_quality[pollutant] = [1.0] * hours

    weather_values = {
        "temperature_2m": 15.0,
        "relative_humidity_2m": 60.0,
        "wind_speed_10m": 3.0,
        "wind_direction_10m": 180.0,
        "precipitation": 0.0,
        "dew_point_2m": 8.0,
        "surface_pressure": 1013.0,
        "cloud_cover": 20.0,
        "shortwave_radiation": 0.0,
        "boundary_layer_height": 500.0,
    }
    weather = {"time": timestamps}
    for feature in EXPECTED_WEATHER_FEATURES:
        weather[feature] = [weather_values[feature]] * hours

    air_path = tmp_path / "air_quality" / city / f"air_quality_{year}.json"
    weather_path = tmp_path / "open_meteo" / city / f"weather_{year}.json"
    air_path.parent.mkdir(parents=True)
    weather_path.parent.mkdir(parents=True)

    air_path.write_text(
        json.dumps({"hourly": air_quality}), encoding="utf-8"
    )
    weather_path.write_text(
        json.dumps({"hourly": weather}), encoding="utf-8"
    )

    return tmp_path


def test_process_city_year_schema(bronze_dir: Path) -> None:
    """Verify the Silver schema, types, and hourly timestamp integrity."""
    city = "Paris"
    year = 2021

    df = process_city_year(
        city=city,
        year=year,
        base_bronze_dir=bronze_dir,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 8760
    assert "city" in df.columns
    assert "timestamp" in df.columns
    assert (df["city"] == city).all()

    for column in EXPECTED_POLLUTANTS + EXPECTED_WEATHER_FEATURES:
        assert column in df.columns
        assert pd.api.types.is_float_dtype(df[column])

    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["timestamp"].is_monotonic_increasing
    assert not df["timestamp"].duplicated().any()


def test_save_silver_parquet_round_trip(
    bronze_dir: Path,
    tmp_path: Path,
) -> None:
    """Verify Silver Parquet output can be read back without data loss."""
    original_df = process_city_year(
        city="Paris",
        year=2021,
        base_bronze_dir=bronze_dir,
    )

    output_path = save_silver_parquet(
        df=original_df,
        city="Paris",
        year=2021,
        base_silver_dir=tmp_path / "silver",
    )

    restored_df = pd.read_parquet(output_path)

    assert output_path.is_file()
    assert len(restored_df) == len(original_df)
    assert list(restored_df.columns) == list(original_df.columns)
    assert restored_df.dtypes.equals(original_df.dtypes)
    pd.testing.assert_frame_equal(restored_df, original_df)

