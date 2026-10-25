
"""Tests for the structure and alignment of Bronze ingestion payloads."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.validation.data_validation import (
    EXPECTED_POLLUTANTS,
    EXPECTED_WEATHER_FEATURES,
)

EXPECTED_HOURS = 8760


@pytest.fixture
def bronze_dir(tmp_path: Path) -> Path:
    """Create valid synthetic hourly Bronze payloads for 2021."""
    city = "paris"
    timestamps = pd.date_range(
        "2021-01-01 00:00:00",
        periods=EXPECTED_HOURS,
        freq="h",
    ).strftime("%Y-%m-%dT%H:%M:%S").tolist()

    air_quality = {"time": timestamps}
    for pollutant in EXPECTED_POLLUTANTS:
        air_quality[pollutant] = [1.0] * EXPECTED_HOURS

    weather = {"time": timestamps}
    weather_defaults = {
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
    for feature in EXPECTED_WEATHER_FEATURES:
        weather[feature] = [weather_defaults[feature]] * EXPECTED_HOURS

    air_path = (
        tmp_path / "air_quality" / city / "air_quality_2021.json"
    )
    weather_path = (
        tmp_path / "open_meteo" / city / "weather_2021.json"
    )
    air_path.parent.mkdir(parents=True)
    weather_path.parent.mkdir(parents=True)

    air_path.write_text(
        json.dumps({"hourly": air_quality}), encoding="utf-8"
    )
    weather_path.write_text(
        json.dumps({"hourly": weather}), encoding="utf-8"
    )

    return tmp_path


def load_json(filepath: Path) -> dict:
    """Load a JSON payload and require its file to exist."""
    assert filepath.is_file(), f"Expected file not found: {filepath}"
    return json.loads(filepath.read_text(encoding="utf-8"))


def test_air_quality_payload_structure(bronze_dir: Path) -> None:
    """Verify hourly timestamps and pollutant fields."""
    path = bronze_dir / "air_quality" / "paris" / "air_quality_2021.json"
    data = load_json(path)
    assert "hourly" in data

    hourly = data["hourly"]
    assert len(hourly["time"]) == EXPECTED_HOURS
    for pollutant in EXPECTED_POLLUTANTS:
        assert pollutant in hourly
        assert len(hourly[pollutant]) == EXPECTED_HOURS


def test_weather_payload_structure(bronze_dir: Path) -> None:
    """Verify hourly timestamps and weather fields."""
    path = bronze_dir / "open_meteo" / "paris" / "weather_2021.json"
    data = load_json(path)
    assert "hourly" in data

    hourly = data["hourly"]
    assert len(hourly["time"]) == EXPECTED_HOURS
    for feature in EXPECTED_WEATHER_FEATURES:
        assert feature in hourly
        assert len(hourly[feature]) == EXPECTED_HOURS


def test_temporal_alignment_between_sources(bronze_dir: Path) -> None:
    """Verify air quality and weather timestamps align exactly."""
    air = load_json(
        bronze_dir / "air_quality" / "paris" / "air_quality_2021.json"
    )
    weather = load_json(
        bronze_dir / "open_meteo" / "paris" / "weather_2021.json"
    )
    assert air["hourly"]["time"] == weather["hourly"]["time"]

