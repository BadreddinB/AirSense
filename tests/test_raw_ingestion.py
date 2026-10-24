"""
Integration tests for Bronze/Raw ingestion outputs.

Verifies payload completeness, expected variable presence,
and temporal alignment between air quality and weather datasets.
"""

from __future__ import annotations

import json
from pathlib import Path

# Paths to the test artifacts
AIR_QUALITY_FILE = Path("data/bronze/air_quality/paris/air_quality_2021.json")
WEATHER_FILE = Path("data/bronze/open_meteo/paris/weather_2021.json")

EXPECTED_HOURS_LEAP_YEAR = 8760  # 365 days * 24 hours 

EXPECTED_POLLUTANTS = [
    "pm2_5",
    "pm10",
    "ozone",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "carbon_monoxide",
]

EXPECTED_WEATHER_FEATURES = [
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
]


def load_json(filepath: Path) -> dict:
    """Helper function to read and decode a JSON file."""
    assert filepath.is_file(), f"Expected file not found: {filepath}"
    with filepath.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def test_air_quality_payload_structure() -> None:
    """Verify that the air quality JSON contains valid structure and all pollutants."""
    data = load_json(AIR_QUALITY_FILE)
    assert "hourly" in data, "Key 'hourly' missing from air quality data"

    hourly = data["hourly"]
    timestamps = hourly.get("time", [])

    assert len(timestamps) == EXPECTED_HOURS_LEAP_YEAR, (
        f"Expected {EXPECTED_HOURS_LEAP_YEAR} timestamps, got {len(timestamps)}"
    )

    for pollutant in EXPECTED_POLLUTANTS:
        assert pollutant in hourly, f"Pollutant feature '{pollutant}' missing"
        assert len(hourly[pollutant]) == EXPECTED_HOURS_LEAP_YEAR, (
            f"Pollutant '{pollutant}' length does not match timestamp count"
        )


def test_weather_payload_structure() -> None:
    """Verify that the weather JSON contains valid structure and all weather variables."""
    data = load_json(WEATHER_FILE)
    assert "hourly" in data, "Key 'hourly' missing from weather data"

    hourly = data["hourly"]
    timestamps = hourly.get("time", [])

    assert len(timestamps) == EXPECTED_HOURS_LEAP_YEAR, (
        f"Expected {EXPECTED_HOURS_LEAP_YEAR} timestamps, got {len(timestamps)}"
    )

    for feature in EXPECTED_WEATHER_FEATURES:
        assert feature in hourly, f"Weather variable '{feature}' missing"
        assert len(hourly[feature]) == EXPECTED_HOURS_LEAP_YEAR, (
            f"Weather variable '{feature}' length does not match timestamp count"
        )


def test_temporal_alignment_between_sources() -> None:
    """Verify that weather and air quality timestamps match exactly."""
    air_quality_data = load_json(AIR_QUALITY_FILE)
    weather_data = load_json(WEATHER_FILE)

    aq_times = air_quality_data["hourly"]["time"]
    weather_times = weather_data["hourly"]["time"]

    assert aq_times == weather_times, (
        "Timestamps between air quality and weather are not aligned"
    )