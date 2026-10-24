
"""Silver layer transformation module for AirSense.

Loads Bronze JSON data, validates air quality and weather datasets,
merges them, and prepares a clean Silver DataFrame.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.validation.data_validation import (
    EXPECTED_POLLUTANTS,
    EXPECTED_WEATHER_FEATURES,
    validate_air_quality_data,
    validate_weather_data,
)


__all__ = [
    "EXPECTED_POLLUTANTS",
    "EXPECTED_WEATHER_FEATURES",
    "process_city_year",
    "save_silver_parquet",
]


def _find_bronze_file(
    base_dir: Path,
    sub_dir: str,
    city: str,
    prefix: str,
    year: int,
) -> Path:
    """Locate a Bronze JSON file for a city and year."""
    filename = f"{prefix}_{year}.json"
    candidates = [
        base_dir / sub_dir / city / filename,
        base_dir / sub_dir / city.lower() / filename,
        base_dir / sub_dir / city.capitalize() / filename,
    ]

    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"Bronze file not found for city '{city}' and year {year} "
        f"in {base_dir / sub_dir}"
    )


def _load_hourly_json(file_path: Path) -> pd.DataFrame:
    """Load hourly data from an Open-Meteo JSON file."""
    with file_path.open(encoding="utf-8") as file:
        payload: dict[str, Any] = json.load(file)

    if "hourly" not in payload:
        raise KeyError(
            f"Payload in {file_path.name} is missing required 'hourly' key"
        )

    return pd.DataFrame(payload["hourly"])


def process_city_year(
    city: str,
    year: int,
    base_bronze_dir: str | Path = "data/bronze",
) -> pd.DataFrame:
    """Validate and transform Bronze data into a Silver DataFrame."""
    base_path = Path(base_bronze_dir)

    air_quality_path = _find_bronze_file(
        base_path, "air_quality", city, "air_quality", year
    )
    weather_path = _find_bronze_file(
        base_path, "open_meteo", city, "weather", year
    )

    air_quality_df = _load_hourly_json(air_quality_path)
    weather_df = _load_hourly_json(weather_path)

    air_quality_valid, air_quality_report = validate_air_quality_data(
        air_quality_df
    )
    if not air_quality_valid:
        raise ValueError(
            f"Air quality validation failed for {city} ({year}): "
            f"{air_quality_report}"
        )

    weather_valid, weather_report = validate_weather_data(weather_df)
    if not weather_valid:
        raise ValueError(
            f"Weather validation failed for {city} ({year}): "
            f"{weather_report}"
        )

    # Ensure both datasets cover exactly the same timestamps.
    if set(air_quality_df["time"]) != set(weather_df["time"]):
        raise ValueError(
            f"Timestamp mismatch between air quality and weather data "
            f"for {city} ({year})"
        )

    joined_df = air_quality_df.merge(
        weather_df,
        on="time",
        how="inner",
        validate="one_to_one",
    )

    joined_df["timestamp"] = pd.to_datetime(
        joined_df.pop("time"),
        errors="raise",
    )
    joined_df["city"] = city

    numeric_columns = EXPECTED_POLLUTANTS + EXPECTED_WEATHER_FEATURES
    joined_df[numeric_columns] = joined_df[numeric_columns].astype(float)

    silver_df = joined_df[["city", "timestamp", *numeric_columns]]

    return silver_df.sort_values("timestamp").reset_index(drop=True)


def save_silver_parquet(
    df: pd.DataFrame,
    city: str,
    year: int,
    base_silver_dir: str | Path = "data/silver",
) -> Path:
    """Save a Silver DataFrame as a Snappy-compressed Parquet file."""
    target_dir = Path(base_silver_dir) / city.lower()
    target_dir.mkdir(parents=True, exist_ok=True)

    target_file = target_dir / f"{year}.parquet"
    df.to_parquet(
        target_file,
        engine="pyarrow",
        index=False,
        compression="snappy",
    )

    return target_file
