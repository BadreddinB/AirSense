
"""Airflow DAG for Bronze-to-Silver transformation in AirSense."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, "/opt/airflow")

from airflow import DAG
from airflow.operators.python import PythonOperator

from scripts.transformation.transform_silver import (
    process_city_year,
    save_silver_parquet,
)
from scripts.validation.data_validation import (
    validate_air_quality_data,
    validate_weather_data,
)

BRONZE_DIR = Path("/opt/airflow/data/bronze")
SILVER_DIR = Path("/opt/airflow/data/silver")
YEARS = range(2021, 2024)

DEFAULT_ARGS = {
    "owner": "airsense",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def _load_hourly_json(file_path: Path) -> pd.DataFrame:
    """Load hourly data from a Bronze JSON file."""
    with file_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    if "hourly" not in payload:
        raise ValueError(f"Missing 'hourly' key in {file_path}")

    return pd.DataFrame(payload["hourly"])


def _find_file(directory: Path, city: str, prefix: str, year: int) -> Path:
    """Find a Bronze file using the expected directory and filename."""
    path = directory / city / f"{prefix}_{year}.json"

    if not path.is_file():
        raise FileNotFoundError(f"Bronze file not found: {path}")

    return path


def validate_bronze() -> None:
    """Validate every Bronze weather and air-quality file pair."""
    weather_root = BRONZE_DIR / "open_meteo"
    air_quality_root = BRONZE_DIR / "air_quality"

    if not weather_root.is_dir() or not air_quality_root.is_dir():
        raise FileNotFoundError(
            f"Expected Bronze directories under {BRONZE_DIR}"
        )

    cities = sorted(
        path.name for path in air_quality_root.iterdir() if path.is_dir()
    )

    if not cities:
        raise ValueError(f"No city directories found in {air_quality_root}")

    validated_pairs = 0

    for city in cities:
        for year in YEARS:
            air_quality_path = _find_file(
                air_quality_root, city, "air_quality", year
            )
            weather_path = _find_file(
                weather_root, city, "weather", year
            )

            air_quality_df = _load_hourly_json(air_quality_path)
            weather_df = _load_hourly_json(weather_path)

            air_quality_ok, air_quality_report = validate_air_quality_data(
                air_quality_df
            )
            weather_ok, weather_report = validate_weather_data(weather_df)

            if not air_quality_ok:
                raise ValueError(
                    f"Air quality validation failed for {city} ({year}): "
                    f"{air_quality_report}"
                )

            if not weather_ok:
                raise ValueError(
                    f"Weather validation failed for {city} ({year}): "
                    f"{weather_report}"
                )

            if set(air_quality_df["time"]) != set(weather_df["time"]):
                raise ValueError(
                    f"Timestamp mismatch for {city} ({year})"
                )

            validated_pairs += 1

    logging.info("Successfully validated %d Bronze file pairs", validated_pairs)


def transform_silver() -> None:
    """Transform all Bronze pairs and save Silver Parquet files."""
    weather_root = BRONZE_DIR / "open_meteo"
    air_quality_root = BRONZE_DIR / "air_quality"

    cities = sorted(
        path.name for path in air_quality_root.iterdir() if path.is_dir()
    )

    transformed_pairs = 0

    for city in cities:
        for year in YEARS:
            df = process_city_year(
                city=city,
                year=year,
                base_bronze_dir=BRONZE_DIR,
            )
            output_path = save_silver_parquet(
                df=df,
                city=city,
                year=year,
                base_silver_dir=SILVER_DIR,
            )
            logging.info("Saved Silver dataset: %s", output_path)
            transformed_pairs += 1

    logging.info("Successfully transformed %d city-year datasets", transformed_pairs)


with DAG(
    dag_id="airsense_bronze_to_silver",
    default_args=DEFAULT_ARGS,
    description="Validate Bronze data and transform it into Silver Parquet files",
    start_date=datetime(2021, 1, 1),
    schedule=None,
    catchup=False,
    tags=["airsense", "bronze", "silver", "transformation"],
) as dag:
    validate_task = PythonOperator(
        task_id="validate_bronze",
        python_callable=validate_bronze,
    )

    transform_task = PythonOperator(
        task_id="transform_silver",
        python_callable=transform_silver,
    )

    validate_task >> transform_task
