
"""Airflow DAG for Silver-to-Gold transformation in AirSense."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/opt/airflow")

from airflow import DAG
from airflow.operators.python import PythonOperator

from scripts.transformation.transform_gold import (
    process_gold_features,
)

SILVER_DIR = Path("/opt/airflow/data/silver")
GOLD_DIR = Path("/opt/airflow/data/gold")
YEARS = (2021, 2022, 2023)

DEFAULT_ARGS = {
    "owner": "airsense",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def validate_silver() -> None:
    """Validate that all expected Silver Parquet files exist and are readable."""
    if not SILVER_DIR.is_dir():
        raise FileNotFoundError(f"Silver directory not found: {SILVER_DIR}")

    cities = sorted(
        path.name for path in SILVER_DIR.iterdir() if path.is_dir()
    )

    if len(cities) != 12:
        raise ValueError(
            f"Expected 12 Silver city directories, found {len(cities)}"
        )

    validated_files = 0

    for city in cities:
        for year in YEARS:
            path = SILVER_DIR / city / f"{year}.parquet"

            if not path.is_file():
                raise FileNotFoundError(f"Silver file not found: {path}")

            df = pd.read_parquet(path)

            if df.empty:
                raise ValueError(f"Silver file is empty: {path}")

            required_columns = {"city", "timestamp"}
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(
                    f"Missing columns in {path}: {sorted(missing_columns)}"
                )

            validated_files += 1

    logging.info("Successfully validated %d Silver files", validated_files)


def transform_gold() -> None:
    """Create one Gold Parquet file per city across all three years."""
    cities = sorted(
        path.name for path in SILVER_DIR.iterdir() if path.is_dir()
    )
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    transformed_cities = 0

    for city in cities:
        yearly_frames = []

        for year in YEARS:
            path = SILVER_DIR / city / f"{year}.parquet"
            df = pd.read_parquet(path)
            yearly_frames.append(df)

        silver_df = pd.concat(yearly_frames, ignore_index=True)
        silver_df["timestamp"] = pd.to_datetime(
            silver_df["timestamp"], errors="raise"
        )
        silver_df = silver_df.sort_values("timestamp").reset_index(drop=True)

        gold_df = process_gold_features(silver_df)

        output_path = GOLD_DIR / f"{city}.parquet"
        gold_df.to_parquet(
            output_path,
            engine="pyarrow",
            index=False,
            compression="snappy",
        )

        logging.info(
            "Saved Gold dataset for %s: %s (%d rows)",
            city,
            output_path,
            len(gold_df),
        )
        transformed_cities += 1

    logging.info(
        "Successfully transformed %d city datasets", transformed_cities
    )


with DAG(
    dag_id="airsense_silver_to_gold",
    default_args=DEFAULT_ARGS,
    description="Validate Silver data and build Gold features across 2021-2023",
    start_date=datetime(2021, 1, 1),
    schedule=None,
    catchup=False,
    tags=["airsense", "silver", "gold", "transformation"],
) as dag:
    validate_task = PythonOperator(
        task_id="validate_silver",
        python_callable=validate_silver,
    )

    transform_task = PythonOperator(
        task_id="transform_gold",
        python_callable=transform_gold,
    )

    validate_task >> transform_task
