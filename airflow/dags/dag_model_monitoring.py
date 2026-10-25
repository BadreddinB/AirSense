"""Airflow DAG for batch feature drift monitoring."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path("/opt/airflow")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.train_model import FEATURE_COLUMNS
from scripts.validation.model_monitoring import monitor_feature_drift

GOLD_DIR = PROJECT_ROOT / "data" / "gold"
MONITORING_DIR = PROJECT_ROOT / "reports" / "monitoring"

DEFAULT_ARGS = {
    "owner": "airsense",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def monitor_gold_feature_drift() -> None:
    """Compare earlier and later observations for each Gold city dataset."""
    if not GOLD_DIR.is_dir():
        raise FileNotFoundError(f"Gold directory not found: {GOLD_DIR}")

    gold_files = sorted(GOLD_DIR.glob("*.parquet"))
    if not gold_files:
        raise FileNotFoundError(f"No Gold Parquet files found in {GOLD_DIR}")

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    successful_cities = 0

    for path in gold_files:
        logging.info("Monitoring feature drift for %s", path.stem)

        df = pd.read_parquet(path)

        missing = sorted(
            (set(FEATURE_COLUMNS) | {"timestamp"}) - set(df.columns)
        )
        if missing:
            raise ValueError(
                f"{path.name} is missing required columns: {missing}"
            )

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = (
            df.dropna(subset=["timestamp"])
            .sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .reset_index(drop=True)
        )

        if len(df) < 10:
            raise ValueError(
                f"{path.name} has too few valid rows for drift monitoring: "
                f"{len(df)}"
            )

        split_index = int(len(df) * 0.8)
        reference_df = df.iloc[:split_index]
        current_df = df.iloc[split_index:]

        city_output_dir = MONITORING_DIR / path.stem

        summary = monitor_feature_drift(
            reference_df=reference_df,
            current_df=current_df,
            output_dir=city_output_dir,
        )

        logging.info(
            "Drift report for %s: reference_rows=%d, current_rows=%d, "
            "html=%s, json=%s",
            path.stem,
            summary["reference_rows"],
            summary["current_rows"],
            summary["html_report"],
            summary["json_report"],
        )
        successful_cities += 1

    logging.info(
        "Feature drift monitoring completed for %d cities.",
        successful_cities,
    )


with DAG(
    dag_id="airsense_model_monitoring",
    default_args=DEFAULT_ARGS,
    description=(
        "Generate batch feature drift reports for each city in Gold data"
    ),
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["airsense", "ml", "monitoring"],
) as dag:
    monitor_features_task = PythonOperator(
        task_id="monitor_gold_feature_drift",
        python_callable=monitor_gold_feature_drift,
        do_xcom_push=False,
    )