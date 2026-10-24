
"""Airflow DAG for AirSense model training and evaluation."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import sys

PROJECT_ROOT = Path("/opt/airflow")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

from scripts.training.train_model import (
    build_training_dataset,
    run_training_workflow,
)

GOLD_DIR = Path("/opt/airflow/data/gold")
RESULTS_DIR = Path("/opt/airflow/models")

DEFAULT_ARGS = {
    "owner": "airsense",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def validate_gold_files() -> None:
    """Check that Gold Parquet files exist and are readable."""
    if not GOLD_DIR.is_dir():
        raise FileNotFoundError(
            f"Gold directory not found: {GOLD_DIR}"
        )

    gold_files = sorted(GOLD_DIR.glob("*.parquet"))

    if not gold_files:
        raise FileNotFoundError(
            f"No Parquet files found in {GOLD_DIR}"
        )

    for path in gold_files:
        if path.stat().st_size == 0:
            raise ValueError(f"Gold file is empty: {path}")

        try:
            # Read only the timestamp column for a lightweight check.
            pd.read_parquet(path, columns=["timestamp"])
        except Exception as exc:
            raise ValueError(
                f"Gold Parquet file is unreadable or lacks "
                f"the timestamp column: {path}"
            ) from exc

    logging.info(
        "Validated %d Gold Parquet files in %s",
        len(gold_files),
        GOLD_DIR,
    )


def train_and_evaluate_model() -> None:
    """Load Gold data and execute the training workflow."""
    gold_files = sorted(GOLD_DIR.glob("*.parquet"))

    if not gold_files:
        raise FileNotFoundError(
            f"No Gold Parquet files found in {GOLD_DIR}"
        )

    logging.info("Loading %d Gold Parquet files", len(gold_files))

    gold_df = pd.concat(
        [pd.read_parquet(path) for path in gold_files],
        ignore_index=True,
    )

    logging.info(
        "Loaded Gold dataset: %d rows and %d columns",
        len(gold_df),
        len(gold_df.columns),
    )

    # Validate the feature/target contract before model training.
    build_training_dataset(gold_df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result = run_training_workflow(
        gold_df=gold_df,
        output_dir=RESULTS_DIR,
    )

    logging.info(
        "Training completed. Selected model: %s",
        result["model_name"],
    )
    logging.info(
        "MLflow run ID: %s",
        result["run_id"],
    )
    logging.info(
        "Final test metrics:\n%s",
        result["final_metrics"].to_string(index=False),
    )
    logging.info(
        "Saved model artifacts: %s",
        result["artifacts"],
    )


with DAG(
    dag_id="airsense_model_training",
    default_args=DEFAULT_ARGS,
    description=(
        "Validate Gold data, train and evaluate AirSense models, "
        "and log results to MLflow"
    ),
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["airsense", "ml", "training"],
) as dag:

    validate_gold_task = PythonOperator(
        task_id="validate_gold_files",
        python_callable=validate_gold_files,
    )

    train_model_task = PythonOperator(
        task_id="train_and_evaluate_model",
        python_callable=train_and_evaluate_model,
        do_xcom_push=False,
    )

    validate_gold_task >> train_model_task
