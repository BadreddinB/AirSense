"""
Airflow DAG: Raw Data Ingestion (AirSense).

Orchestrates parallel raw ingestion for Open-Meteo weather and air quality APIs.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator

# Default task configuration
DEFAULT_ARGS = {
    "owner": "airsense",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# DAG definition
with DAG(
    dag_id="airsense_ingestion_raw",
    default_args=DEFAULT_ARGS,
    description="Raw ingestion pipeline for air quality and weather datasets",
    start_date=datetime(2021, 1, 1),
    schedule=None,  # Manual trigger to maintain control during development
    catchup=False,
    tags=["airsense", "raw", "ingestion"],
    params={
        "cities": Param(
            default="Paris",
            type="string",
            description="Space-separated list of cities (leave empty to ingest all configured cities)",
        ),
        "start_year": Param(
            default=2021,
            type="integer",
            description="Starting year for ingestion",
        ),
        "end_year": Param(
            default=2023,
            type="integer",
            description="Ending year for ingestion",
        ),
    },
) as dag:

    # Bash command templates
    # Jinja templates dynamically resolve user inputs from the Airflow UI
    CMD_AIR_QUALITY = """
    cd /opt/airflow && \
    python scripts/ingestion/ingest_pollution.py \
        --start-year {{ params.start_year }} \
        --end-year {{ params.end_year }} \
        {% if params.cities %}--cities {{ params.cities }}{% endif %}
    """

    CMD_WEATHER = """
    cd /opt/airflow && \
    python scripts/ingestion/ingest_weather.py \
        --start-year {{ params.start_year }} \
        --end-year {{ params.end_year }} \
        {% if params.cities %}--cities {{ params.cities }}{% endif %}
    """

    # Task definitions
    task_ingest_air_quality = BashOperator(
        task_id="ingest_air_quality_raw",
        bash_command=CMD_AIR_QUALITY,
    )

    task_ingest_weather = BashOperator(
        task_id="ingest_weather_raw",
        bash_command=CMD_WEATHER,
    )

    # Task dependencies: both ingestion tasks run concurrently
    [task_ingest_air_quality, task_ingest_weather]