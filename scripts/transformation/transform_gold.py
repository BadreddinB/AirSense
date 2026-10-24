"""Gold layer feature engineering for AirSense."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.validation.data_validation import (
    EXPECTED_POLLUTANTS,
    EXPECTED_WEATHER_FEATURES,
)


__all__ = [
    "EXPECTED_POLLUTANTS",
    "EXPECTED_WEATHER_FEATURES",
    "process_gold_features",
    "save_gold_parquet",
]

LAG_HOURS = (1, 2, 3, 4, 5)
TIMESTAMP_COLUMN = "timestamp"
CITY_COLUMN = "city"


def process_gold_features(silver_df: pd.DataFrame) -> pd.DataFrame:
    """Build hourly lag features, cyclic time encodings, and t+1 targets."""
    required_columns = [
        CITY_COLUMN,
        TIMESTAMP_COLUMN,
        *EXPECTED_POLLUTANTS,
        *EXPECTED_WEATHER_FEATURES,
    ]

    missing_columns = sorted(set(required_columns) - set(silver_df.columns))
    if missing_columns:
        raise ValueError(f"Silver DataFrame is missing columns: {missing_columns}")

    if silver_df.empty:
        raise ValueError("Silver DataFrame must not be empty")

    df = silver_df[required_columns].copy()
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="raise")

    if df[TIMESTAMP_COLUMN].isna().any():
        raise ValueError("Timestamps must not contain null values")

    if df.duplicated([CITY_COLUMN, TIMESTAMP_COLUMN]).any():
        raise ValueError("Duplicate city/timestamp pairs are not allowed")

    if df[CITY_COLUMN].nunique(dropna=False) != 1:
        raise ValueError("Process one city at a time")

    df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    # Keep only rows belonging to uninterrupted hourly sequences.
    hourly_steps = df[TIMESTAMP_COLUMN].diff().eq(pd.Timedelta(hours=1))
    segment_ids = (~hourly_steps).cumsum()

    numeric_columns = EXPECTED_POLLUTANTS + EXPECTED_WEATHER_FEATURES
    feature_columns = [
        *numeric_columns,
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
    ]

    df["hour_sin"] = np.sin(2 * np.pi * df[TIMESTAMP_COLUMN].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df[TIMESTAMP_COLUMN].dt.hour / 24)
    df["month_sin"] = np.sin(
        2 * np.pi * (df[TIMESTAMP_COLUMN].dt.month - 1) / 12
    )
    df["month_cos"] = np.cos(
        2 * np.pi * (df[TIMESTAMP_COLUMN].dt.month - 1) / 12
    )

    for lag in LAG_HOURS:
        for column in numeric_columns:
            df[f"{column}_lag_{lag}"] = (
                df.groupby(segment_ids, sort=False)[column].shift(lag)
            )

    for pollutant in EXPECTED_POLLUTANTS:
        df[f"target_{pollutant}_t1"] = (
            df.groupby(segment_ids, sort=False)[pollutant].shift(-1)
        )

    # Remove rows without the required five-hour history or t+1 targets.
    lag_columns = [
        f"{column}_lag_{lag}"
        for column in numeric_columns
        for lag in LAG_HOURS
    ]
    target_columns = [
        f"target_{pollutant}_t1" for pollutant in EXPECTED_POLLUTANTS
    ]

    df = df.dropna(subset=[*lag_columns, *target_columns]).copy()

    output_columns = [
        CITY_COLUMN,
        TIMESTAMP_COLUMN,
        *feature_columns,
        *lag_columns,
        *target_columns,
    ]

    return df[output_columns].reset_index(drop=True)


def save_gold_parquet(
    df: pd.DataFrame,
    city: str,
    year: int,
    base_gold_dir: str | Path = "data/gold",
) -> Path:
    """Save Gold features to a Snappy-compressed Parquet file."""
    target_dir = Path(base_gold_dir) / city.lower()
    target_dir.mkdir(parents=True, exist_ok=True)

    target_file = target_dir / f"{year}.parquet"
    df.to_parquet(
        target_file,
        engine="pyarrow",
        index=False,
        compression="snappy",
    )

    return target_file