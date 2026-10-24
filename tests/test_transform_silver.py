"""
Unit tests for the Silver transformation pipeline.

Verifies schema adherence, column completeness, data types,
and temporal integrity when converting Bronze JSON into Silver DataFrames.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.transformation.transform_silver import (
    EXPECTED_POLLUTANTS,
    EXPECTED_WEATHER_FEATURES,
    process_city_year,
)


def test_process_city_year_schema() -> None:
    """Verify that process_city_year produces a clean DataFrame with all expected features."""
    city = "Paris"
    year = 2021

    df = process_city_year(city=city, year=year)

    # 1. Structure assertions
    assert isinstance(df, pd.DataFrame), "Output must be a pandas DataFrame"
    assert len(df) == 8760, f"Expected 8760 rows for leap year {year}, got {len(df)}"

    # 2. Metadata columns assertions
    assert "city" in df.columns, "Metadata column 'city' missing"
    assert "timestamp" in df.columns, "Timestamp column missing"
    assert (df["city"] == city).all(), "City values do not match input argument"

    # 3. Features and targets presence assertions
    for pollutant in EXPECTED_POLLUTANTS:
        assert pollutant in df.columns, f"Target pollutant column '{pollutant}' missing"
        assert pd.api.types.is_float_dtype(df[pollutant]), f"Column '{pollutant}' must be float"

    for weather_features in EXPECTED_WEATHER_FEATURES:
        assert weather_features in df.columns, f"Weather column '{weather_features}' missing"
        assert pd.api.types.is_float_dtype(df[weather_features]), f"Column '{weather_features}' must be float"

    # 4. Temporal consistency assertions
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]), "Timestamp must be datetime type"
    assert df["timestamp"].is_monotonic_increasing, "Timestamps must be sorted in ascending order"
    assert not df["timestamp"].duplicated().any(), "Duplicate timestamps detected"


def test_save_silver_parquet_round_trip(tmp_path) -> None:
    """Verify that Silver data survives a Parquet write/read round trip."""
    from scripts.transformation.transform_silver import (
        process_city_year,
        save_silver_parquet,
    )

    city = "Paris"
    year = 2021

    original_df = process_city_year(city=city, year=year)

    output_path = save_silver_parquet(
        df=original_df,
        city=city,
        year=year,
        base_silver_dir=tmp_path,
    )

    restored_df = pd.read_parquet(output_path)

    assert output_path.is_file()
    assert len(restored_df) == len(original_df)
    assert list(restored_df.columns) == list(original_df.columns)
    assert restored_df.dtypes.equals(original_df.dtypes)
    pd.testing.assert_frame_equal(restored_df, original_df)
