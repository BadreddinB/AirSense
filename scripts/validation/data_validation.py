
"""Data quality validation for AirSense Bronze datasets."""

from __future__ import annotations

from typing import Any

import great_expectations as gx
import pandas as pd


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


def _has_hourly_regularity(df: pd.DataFrame) -> bool:
    """Return True when timestamps are valid and spaced one hour apart."""
    if "time" not in df.columns or len(df) < 2:
        return False

    timestamps = pd.to_datetime(df["time"], errors="coerce")

    if timestamps.isna().any():
        return False

    timestamps = timestamps.sort_values().reset_index(drop=True)
    intervals = timestamps.diff().dropna()

    return bool((intervals == pd.Timedelta(hours=1)).all())


def _validate_dataframe(
    df: pd.DataFrame,
    suite_name: str,
    expectations: list[Any],
) -> tuple[bool, dict[str, Any]]:
    """Validate a DataFrame using Great Expectations and hourly checks."""
    if df.empty:
        return False, {
            "success": False,
            "suite_name": suite_name,
            "row_count": 0,
            "error": "The DataFrame is empty.",
            "hourly_regularity": False,
        }

    context = gx.get_context(mode="ephemeral")
    suite = gx.ExpectationSuite(name=suite_name)

    for expectation in expectations:
        suite.add_expectation(expectation)

    data_source = context.data_sources.add_pandas("bronze_source")
    data_asset = data_source.add_dataframe_asset("bronze_data")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "bronze_batch"
    )

    context.suites.add(suite)

    validation_definition = gx.ValidationDefinition(
        name=f"{suite_name}_validation",
        data=batch_definition,
        suite=suite,
    )
    context.validation_definitions.add(validation_definition)

    results = validation_definition.run(
        batch_parameters={"dataframe": df}
    )

    gx_success = bool(results.success)
    hourly_success = _has_hourly_regularity(df)
    success = gx_success and hourly_success

    return success, {
        "success": success,
        "suite_name": suite_name,
        "row_count": len(df),
        "statistics": results.statistics,
        "hourly_regularity": hourly_success,
    }


def _common_expectations() -> list[Any]:
    """Return expectations shared by hourly Bronze datasets."""
    return [
        gx.expectations.ExpectColumnToExist(column="time"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="time"),
        gx.expectations.ExpectColumnValuesToBeUnique(column="time"),
    ]


def validate_air_quality_data(
    df: pd.DataFrame,
) -> tuple[bool, dict[str, Any]]:
    """Validate the schema and values of hourly air quality data."""
    
    expectations = _common_expectations()

    for column in EXPECTED_POLLUTANTS:
        expectations.extend(
            [
                gx.expectations.ExpectColumnToExist(column=column),
                gx.expectations.ExpectColumnValuesToNotBeNull(
                    column=column
                ),
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column=column,
                    min_value=0.0,
                ),
            ]
        )

    return _validate_dataframe(
        df,
        "air_quality_bronze_suite",
        expectations,
    )


def validate_weather_data(
    df: pd.DataFrame,
) -> tuple[bool, dict[str, Any]]:
    """Validate the schema and physical ranges of hourly weather data."""
    expectations = _common_expectations()


    for column in EXPECTED_WEATHER_FEATURES:
        expectations.append(
            gx.expectations.ExpectColumnToExist(column=column)
        )

        if column != "boundary_layer_height":
            expectations.append(
                gx.expectations.ExpectColumnValuesToNotBeNull(
                    column=column
                )
            )

    ranges = {
        "temperature_2m": (-70.0, 70.0),
        "dew_point_2m": (-70.0, 70.0),
        "relative_humidity_2m": (0.0, 100.0),
        "cloud_cover": (0.0, 100.0),
        "wind_direction_10m": (0.0, 360.0),
        "wind_speed_10m": (0.0, None),
        "precipitation": (0.0, None),
        "shortwave_radiation": (0.0, None),
        "boundary_layer_height": (0.0, None),
        "surface_pressure": (700.0, 1150.0),
    }

    for column, (minimum, maximum) in ranges.items():
        expectations.append(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column,
                min_value=minimum,
                max_value=maximum,
            )
        )

    return _validate_dataframe(
        df,
        "weather_bronze_suite",
        expectations,
    )
