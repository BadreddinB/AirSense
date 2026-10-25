"""Batch feature drift monitoring for AirSense."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from scripts.training.train_model import FEATURE_COLUMNS


def monitor_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: str | Path = "reports/monitoring",
) -> dict[str, Any]:
    """Compare reference and current feature distributions and save a report."""
    if reference_df.empty:
        raise ValueError("Reference dataset must not be empty.")

    if current_df.empty:
        raise ValueError("Current dataset must not be empty.")

    missing_reference = sorted(set(FEATURE_COLUMNS) - set(reference_df.columns))
    missing_current = sorted(set(FEATURE_COLUMNS) - set(current_df.columns))

    if missing_reference:
        raise ValueError(
            f"Reference dataset is missing required features: {missing_reference}"
        )

    if missing_current:
        raise ValueError(
            f"Current dataset is missing required features: {missing_current}"
        )

    reference_features = reference_df.loc[:, FEATURE_COLUMNS].copy()
    current_features = current_df.loc[:, FEATURE_COLUMNS].copy()

    for column in FEATURE_COLUMNS:
        reference_features[column] = pd.to_numeric(
            reference_features[column], errors="coerce"
        )
        current_features[column] = pd.to_numeric(
            current_features[column], errors="coerce"
        )

    if reference_features.isna().all().any():
        raise ValueError("Reference dataset contains an entirely invalid feature.")

    if current_features.isna().all().any():
        raise ValueError("Current dataset contains an entirely invalid feature.")

    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )
    result = report.run(
        reference_data=reference_features,
        current_data=current_features,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    html_path = output_path / f"feature_drift_{timestamp}.html"
    json_path = output_path / f"feature_drift_{timestamp}.json"

    result.save_html(str(html_path))

    report_data = result.dict()
    with json_path.open("w", encoding="utf-8") as report_file:
        json.dump(report_data, report_file, indent=2, default=str)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_rows": len(reference_features),
        "current_rows": len(current_features),
        "feature_count": len(FEATURE_COLUMNS),
        "html_report": str(html_path),
        "json_report": str(json_path),
    }

    logging.info(
        "Feature drift report created: %s",
        html_path,
    )

    return summary
