
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import tempfile
import time
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import requests

API_URL = "https://archive-api.open-meteo.com/v1/archive"
OUTPUT_DIR = Path("data/bronze/open_meteo")
LOG_DIR = Path("logs/ingestion")

START_YEAR = 2021
END_YEAR = 2023

WEATHER_FEATURES = [
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

MAX_ATTEMPTS = 6
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 60
MAX_RETRY_DELAY = 90
REQUEST_DELAY = 2

# Coordinates of the 44 European capitals
CITIES = {
    "Paris": (48.8566, 2.3522),
    "Madrid": (40.4168, -3.7038),
    "London": (51.5072, -0.1276),
    "Berlin": (52.5200, 13.4050),
    "Rome": (41.9028, 12.4964),
    "Lisbon": (38.7223, -9.1393),
    "Brussels": (50.8503, 4.3517),
    "Amsterdam": (52.3676, 4.9041),
    "Stockholm": (59.3293, 18.0686),
    "Copenhagen": (55.6761, 12.5683),
    "Helsinki": (60.1699, 24.9384),
    "Luxembourg": (49.6116, 6.1319),
}


def setup_logging() -> None:
    """Configure console and file logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                LOG_DIR / "ingestion.log",
                encoding="utf-8",
            ),
        ],
        force=True,
    )


def city_slug(city: str) -> str:
    """Create a filesystem-friendly city name."""
    slug = city.lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_-]", "", slug)


def retry_delay(
    response: requests.Response | None,
    attempt: int,
) -> float:
    """Respect Retry-After when available; otherwise use progressive backoff."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")

        if retry_after:
            try:
                return min(float(retry_after), MAX_RETRY_DELAY)
            except ValueError:
                try:
                    retry_date = parsedate_to_datetime(retry_after)
                    if retry_date.tzinfo is None:
                        retry_date = retry_date.replace(tzinfo=timezone.utc)

                    seconds = (
                        retry_date - datetime.now(timezone.utc)
                    ).total_seconds()

                    return min(max(seconds, 0), MAX_RETRY_DELAY)
                except (TypeError, ValueError, OverflowError):
                    pass

        # Dedicated backoff policy for rate limiting (HTTP 429)
        if response.status_code == 429:
            backoff_429 = (
                5.0 * (2 ** (attempt - 1))
            ) + random.uniform(1.0, 3.0)
            return min(backoff_429, MAX_RETRY_DELAY)

    backoff = 2 ** (attempt - 1)
    return min(backoff + random.uniform(0, 1), MAX_RETRY_DELAY)


def validate_payload(
    payload: Any,
    expected_start: str,
    expected_end: str,
) -> bool:
    """Check that the API response contains complete hourly series."""
    if not isinstance(payload, dict):
        return False

    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        return False

    timestamps = hourly.get("time")
    if not isinstance(timestamps, list) or not timestamps:
        return False

    if timestamps[0] != expected_start or timestamps[-1] != expected_end:
        return False

    if len(timestamps) != len(set(timestamps)):
        return False

    for feature in WEATHER_FEATURES:
        values = hourly.get(feature)

        if not isinstance(values, list):
            return False

        if len(values) != len(timestamps):
            return False

    units = payload.get("hourly_units")
    if not isinstance(units, dict):
        return False

    return all(feature in units for feature in WEATHER_FEATURES)


def fetch_json(
    session: requests.Session,
    params: dict[str, Any],
    expected_start: str,
    expected_end: str,
) -> dict[str, Any]:
    """Fetch the API response with retries for transient failures."""
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        response: requests.Response | None = None

        try:
            response = session.get(
                API_URL,
                params=params,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )

            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()

            response.raise_for_status()
            payload = response.json()

            if not validate_payload(
                payload,
                expected_start,
                expected_end,
            ):
                raise ValueError(
                    "Invalid API response: inconsistent timestamps, "
                    "features, or lengths."
                )

            return payload

        except requests.HTTPError as exc:
            last_error = exc
            status = response.status_code if response is not None else None

            # Do not retry non-transient 4xx client errors
            if status is not None and 400 <= status < 500 and status != 429:
                raise

        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.JSONDecodeError,
        ) as exc:
            last_error = exc

        except ValueError:
            raise

        if attempt < MAX_ATTEMPTS:
            delay = retry_delay(response, attempt)
            logging.warning(
                "Attempt %s/%s failed (%s). Retrying in %.1f seconds.",
                attempt,
                MAX_ATTEMPTS,
                last_error,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Failed after {MAX_ATTEMPTS} attempts: {last_error}"
    )


def valid_existing_file(
    path: Path,
    expected_start: str,
    expected_end: str,
) -> bool:
    """Return True only if an existing file has the expected variables."""
    if not path.is_file():
        return False

    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        return validate_payload(
            payload,
            expected_start,
            expected_end,
        )

    except (OSError, json.JSONDecodeError, TypeError):
        return False


def atomic_save(payload: dict[str, Any], output_path: Path) -> None:
    """Write to a temporary file before replacing the destination."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            json.dump(
                payload,
                temporary_file,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        temporary_path.replace(output_path)

    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def ingest_city_year(
    session: requests.Session,
    city: str,
    latitude: float,
    longitude: float,
    year: int,
) -> bool:
    """Download one city's weather data for one calendar year."""
    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)

    expected_start = f"{start_date.isoformat()}T00:00"
    expected_end = f"{end_date.isoformat()}T23:00"

    output_path = OUTPUT_DIR / city_slug(city) / f"weather_{year}.json"

    if valid_existing_file(
        output_path,
        expected_start,
        expected_end,
    ):
        logging.info(
            "Skipping %s, %s: valid existing file.",
            city,
            year,
        )
        return True

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(WEATHER_FEATURES),
        "timezone": "GMT",
    }

    logging.info(
        "Fetching %s, %s (%s to %s)",
        city,
        year,
        start_date,
        end_date,
    )

    payload = fetch_json(
        session,
        params,
        expected_start,
        expected_end,
    )

    atomic_save(payload, output_path)

    logging.info(
        "Saved %s (%s timestamps)",
        output_path,
        len(payload["hourly"]["time"]),
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open-Meteo historical weather ingestion to Bronze."
    )

    parser.add_argument(
        "--cities",
        nargs="+",
        choices=sorted(CITIES),
        help="One or more cities. Defaults to all configured cities.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=START_YEAR,
        help=f"First year to include (default: {START_YEAR}).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=END_YEAR,
        help=f"Last year to include (default: {END_YEAR}).",
    )

    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    if args.start_year > args.end_year:
        logging.error("start-year must be less than or equal to end-year.")
        return 2

    if args.start_year < 1940 or args.end_year > date.today().year:
        logging.error(
            "Invalid year range: use a period between 1940 "
            "and the current year."
        )
        return 2

    selected_cities = args.cities or list(CITIES.keys())

    if len(CITIES) != 12:
        logging.error(
            "Expected 12 configured cities, found %s.",
            len(CITIES),
        )
        return 2

    successful = 0
    failed = 0

    with requests.Session() as session:
        session.headers.update(
            {"User-Agent": "AirSense/1.0 (weather data ingestion)"}
        )

        for city in selected_cities:
            latitude, longitude = CITIES[city]

            for year in range(args.start_year, args.end_year + 1):
                try:
                    ingest_city_year(
                        session,
                        city,
                        latitude,
                        longitude,
                        year,
                    )
                    successful += 1

                except Exception:
                    failed += 1
                    logging.exception(
                        "Ingestion failed for %s, %s",
                        city,
                        year,
                    )

                time.sleep(REQUEST_DELAY)

    logging.info(
        "Finished. Successful/skipped: %s; failed: %s",
        successful,
        failed,
    )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
