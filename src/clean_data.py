#!/usr/bin/env python3
"""Validate the supplied ward-level dengue CSV and add historical features.

No third-party packages are required. Per project-owner confirmation,
``dengue_cases`` is preserved as the number of new cases in the associated
seven-day ward reporting period; it is not cumulative. The exact week boundary
convention is not documented, so the source observation date remains its label.

Run from the repository root with::

    python3 src/clean_data.py

Optional positional arguments override the input and output CSV paths.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = (
    REPOSITORY_ROOT
    / "data"
    / "raw"
    / "hackathon_preli_dataset - bangladesh_dengue_large_dataset.csv"
)

DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "data"
    / "processed"
    / "dengue_dataset.csv"
)

REQUIRED_COLUMNS = {
    "date",
    "city",
    "ward_id",
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
    "avg_temp_c",
    "max_temp_c",
    "min_temp_c",
    "rainfall_mm",
    "relative_humidity_pct",
    "dengue_cases",
}
NUMERIC_COLUMNS = {
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
    "avg_temp_c",
    "max_temp_c",
    "min_temp_c",
    "rainfall_mm",
    "relative_humidity_pct",
    "dengue_cases",
}
MISSING_MARKERS = {"", "na", "n/a", "null", "none", "nan"}
ENGINEERED_COLUMNS = [
    "rain_lag_1w",
    "rain_lag_2w",
    "rain_lag_3w",
    "rain_lag_4w",
    "rain_sum_2w",
    "rain_sum_4w",
    *[f"dengue_cases_lag_{week}w" for week in range(1, 5)],
    *[f"avg_temp_C_lag_{week}w" for week in range(1, 5)],
    *[f"relative_humidity_pct_lag_{week}w" for week in range(1, 5)],
    "avg_temp_C_mean_prev_4w",
    "relative_humidity_pct_mean_prev_4w",
    "calendar_week",
    "calendar_month",
]


class DatasetValidationError(ValueError):
    """Raised when data cannot safely be processed without guessing."""


def _format_number(value: float | int) -> str:
    """Write compact numeric text without unnecessary trailing zeroes."""
    if isinstance(value, int):
        return str(value)
    return format(value, ".12g")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames:
                raise DatasetValidationError("Input CSV is empty or has no header.")
            columns = list(reader.fieldnames)
            rows: list[dict[str, str]] = []
            malformed_lines: list[int] = []
            for row in reader:
                if None in row:
                    malformed_lines.append(reader.line_num)
                    continue
                rows.append(row)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetValidationError(f"Could not read input CSV: {exc}") from exc

    if malformed_lines:
        raise DatasetValidationError(
            "CSV rows contain more fields than the header at line(s): "
            f"{malformed_lines[:10]}"
        )
    missing_columns = sorted(REQUIRED_COLUMNS - set(columns))
    if missing_columns:
        raise DatasetValidationError(f"Missing required column(s): {missing_columns}")
    if len(columns) != len(set(columns)):
        raise DatasetValidationError("Input CSV contains duplicate column names.")
    if "avg_temp_C" in columns:
        raise DatasetValidationError(
            "Input already contains avg_temp_C; refusing to overwrite it while "
            "normalizing avg_temp_c."
        )
    if not rows:
        raise DatasetValidationError("Input CSV has no data rows.")
    return columns, rows


def _drop_rows_with_missing(
    rows: list[dict[str, str]],
    columns: list[str],
) -> list[dict[str, str]]:
    """Remove source rows containing missing values in any column."""

    complete_rows = []
    dropped_count = 0

    for row in rows:
        has_missing = any(
            row.get(column) is None
            or row[column].strip().lower() in MISSING_MARKERS
            for column in columns
        )

        if has_missing:
            dropped_count += 1
        else:
            complete_rows.append(row)

    if not complete_rows:
        raise DatasetValidationError(
            "No complete rows remain after removing missing values."
        )

    print(f"Original rows: {len(rows):,}")
    print(f"Rows removed due to missing values: {dropped_count:,}")
    print(f"Complete rows retained: {len(complete_rows):,}")

    return complete_rows


def _drop_incomplete_engineered_rows(
    rows: list[dict[str, Any]],
    columns: list[str],
) -> list[dict[str, Any]]:
    """Remove rows with missing source or engineered values before export."""

    complete_rows: list[dict[str, Any]] = []
    dropped_count = 0

    for row in rows:
        has_missing = any(
            row.get(column) is None
            or (
                isinstance(row.get(column), str)
                and row[column].strip().lower() in MISSING_MARKERS
            )
            for column in columns
        )

        if has_missing:
            dropped_count += 1
        else:
            complete_rows.append(row)

    if not complete_rows:
        raise DatasetValidationError(
            "No complete rows remain after feature engineering."
        )

    print("\nFINAL MISSING-VALUE CLEANING")
    print(f"Engineered rows: {len(rows):,}")
    print(f"Incomplete rows removed: {dropped_count:,}")
    print(f"Complete rows retained: {len(complete_rows):,}")

    return complete_rows


def _validate_labels(rows: list[dict[str, str]]) -> None:
    for column in ("city", "ward_id"):
        spellings: dict[str, set[str]] = defaultdict(set)
        for line_number, row in enumerate(rows, start=2):
            label = row[column]
            if label != label.strip():
                raise DatasetValidationError(
                    f"{column} has surrounding whitespace on CSV line {line_number}: "
                    f"{label!r}"
                )
            spellings[label.casefold()].add(label)
        inconsistent = {
            key: sorted(values) for key, values in spellings.items() if len(values) > 1
        }
        if inconsistent:
            raise DatasetValidationError(
                f"{column} has case-inconsistent labels: {inconsistent}"
            )

    ward_cities: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        ward_cities[row["ward_id"]].add(row["city"])
    inconsistent_wards = {
        ward: sorted(cities) for ward, cities in ward_cities.items() if len(cities) > 1
    }
    if inconsistent_wards:
        raise DatasetValidationError(
            f"Ward IDs map to multiple city labels: {inconsistent_wards}"
        )


def _parse_dates(rows: list[dict[str, str]]) -> list[datetime]:
    parsed: list[datetime] = []
    errors: list[tuple[int, str]] = []
    for line_number, row in enumerate(rows, start=2):
        try:
            value = datetime.fromisoformat(row["date"])
        except ValueError:
            errors.append((line_number, row["date"]))
            continue
        parsed.append(value)
    if errors:
        raise DatasetValidationError(f"Invalid ISO date value(s): {errors[:10]}")
    return parsed


def _parse_numeric(rows: list[dict[str, str]]) -> list[dict[str, float]]:
    parsed_rows: list[dict[str, float]] = []
    errors: list[str] = []
    for line_number, row in enumerate(rows, start=2):
        numeric: dict[str, float] = {}
        for column in NUMERIC_COLUMNS:
            try:
                value = float(row[column])
            except (TypeError, ValueError):
                errors.append(f"line {line_number} {column}={row[column]!r}")
                continue
            if not math.isfinite(value):
                errors.append(f"line {line_number} {column}={row[column]!r} (non-finite)")
                continue
            numeric[column] = value
        parsed_rows.append(numeric)
    if errors:
        raise DatasetValidationError("Invalid numeric value(s): " + "; ".join(errors[:10]))

    for index, numeric in enumerate(parsed_rows, start=2):
        for column in ("population_density_sqkm", "rainfall_mm", "dengue_cases"):
            if numeric[column] < 0:
                errors.append(f"line {index} {column} is negative ({numeric[column]})")
        if numeric["population_density_sqkm"] <= 0:
            errors.append(f"line {index} population_density_sqkm is not positive")
        for column in ("water_logging_risk_index", "sanitation_score"):
            if not 0 <= numeric[column] <= 1:
                errors.append(f"line {index} {column} is outside [0, 1]")
        if not 0 <= numeric["relative_humidity_pct"] <= 100:
            errors.append(f"line {index} relative_humidity_pct is outside [0, 100]")
        for column in ("min_temp_c", "avg_temp_c", "max_temp_c"):
            if not -90 <= numeric[column] <= 60:
                errors.append(f"line {index} {column} is outside [-90, 60] °C")
        if not (
            numeric["min_temp_c"]
            <= numeric["avg_temp_c"]
            <= numeric["max_temp_c"]
        ):
            errors.append(
                f"line {index} violates min_temp_c <= avg_temp_c <= max_temp_c"
            )
        if not numeric["dengue_cases"].is_integer():
            errors.append(f"line {index} dengue_cases is not an integer count")
    if errors:
        raise DatasetValidationError(
            "Impossible or inconsistent numeric value(s): " + "; ".join(errors[:10])
        )
    return parsed_rows


def _validate_duplicates(
    rows: list[dict[str, str]], dates: list[datetime]
) -> None:
    seen_rows: set[tuple[str | None, ...]] = set()
    duplicate_row_count = 0
    seen_keys: set[tuple[datetime, str]] = set()
    duplicate_keys: list[tuple[str, str]] = []
    for row, current_date in zip(rows, dates):
        full_row = tuple(row.get(column) for column in row)
        if full_row in seen_rows:
            duplicate_row_count += 1
        seen_rows.add(full_row)

        key = (current_date, row["ward_id"])
        if key in seen_keys:
            duplicate_keys.append((current_date.date().isoformat(), row["ward_id"]))
        seen_keys.add(key)
    if duplicate_row_count:
        raise DatasetValidationError(
            f"Found {duplicate_row_count} exact duplicate row(s)."
        )
    if duplicate_keys:
        raise DatasetValidationError(
            "Found duplicate (date, ward_id) records: "
            f"{duplicate_keys[:10]}"
        )


def _validate_weekly_continuity(
    rows: list[dict[str, str]], dates: list[datetime]
) -> None:
    ward_dates: dict[str, list[datetime]] = defaultdict(list)
    for row, current_date in zip(rows, dates):
        ward_dates[row["ward_id"]].append(current_date)

    problems: list[str] = []
    for ward_id, current_dates in ward_dates.items():
        chronological = sorted(current_dates)
        broken = [
            (previous, current)
            for previous, current in zip(chronological, chronological[1:])
            if (current - previous).days != 7
        ]
        if broken:
            examples = [
                f"{previous.date().isoformat()}->{current.date().isoformat()}"
                for previous, current in broken[:3]
            ]
            problems.append(
                f"{ward_id}: {len(broken)} non-weekly interval(s), examples {examples}"
            )
    if problems:
        raise DatasetValidationError(
            "Weekly date continuity failed (expected exactly 7 days per ward): "
            + "; ".join(problems[:10])
        )


def _lag_value(values: list[float], index: int, weeks: int) -> float | None:
    if index < weeks:
        return None
    return values[index - weeks]


def _rolling_previous(
    values: list[float], index: int, window: int, operation: str
) -> float | None:
    # The slice ends at index, so only observations before the current row enter.
    if index < window:
        return None
    previous = values[index - window : index]
    if operation == "sum":
        return sum(previous)
    if operation == "mean":
        return sum(previous) / window
    raise ValueError(f"Unsupported rolling operation: {operation}")


def _engineer_features(
    rows: list[dict[str, str]],
    parsed_dates: list[date],
    numeric_rows: list[dict[str, float]],
    original_columns: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    enriched: list[dict[str, Any]] = []
    indices_by_ward: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices_by_ward[row["ward_id"]].append(index)

    for ward_id, row_indices in indices_by_ward.items():
        row_indices.sort(key=lambda index: parsed_dates[index])
        rain = [numeric_rows[index]["rainfall_mm"] for index in row_indices]
        cases = [numeric_rows[index]["dengue_cases"] for index in row_indices]
        temps = [numeric_rows[index]["avg_temp_c"] for index in row_indices]
        humidity = [numeric_rows[index]["relative_humidity_pct"] for index in row_indices]

        for position, source_index in enumerate(row_indices):
            original = dict(rows[source_index])
            original["date"] = parsed_dates[source_index].date().isoformat()
            output_row: dict[str, Any] = {}
            for column in original_columns:
                output_name = "avg_temp_C" if column == "avg_temp_c" else column
                if column == "date":
                    output_row[output_name] = original[column]
                elif column in NUMERIC_COLUMNS:
                    value = numeric_rows[source_index][column]
                    output_row[output_name] = _format_number(value)
                else:
                    output_row[output_name] = original[column]

            for week in range(1, 5):
                output_row[f"rain_lag_{week}w"] = _format_number(
                    _lag_value(rain, position, week)
                ) if position >= week else ""
            rain_2w = _rolling_previous(rain, position, 2, "sum")
            rain_4w = _rolling_previous(rain, position, 4, "sum")
            output_row["rain_sum_2w"] = "" if rain_2w is None else _format_number(rain_2w)
            output_row["rain_sum_4w"] = "" if rain_4w is None else _format_number(rain_4w)

            for source_name, values, prefix in (
                ("dengue_cases", cases, "dengue_cases_lag"),
                ("avg_temp_C", temps, "avg_temp_C_lag"),
                ("relative_humidity_pct", humidity, "relative_humidity_pct_lag"),
            ):
                for week in range(1, 5):
                    lag = _lag_value(values, position, week)
                    output_row[f"{prefix}_{week}w"] = (
                        "" if lag is None else _format_number(lag)
                    )

            temp_mean = _rolling_previous(temps, position, 4, "mean")
            humidity_mean = _rolling_previous(humidity, position, 4, "mean")
            output_row["avg_temp_C_mean_prev_4w"] = (
                "" if temp_mean is None else _format_number(temp_mean)
            )
            output_row["relative_humidity_pct_mean_prev_4w"] = (
                "" if humidity_mean is None else _format_number(humidity_mean)
            )
            iso_week = parsed_dates[source_index].isocalendar().week
            output_row["calendar_week"] = str(iso_week)
            output_row["calendar_month"] = str(parsed_dates[source_index].month)
            enriched.append(output_row)

    enriched.sort(key=lambda row: (row["ward_id"], row["date"]))
    normalized_original = [
        "avg_temp_C" if column == "avg_temp_c" else column
        for column in original_columns
    ]
    output_columns = [
        column for column in normalized_original if column != "dengue_cases"
    ] + ENGINEERED_COLUMNS + ["dengue_cases"]
    return enriched, output_columns


def clean_dataset(
    input_path: Path,
    output_path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Clean source rows, engineer features, and export only complete rows."""

    if input_path.resolve() == output_path.resolve():
        raise DatasetValidationError(
            "Input and output paths must differ; refusing to overwrite input."
        )

    # Read source data and remove records that are incomplete in the raw CSV.
    columns, rows = _read_csv(input_path)
    rows = _drop_rows_with_missing(rows, columns)

    # Validate source data before deriving historical features.
    _validate_labels(rows)
    parsed_dates = _parse_dates(rows)
    numeric_rows = _parse_numeric(rows)
    _validate_duplicates(rows, parsed_dates)
    _validate_weekly_continuity(rows, parsed_dates)

    # Engineer lag and rolling-window features within each ward.
    enriched, output_columns = _engineer_features(
        rows,
        parsed_dates,
        numeric_rows,
        columns,
    )

    if len(enriched) != len(rows):
        raise AssertionError(
            "Internal error: row count changed during feature creation."
        )

    # Feature engineering intentionally leaves early observations blank because
    # prior weeks do not exist. Remove those rows before writing the training CSV.
    enriched = _drop_incomplete_engineered_rows(
        enriched,
        output_columns,
    )

    # Build and verify a sibling temporary file before replacing a previous
    # processed dataset, so an interrupted export cannot leave a partial CSV.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            writer = csv.DictWriter(destination, fieldnames=output_columns)
            writer.writeheader()
            writer.writerows(enriched)
            destination.flush()
            os.fsync(destination.fileno())
        with temporary_path.open("r", newline="", encoding="utf-8") as generated:
            reader = csv.DictReader(generated)
            if reader.fieldnames != output_columns:
                raise DatasetValidationError("Generated CSV header verification failed.")
            verified_rows = 0
            for row in reader:
                if None in row or any(value is None or value.strip().lower() in MISSING_MARKERS for value in row.values()):
                    raise DatasetValidationError("Generated CSV contains malformed or missing values.")
                verified_rows += 1
        if verified_rows != len(enriched):
            raise DatasetValidationError(
                f"Generated CSV row verification failed: expected {len(enriched)}, found {verified_rows}."
            )
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"\nFinal dataset saved: {output_path}")
    print(f"Final dataset rows: {len(enriched):,}")
    print(f"Final dataset columns: {len(output_columns):,}")

    return enriched, output_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate dengue ward data and add historical weather/case features."
    )
    parser.add_argument("input_csv", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("output_csv", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        processed, _ = clean_dataset(args.input_csv, args.output_csv)
    except (DatasetValidationError, FileNotFoundError) as exc:
        print(f"Data cleaning stopped: {exc}", file=sys.stderr)
        return 1

    wards = {row["ward_id"] for row in processed}
    dates = [datetime.fromisoformat(row["date"]) for row in processed]
    print(f"Validated and processed {len(processed):,} rows.")
    print(f"Wards: {len(wards):,}")
    print(
        f"Date range: {min(dates).date().isoformat()} to "
        f"{max(dates).date().isoformat()}"
    )
    print(f"Output: {args.output_csv}")
    print(
        "Case values were preserved as supplied; the source does not specify "
        "whether they are weekly new cases or cumulative totals."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
