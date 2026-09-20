"""Validate and profile the raw UCI Bank Marketing dataset."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank-additional-full.csv"
EXPORT_DIR = PROJECT_ROOT / "data" / "exports"

EXPECTED_ROW_COUNT = 41_188

EXPECTED_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "y",
]

EXPECTED_CATEGORIES = {
    "job": {
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown",
    },
    "marital": {
        "divorced",
        "married",
        "single",
        "unknown",
    },
    "education": {
        "basic.4y",
        "basic.6y",
        "basic.9y",
        "high.school",
        "illiterate",
        "professional.course",
        "university.degree",
        "unknown",
    },
    "default": {"no", "yes", "unknown"},
    "housing": {"no", "yes", "unknown"},
    "loan": {"no", "yes", "unknown"},
    "contact": {"cellular", "telephone"},
    "month": {
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    },
    "day_of_week": {"mon", "tue", "wed", "thu", "fri"},
    "poutcome": {"failure", "nonexistent", "success"},
    "y": {"no", "yes"},
}


def add_check(
    checks: list[dict[str, str]],
    name: str,
    status: str,
    details: str,
) -> None:
    """Append one validation result."""
    checks.append(
        {
            "check": name,
            "status": status,
            "details": details,
        }
    )


def load_raw_data() -> pd.DataFrame:
    """Load the semicolon-delimited raw CSV."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Run src/01_download_data.py first."
        )

    return pd.read_csv(RAW_DATA_PATH, sep=";")


def build_validation_checks(data: pd.DataFrame) -> list[dict[str, str]]:
    """Run structural and business-rule validation checks."""
    checks: list[dict[str, str]] = []

    add_check(
        checks,
        "Expected row count",
        "PASS" if len(data) == EXPECTED_ROW_COUNT else "FAIL",
        f"Expected {EXPECTED_ROW_COUNT:,}; found {len(data):,}.",
    )

    add_check(
        checks,
        "Expected column count",
        "PASS" if len(data.columns) == len(EXPECTED_COLUMNS) else "FAIL",
        f"Expected {len(EXPECTED_COLUMNS)}; found {len(data.columns)}.",
    )

    add_check(
        checks,
        "Expected column names and order",
        "PASS" if data.columns.tolist() == EXPECTED_COLUMNS else "FAIL",
        "Columns match the documented schema."
        if data.columns.tolist() == EXPECTED_COLUMNS
        else "Columns differ from the documented schema.",
    )

    null_cells = int(data.isna().sum().sum())
    add_check(
        checks,
        "Conventional null cells",
        "PASS" if null_cells == 0 else "FAIL",
        f"Found {null_cells:,} null cells.",
    )

    duplicate_rows = int(data.duplicated().sum())
    add_check(
        checks,
        "Exact duplicate rows",
        "PASS" if duplicate_rows == 0 else "WARN",
        (
            f"Found {duplicate_rows:,} exact duplicate rows. "
            "They will not be automatically removed because no customer "
            "or contact identifier is available."
        ),
    )

    for column, allowed_values in EXPECTED_CATEGORIES.items():
        observed_values = set(data[column].dropna().unique())
        unexpected_values = sorted(observed_values - allowed_values)

        add_check(
            checks,
            f"Allowed values: {column}",
            "PASS" if not unexpected_values else "FAIL",
            (
                "No unexpected categories."
                if not unexpected_values
                else f"Unexpected categories: {unexpected_values}"
            ),
        )

    numeric_columns = data.select_dtypes(include="number").columns
    infinite_cells = int(
        data[numeric_columns]
        .isin([float("inf"), float("-inf")])
        .sum()
        .sum()
    )
    add_check(
        checks,
        "Infinite numeric values",
        "PASS" if infinite_cells == 0 else "FAIL",
        f"Found {infinite_cells:,} infinite numeric cells.",
    )

    age_is_valid = bool(data["age"].between(0, 120).all())
    add_check(
        checks,
        "Plausible age values",
        "PASS" if age_is_valid else "FAIL",
        (
            f"Observed range: {data['age'].min()} "
            f"to {data['age'].max()} years."
        ),
    )

    duration_is_valid = bool((data["duration"] >= 0).all())
    add_check(
        checks,
        "Nonnegative call duration",
        "PASS" if duration_is_valid else "FAIL",
        f"Minimum duration: {data['duration'].min()} seconds.",
    )

    campaign_is_valid = bool((data["campaign"] >= 1).all())
    add_check(
        checks,
        "Campaign contact count",
        "PASS" if campaign_is_valid else "FAIL",
        f"Observed range: {data['campaign'].min()} to "
        f"{data['campaign'].max()}.",
    )

    previous_is_valid = bool((data["previous"] >= 0).all())
    add_check(
        checks,
        "Previous contact count",
        "PASS" if previous_is_valid else "FAIL",
        f"Observed range: {data['previous'].min()} to "
        f"{data['previous'].max()}.",
    )

    pdays_is_valid = bool(data["pdays"].between(0, 999).all())
    add_check(
        checks,
        "Previous-contact day range",
        "PASS" if pdays_is_valid else "FAIL",
        f"Observed range: {data['pdays'].min()} to "
        f"{data['pdays'].max()}.",
    )

    history_relationship_is_valid = bool(
        (
            (data["previous"] == 0)
            == (data["poutcome"] == "nonexistent")
        ).all()
    )
    add_check(
        checks,
        "Previous contacts and previous outcome",
        "PASS" if history_relationship_is_valid else "WARN",
        (
            "`previous = 0` agrees with `poutcome = nonexistent`."
            if history_relationship_is_valid
            else "Previous-contact fields contain inconsistent states."
        ),
    )

    return checks


def build_profile(data: pd.DataFrame) -> dict:
    """Create a JSON-serializable data profile."""
    unknown_counts = {
        column: int((data[column] == "unknown").sum())
        for column in data.select_dtypes(include="str").columns
        if bool((data[column] == "unknown").any())
    }

    target_counts = {
        str(label): int(count)
        for label, count in data["y"].value_counts().items()
    }

    numeric_summary = {}
    for column in data.select_dtypes(include="number").columns:
        series = data[column]
        numeric_summary[column] = {
            "minimum": float(series.min()),
            "median": float(series.median()),
            "maximum": float(series.max()),
            "distinct_values": int(series.nunique()),
        }

    category_counts = {}
    for column in data.select_dtypes(include="str").columns:
        category_counts[column] = {
            str(label): int(count)
            for label, count in data[column].value_counts().items()
        }

    month_change = data["month"].ne(data["month"].shift())
    month_blocks = [
        {
            "start_row": int(index) + 1,
            "month": str(data.loc[index, "month"]),
        }
        for index in data.index[month_change]
    ]

    pdays_999_previous_positive = int(
        (
            (data["pdays"] == 999)
            & (data["previous"] > 0)
        ).sum()
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": RAW_DATA_PATH.name,
        "row_count": int(len(data)),
        "column_count": int(len(data.columns)),
        "columns": data.columns.tolist(),
        "target": {
            "counts": target_counts,
            "positive_rate": float((data["y"] == "yes").mean()),
        },
        "pandas_null_cells": int(data.isna().sum().sum()),
        "unknown_category_counts": unknown_counts,
        "exact_duplicate_rows": int(data.duplicated().sum()),
        "numeric_summary": numeric_summary,
        "category_counts": category_counts,
        "chronological_month_blocks": month_blocks,
        "history_field_notes": {
            "pdays_999_with_previous_contacts": (
                pdays_999_previous_positive
            ),
            "interpretation": (
                "Do not use pdays=999 alone to identify clients with no "
                "previous contact. Use previous and poutcome together and "
                "treat pdays=999 as a special sentinel category."
            ),
        },
    }


def main() -> None:
    """Run validation, export results, and fail on critical errors."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_raw_data()
    checks = build_validation_checks(data)
    profile = build_profile(data)

    checks_frame = pd.DataFrame(checks)

    checks_path = EXPORT_DIR / "data_validation_checks.csv"
    profile_path = EXPORT_DIR / "data_profile.json"

    checks_frame.to_csv(checks_path, index=False)
    profile_path.write_text(
        json.dumps(profile, indent=2),
        encoding="utf-8",
    )

    print("\nDATA VALIDATION RESULTS")
    print(checks_frame.to_string(index=False))

    print("\nDATASET SUMMARY")
    print(f"Rows: {profile['row_count']:,}")
    print(f"Columns: {profile['column_count']}")
    print(f"Positive outcomes: {target_counts_text(profile)}")
    print(f"Exact duplicate rows: {profile['exact_duplicate_rows']:,}")
    print(
        "pdays=999 with previous contacts: "
        f"{profile['history_field_notes']['pdays_999_with_previous_contacts']:,}"
    )

    print(f"\nChecks saved to: {checks_path}")
    print(f"Profile saved to: {profile_path}")

    failed_checks = checks_frame.loc[
        checks_frame["status"] == "FAIL"
    ]

    if not failed_checks.empty:
        raise SystemExit(
            f"Validation failed: {len(failed_checks)} critical checks failed."
        )

    print("\nAll critical validation checks passed.")


def target_counts_text(profile: dict) -> str:
    """Format the target counts for terminal output."""
    yes_count = profile["target"]["counts"].get("yes", 0)
    positive_rate = profile["target"]["positive_rate"]

    return f"{yes_count:,} ({positive_rate:.2%})"


if __name__ == "__main__":
    main()