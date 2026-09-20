"""Create chronological training, validation, and test datasets."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank-additional-full.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXPORT_DIR = PROJECT_ROOT / "data" / "exports"

MONTH_NUMBER = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

TRAIN_END = "2008-11"
VALIDATION_START = "2008-12"
VALIDATION_END = "2009-05"
TEST_START = "2009-06"


def load_data() -> pd.DataFrame:
    """Load the verified raw data."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Run src/01_download_data.py first."
        )

    return pd.read_csv(RAW_DATA_PATH, sep=";")


def infer_month_periods(data: pd.DataFrame) -> pd.Series:
    """Infer month periods from UCI's documented chronological order."""
    current_year = 2008
    previous_month_number: int | None = None
    periods: list[str] = []

    for month_name in data["month"]:
        current_month_number = MONTH_NUMBER[month_name]

        if (
            previous_month_number is not None
            and current_month_number < previous_month_number
        ):
            current_year += 1

        periods.append(
            f"{current_year:04d}-{current_month_number:02d}"
        )
        previous_month_number = current_month_number

    return pd.Series(periods, index=data.index, dtype="string")


def assign_splits(data: pd.DataFrame) -> pd.DataFrame:
    """Assign each observation to one chronological split."""
    prepared = data.copy()

    prepared.insert(
        0,
        "observation_id",
        [f"OBS{number:06d}" for number in range(1, len(data) + 1)],
    )
    prepared.insert(
        1,
        "observation_order",
        range(1, len(data) + 1),
    )

    prepared["inferred_period"] = infer_month_periods(prepared)
    prepared["target"] = prepared["y"].map({"no": 0, "yes": 1})
    prepared["split"] = "unassigned"

    train_mask = prepared["inferred_period"] <= TRAIN_END
    validation_mask = (
        prepared["inferred_period"].between(
            VALIDATION_START,
            VALIDATION_END,
        )
    )
    test_mask = prepared["inferred_period"] >= TEST_START

    prepared.loc[train_mask, "split"] = "train"
    prepared.loc[validation_mask, "split"] = "validation"
    prepared.loc[test_mask, "split"] = "test"

    return prepared


def validate_splits(data: pd.DataFrame) -> None:
    """Validate split completeness, exclusivity, and chronology."""
    if bool((data["split"] == "unassigned").any()):
        raise ValueError("Some observations were not assigned to a split.")

    if data["observation_id"].duplicated().any():
        raise ValueError("Observation IDs are not unique.")

    split_counts = data["split"].value_counts()

    if int(split_counts.sum()) != len(data):
        raise ValueError("Split row counts do not reconcile to the source.")

    train_max_order = data.loc[
        data["split"] == "train",
        "observation_order",
    ].max()
    validation_min_order = data.loc[
        data["split"] == "validation",
        "observation_order",
    ].min()
    validation_max_order = data.loc[
        data["split"] == "validation",
        "observation_order",
    ].max()
    test_min_order = data.loc[
        data["split"] == "test",
        "observation_order",
    ].min()

    if not (
        train_max_order < validation_min_order
        and validation_max_order < test_min_order
    ):
        raise ValueError("The splits are not strictly chronological.")


def build_split_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize the locked chronological splits."""
    summary = (
        data.groupby("split", sort=False)
        .agg(
            start_row=("observation_order", "min"),
            end_row=("observation_order", "max"),
            start_period=("inferred_period", "first"),
            end_period=("inferred_period", "last"),
            observations=("target", "size"),
            subscribers=("target", "sum"),
        )
        .reset_index()
    )

    summary["non_subscribers"] = (
        summary["observations"] - summary["subscribers"]
    )
    summary["response_rate"] = (
        summary["subscribers"] / summary["observations"]
    )

    split_order = pd.CategoricalDtype(
        categories=["train", "validation", "test"],
        ordered=True,
    )
    summary["split"] = summary["split"].astype(split_order)
    summary = summary.sort_values("split").reset_index(drop=True)
    summary["split"] = summary["split"].astype("string")

    return summary


def save_splits(data: pd.DataFrame) -> None:
    """Save each split as a separate Parquet file."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "validation", "test"]:
        split_data = (
            data.loc[data["split"] == split_name]
            .reset_index(drop=True)
        )

        output_path = PROCESSED_DIR / f"{split_name}.parquet"
        split_data.to_parquet(output_path, index=False)

        print(f"Saved {split_name}: {output_path}")


def save_manifest(
    split_summary: pd.DataFrame,
) -> None:
    """Save the split definition and limitations."""
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "chronological month boundaries",
        "source_order": (
            "UCI documents the full dataset as ordered from "
            "May 2008 through November 2010."
        ),
        "boundaries": {
            "train": {
                "start": "2008-05",
                "end": TRAIN_END,
            },
            "validation": {
                "start": VALIDATION_START,
                "end": VALIDATION_END,
            },
            "test": {
                "start": TEST_START,
                "end": "2010-11",
            },
        },
        "purpose": {
            "train": "Fit preprocessing and model parameters.",
            "validation": (
                "Compare models, calibration methods, and thresholds."
            ),
            "test": (
                "Perform one final evaluation after all choices are locked."
            ),
        },
        "limitations": [
            "The public data does not contain complete contact dates.",
            "Year-month labels are inferred from documented ordering.",
            "The data does not contain a customer identifier.",
            "Repeated customers cannot be grouped across splits.",
            "Response prevalence changes substantially over time.",
        ],
        "summary": split_summary.to_dict(orient="records"),
    }

    manifest_path = EXPORT_DIR / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """Create, validate, summarize, and save chronological splits."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = load_data()
    split_data = assign_splits(raw_data)

    validate_splits(split_data)
    split_summary = build_split_summary(split_data)

    split_summary.to_csv(
        EXPORT_DIR / "split_summary.csv",
        index=False,
    )
    save_splits(split_data)
    save_manifest(split_summary)

    print("\nCHRONOLOGICAL SPLIT SUMMARY")
    print(
        split_summary.to_string(
            index=False,
            formatters={
                "response_rate": lambda value: f"{value:.2%}",
            },
        )
    )

    print("\nSplit controls:")
    print("- All source rows were assigned exactly once.")
    print("- Training observations occur before validation observations.")
    print("- Validation observations occur before test observations.")
    print("- Test data is now locked for final evaluation only.")
    print(
        "- Split boundaries must not be changed in response to "
        "model performance."
    )


if __name__ == "__main__":
    main()