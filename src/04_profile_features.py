"""Profile every feature in the UCI Bank Marketing dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank-additional-full.csv"
EXPORT_DIR = PROJECT_ROOT / "data" / "exports"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

NUMERIC_FEATURES = [
    "age",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

FEATURE_FAMILIES = {
    "age": "client",
    "job": "client",
    "marital": "client",
    "education": "client",
    "default": "client",
    "housing": "client",
    "loan": "client",
    "contact": "current campaign",
    "month": "current campaign",
    "day_of_week": "current campaign",
    "duration": "current call",
    "campaign": "current campaign",
    "pdays": "previous campaign",
    "previous": "previous campaign",
    "poutcome": "previous campaign",
    "emp.var.rate": "economic context",
    "cons.price.idx": "economic context",
    "cons.conf.idx": "economic context",
    "euribor3m": "economic context",
    "nr.employed": "economic context",
}


def load_data() -> pd.DataFrame:
    """Load the verified raw dataset."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Run src/01_download_data.py first."
        )

    return pd.read_csv(RAW_DATA_PATH, sep=";")


def build_feature_inventory(data: pd.DataFrame) -> pd.DataFrame:
    """Create one inventory row for every candidate feature."""
    records: list[dict] = []

    for feature in CATEGORICAL_FEATURES + NUMERIC_FEATURES:
        counts = data[feature].value_counts(dropna=False)
        top_value = counts.index[0]
        top_count = int(counts.iloc[0])

        is_categorical = feature in CATEGORICAL_FEATURES

        record = {
            "feature": feature,
            "feature_family": FEATURE_FAMILIES[feature],
            "feature_type": (
                "categorical" if is_categorical else "numeric"
            ),
            "distinct_values": int(data[feature].nunique(dropna=False)),
            "null_cells": int(data[feature].isna().sum()),
            "unknown_cells": (
                int((data[feature] == "unknown").sum())
                if is_categorical
                else 0
            ),
            "top_value": str(top_value),
            "top_value_count": top_count,
            "top_value_share": top_count / len(data),
            "minimum": (
                None if is_categorical else float(data[feature].min())
            ),
            "median": (
                None if is_categorical else float(data[feature].median())
            ),
            "maximum": (
                None if is_categorical else float(data[feature].max())
            ),
        }
        records.append(record)

    return pd.DataFrame(records)


def build_unknown_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize semantic missingness stored as 'unknown'."""
    records = []

    for feature in CATEGORICAL_FEATURES:
        unknown_count = int((data[feature] == "unknown").sum())

        if unknown_count > 0:
            records.append(
                {
                    "feature": feature,
                    "unknown_count": unknown_count,
                    "unknown_share": unknown_count / len(data),
                }
            )

    return (
        pd.DataFrame(records)
        .sort_values("unknown_count", ascending=False)
        .reset_index(drop=True)
    )


def build_categorical_profile(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate counts and response rates for categorical values."""
    profiles = []

    for feature in CATEGORICAL_FEATURES:
        grouped = (
            data.groupby(feature, dropna=False)
            .agg(
                observations=("y", "size"),
                subscribers=(
                    "y",
                    lambda values: int((values == "yes").sum()),
                ),
            )
            .reset_index()
            .rename(columns={feature: "category"})
        )

        grouped.insert(0, "feature", feature)
        grouped["observation_share"] = (
            grouped["observations"] / len(data)
        )
        grouped["response_rate"] = (
            grouped["subscribers"] / grouped["observations"]
        )
        grouped["is_unknown"] = grouped["category"].eq("unknown")
        grouped = grouped.sort_values(
            "observations",
            ascending=False,
        )

        profiles.append(grouped)

    return pd.concat(profiles, ignore_index=True)


def build_numeric_profile(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate descriptive statistics for numeric features."""
    rows = []

    for feature in NUMERIC_FEATURES:
        series = data[feature]

        rows.append(
            {
                "feature": feature,
                "observations": int(series.count()),
                "distinct_values": int(series.nunique()),
                "mean": float(series.mean()),
                "standard_deviation": float(series.std()),
                "minimum": float(series.min()),
                "percentile_01": float(series.quantile(0.01)),
                "percentile_25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "percentile_75": float(series.quantile(0.75)),
                "percentile_99": float(series.quantile(0.99)),
                "maximum": float(series.max()),
            }
        )

    return pd.DataFrame(rows)


def plot_unknown_values(unknown_summary: pd.DataFrame) -> None:
    """Create a chart of semantic missing values."""
    plot_data = unknown_summary.sort_values(
        "unknown_share",
        ascending=True,
    )

    figure, axis = plt.subplots(figsize=(9, 5.5))

    bars = axis.barh(
        plot_data["feature"],
        plot_data["unknown_share"] * 100,
        color="#B45309",
    )

    labels = [
        f"{count:,} ({share:.1%})"
        for count, share in zip(
            plot_data["unknown_count"],
            plot_data["unknown_share"],
            strict=True,
        )
    ]

    axis.bar_label(bars, labels=labels, padding=4)
    axis.set_title("Categorical values recorded as unknown")
    axis.set_xlabel("Share of observations (%)")
    axis.set_ylabel("Feature")
    axis.set_xlim(
        0,
        max(plot_data["unknown_share"] * 100) * 1.3,
    )
    axis.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "03_unknown_category_values.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    """Create feature-profile exports and visualization."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()

    feature_inventory = build_feature_inventory(data)
    unknown_summary = build_unknown_summary(data)
    categorical_profile = build_categorical_profile(data)
    numeric_profile = build_numeric_profile(data)

    feature_inventory.to_csv(
        EXPORT_DIR / "feature_inventory.csv",
        index=False,
    )
    unknown_summary.to_csv(
        EXPORT_DIR / "unknown_value_summary.csv",
        index=False,
    )
    categorical_profile.to_csv(
        EXPORT_DIR / "categorical_feature_profile.csv",
        index=False,
    )
    numeric_profile.to_csv(
        EXPORT_DIR / "numeric_feature_profile.csv",
        index=False,
    )

    plot_unknown_values(unknown_summary)

    print("\nFEATURE INVENTORY")
    print(
        feature_inventory[
            [
                "feature",
                "feature_family",
                "feature_type",
                "distinct_values",
                "unknown_cells",
                "minimum",
                "median",
                "maximum",
            ]
        ].to_string(index=False)
    )

    print("\nUNKNOWN VALUE SUMMARY")
    print(
        unknown_summary.to_string(
            index=False,
            formatters={
                "unknown_share": lambda value: f"{value:.2%}",
            },
        )
    )

    print("\nIMPORTANT DATA NOTES")
    print(
        "- `unknown` is semantic missingness, not a conventional null."
    )
    print(
        "- `duration` is profiled for auditing but will be excluded "
        "from deployable pre-call models."
    )
    print(
        "- Rare values and categories will be reviewed before "
        "preprocessing decisions are finalized."
    )

    print("\nFeature profiles and missing-value chart created successfully.")


if __name__ == "__main__":
    main()