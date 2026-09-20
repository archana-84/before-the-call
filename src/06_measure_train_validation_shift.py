"""Measure feature distribution shift from training to validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
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
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and validation data without accessing test data."""
    train_path = PROCESSED_DIR / "train.parquet"
    validation_path = PROCESSED_DIR / "validation.parquet"

    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError(
            "Training or validation data is missing. "
            "Run src/05_create_chronological_splits.py first."
        )

    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(validation_path)

    return train, validation


def categorical_total_variation(
    train: pd.Series,
    validation: pd.Series,
) -> float:
    """
    Calculate total variation distance between category distributions.

    Zero means identical distributions. One means no overlap.
    """
    train_share = train.value_counts(normalize=True, dropna=False)
    validation_share = validation.value_counts(
        normalize=True,
        dropna=False,
    )

    categories = train_share.index.union(validation_share.index)

    train_aligned = train_share.reindex(categories, fill_value=0)
    validation_aligned = validation_share.reindex(
        categories,
        fill_value=0,
    )

    return float(
        0.5 * (train_aligned - validation_aligned).abs().sum()
    )


def build_categorical_shift(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profile categorical shift and unseen validation values."""
    shift_rows = []
    unseen_rows = []

    for feature in CATEGORICAL_FEATURES:
        train_categories = set(train[feature].dropna().unique())
        validation_counts = validation[feature].value_counts(
            dropna=False
        )
        validation_categories = set(
            validation[feature].dropna().unique()
        )

        unseen_categories = sorted(
            validation_categories - train_categories
        )

        shift_rows.append(
            {
                "feature": feature,
                "total_variation_distance": (
                    categorical_total_variation(
                        train[feature],
                        validation[feature],
                    )
                ),
                "training_distinct_values": int(
                    train[feature].nunique(dropna=False)
                ),
                "validation_distinct_values": int(
                    validation[feature].nunique(dropna=False)
                ),
                "unseen_validation_categories": len(
                    unseen_categories
                ),
            }
        )

        for category in unseen_categories:
            category_count = int(validation_counts.get(category, 0))

            unseen_rows.append(
                {
                    "feature": feature,
                    "category": category,
                    "validation_observations": category_count,
                    "validation_share": (
                        category_count / len(validation)
                    ),
                }
            )

    categorical_shift = (
        pd.DataFrame(shift_rows)
        .sort_values(
            "total_variation_distance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    unseen = pd.DataFrame(
        unseen_rows,
        columns=[
            "feature",
            "category",
            "validation_observations",
            "validation_share",
        ],
    )

    return categorical_shift, unseen


def build_numeric_shift(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate standardized numeric shifts and range violations."""
    rows = []

    for feature in NUMERIC_FEATURES:
        train_series = train[feature]
        validation_series = validation[feature]

        train_mean = float(train_series.mean())
        validation_mean = float(validation_series.mean())
        train_standard_deviation = float(train_series.std())

        if train_standard_deviation == 0:
            standardized_difference = float("nan")
        else:
            standardized_difference = (
                validation_mean - train_mean
            ) / train_standard_deviation

        train_minimum = float(train_series.min())
        train_maximum = float(train_series.max())

        rows.append(
            {
                "feature": feature,
                "training_mean": train_mean,
                "validation_mean": validation_mean,
                "training_standard_deviation": (
                    train_standard_deviation
                ),
                "standardized_mean_difference": (
                    standardized_difference
                ),
                "absolute_standardized_difference": abs(
                    standardized_difference
                ),
                "training_minimum": train_minimum,
                "training_maximum": train_maximum,
                "validation_below_training_minimum": int(
                    (validation_series < train_minimum).sum()
                ),
                "validation_above_training_maximum": int(
                    (validation_series > train_maximum).sum()
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "absolute_standardized_difference",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def plot_shift(
    categorical_shift: pd.DataFrame,
    numeric_shift: pd.DataFrame,
) -> None:
    """Plot categorical and numeric shift on separate scales."""
    category_plot = categorical_shift.sort_values(
        "total_variation_distance",
        ascending=True,
    )
    numeric_plot = numeric_shift.sort_values(
        "absolute_standardized_difference",
        ascending=True,
    )

    figure, (category_axis, numeric_axis) = plt.subplots(
        1,
        2,
        figsize=(15, 7),
    )

    category_axis.barh(
        category_plot["feature"],
        category_plot["total_variation_distance"],
        color="#475569",
    )
    category_axis.set_title("Categorical distribution shift")
    category_axis.set_xlabel("Total variation distance")
    category_axis.set_ylabel("Feature")
    category_axis.spines[["top", "right"]].set_visible(False)

    numeric_axis.barh(
        numeric_plot["feature"],
        numeric_plot["absolute_standardized_difference"],
        color="#0F766E",
    )
    numeric_axis.set_title("Numeric mean shift")
    numeric_axis.set_xlabel(
        "Absolute standardized mean difference"
    )
    numeric_axis.set_ylabel("Feature")
    numeric_axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "Training-to-validation feature shift "
        "(test data not accessed)",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "04_train_validation_feature_shift.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    """Measure, export, and visualize feature shift."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    train, validation = load_splits()

    categorical_shift, unseen_categories = (
        build_categorical_shift(train, validation)
    )
    numeric_shift = build_numeric_shift(train, validation)

    categorical_shift.to_csv(
        EXPORT_DIR / "categorical_train_validation_shift.csv",
        index=False,
    )
    numeric_shift.to_csv(
        EXPORT_DIR / "numeric_train_validation_shift.csv",
        index=False,
    )
    unseen_categories.to_csv(
        EXPORT_DIR / "unseen_validation_categories.csv",
        index=False,
    )

    plot_shift(categorical_shift, numeric_shift)

    print("\nCATEGORICAL TRAINING-TO-VALIDATION SHIFT")
    print(
        categorical_shift.to_string(
            index=False,
            formatters={
                "total_variation_distance": (
                    lambda value: f"{value:.3f}"
                ),
            },
        )
    )

    print("\nNUMERIC TRAINING-TO-VALIDATION SHIFT")
    print(
        numeric_shift.to_string(
            index=False,
            formatters={
                "standardized_mean_difference": (
                    lambda value: f"{value:.3f}"
                ),
                "absolute_standardized_difference": (
                    lambda value: f"{value:.3f}"
                ),
            },
        )
    )

    print("\nUNSEEN VALIDATION CATEGORIES")
    if unseen_categories.empty:
        print("None")
    else:
        print(
            unseen_categories.to_string(
                index=False,
                formatters={
                    "validation_share": (
                        lambda value: f"{value:.2%}"
                    ),
                },
            )
        )

    print("\nInterpretation:")
    print(
        "- Total variation distance ranges from 0 to 1; "
        "larger values indicate more categorical shift."
    )
    print(
        "- Standardized mean difference measures validation mean "
        "movement in training-standard-deviation units."
    )
    print(
        "- The test dataset was not loaded or used by this script."
    )


if __name__ == "__main__":
    main()