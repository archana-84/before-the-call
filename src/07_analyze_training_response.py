"""Analyze response patterns using training data only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"
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

UNKNOWN_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
]

ECONOMIC_FEATURES = [
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]


def load_training_data() -> pd.DataFrame:
    """Load training data without accessing validation or test data."""
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            "Training data is missing. "
            "Run src/05_create_chronological_splits.py first."
        )

    return pd.read_parquet(TRAIN_PATH)


def summarize_group(
    data: pd.DataFrame,
    grouping: pd.Series,
    feature_name: str,
) -> pd.DataFrame:
    """Summarize response outcomes for one grouped feature."""
    working = pd.DataFrame(
        {
            "category": grouping,
            "target": data["target"],
        }
    )

    summary = (
        working.groupby(
            "category",
            observed=True,
            dropna=False,
            sort=False,
        )
        .agg(
            observations=("target", "size"),
            subscribers=("target", "sum"),
        )
        .reset_index()
    )

    summary.insert(0, "feature", feature_name)
    summary["category"] = summary["category"].astype("string")
    summary["response_rate"] = (
        summary["subscribers"] / summary["observations"]
    )

    return summary


def build_categorical_response(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Profile training response rates for categorical features."""
    profiles = []

    for feature in CATEGORICAL_FEATURES:
        profiles.append(
            summarize_group(
                data,
                data[feature],
                feature,
            )
        )

    return pd.concat(profiles, ignore_index=True)


def build_numeric_response(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create interpretable training-only numeric response bins."""
    profiles = []

    age_bins = pd.cut(
        data["age"],
        bins=[0, 24, 34, 44, 54, 64, float("inf")],
        labels=[
            "Under 25",
            "25-34",
            "35-44",
            "45-54",
            "55-64",
            "65+",
        ],
        include_lowest=True,
    )
    profiles.append(
        summarize_group(data, age_bins, "age_band")
    )

    pdays_group = pd.Series(
        "999 sentinel",
        index=data.index,
        dtype="string",
    )
    pdays_group.loc[data["pdays"].between(0, 7)] = "0-7"
    pdays_group.loc[data["pdays"].between(8, 30)] = "8-30"
    pdays_group.loc[data["pdays"].between(31, 90)] = "31-90"
    pdays_group.loc[data["pdays"].between(91, 998)] = "91-998"

    profiles.append(
        summarize_group(data, pdays_group, "pdays_group")
    )

    previous_group = pd.cut(
        data["previous"],
        bins=[-1, 0, 1, 2, float("inf")],
        labels=["0", "1", "2", "3+"],
    )
    profiles.append(
        summarize_group(
            data,
            previous_group,
            "previous_group",
        )
    )

    for feature in ECONOMIC_FEATURES:
        quantile_group = pd.qcut(
            data[feature],
            q=5,
            duplicates="drop",
        )
        profiles.append(
            summarize_group(
                data,
                quantile_group,
                feature,
            )
        )

    return pd.concat(profiles, ignore_index=True)


def build_unknown_patterns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Profile overlapping unknown-value patterns."""
    unknown_flags = data[UNKNOWN_FEATURES].eq("unknown")

    patterns = unknown_flags.apply(
        lambda row: (
            ", ".join(row.index[row].tolist())
            if row.any()
            else "none"
        ),
        axis=1,
    )

    return summarize_group(
        data,
        patterns,
        "unknown_pattern",
    ).sort_values(
        "observations",
        ascending=False,
    )


def build_rare_categories(
    categorical_response: pd.DataFrame,
) -> pd.DataFrame:
    """Identify categories with fewer than 100 training rows."""
    return (
        categorical_response.loc[
            categorical_response["observations"] < 100
        ]
        .sort_values(
            ["observations", "feature"],
            ascending=[True, True],
        )
        .reset_index(drop=True)
    )


def add_response_labels(axis, bars, data: pd.DataFrame) -> None:
    """Add rate and observation-count labels to bars."""
    labels = [
        f"{rate:.1%} | n={count:,}"
        for rate, count in zip(
            data["response_rate"],
            data["observations"],
            strict=True,
        )
    ]
    axis.bar_label(bars, labels=labels, padding=4)


def plot_training_patterns(
    categorical_response: pd.DataFrame,
    numeric_response: pd.DataFrame,
    overall_rate: float,
) -> None:
    """Plot previous outcome and age-band response patterns."""
    poutcome_order = ["nonexistent", "failure", "success"]
    age_order = [
        "Under 25",
        "25-34",
        "35-44",
        "45-54",
        "55-64",
        "65+",
    ]

    poutcome_data = categorical_response.loc[
        categorical_response["feature"] == "poutcome"
    ].copy()
    poutcome_data["category"] = pd.Categorical(
        poutcome_data["category"],
        categories=poutcome_order,
        ordered=True,
    )
    poutcome_data = poutcome_data.sort_values("category")

    age_data = numeric_response.loc[
        numeric_response["feature"] == "age_band"
    ].copy()
    age_data["category"] = pd.Categorical(
        age_data["category"],
        categories=age_order,
        ordered=True,
    )
    age_data = age_data.sort_values("category")

    figure, (history_axis, age_axis) = plt.subplots(
        1,
        2,
        figsize=(15, 6),
    )

    history_bars = history_axis.barh(
        poutcome_data["category"].astype("string"),
        poutcome_data["response_rate"] * 100,
        color="#475569",
    )
    history_axis.axvline(
        overall_rate * 100,
        color="#B45309",
        linestyle="--",
        label=f"Training rate: {overall_rate:.1%}",
    )
    history_axis.set_title("Previous campaign outcome")
    history_axis.set_xlabel("Historical response rate (%)")
    history_axis.set_ylabel("Previous outcome")
    history_axis.legend()
    history_axis.spines[["top", "right"]].set_visible(False)
    add_response_labels(history_axis, history_bars, poutcome_data)

    age_bars = age_axis.barh(
        age_data["category"].astype("string"),
        age_data["response_rate"] * 100,
        color="#0F766E",
    )
    age_axis.axvline(
        overall_rate * 100,
        color="#B45309",
        linestyle="--",
        label=f"Training rate: {overall_rate:.1%}",
    )
    age_axis.set_title("Age band")
    age_axis.set_xlabel("Historical response rate (%)")
    age_axis.set_ylabel("Age band")
    age_axis.legend()
    age_axis.spines[["top", "right"]].set_visible(False)
    add_response_labels(age_axis, age_bars, age_data)

    figure.suptitle(
        "Training-only historical response patterns",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "05_training_response_patterns.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    """Create training-only response analyses."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    train = load_training_data()

    categorical_response = build_categorical_response(train)
    numeric_response = build_numeric_response(train)
    unknown_patterns = build_unknown_patterns(train)
    rare_categories = build_rare_categories(
        categorical_response
    )

    categorical_response.to_csv(
        EXPORT_DIR / "training_categorical_response.csv",
        index=False,
    )
    numeric_response.to_csv(
        EXPORT_DIR / "training_numeric_response.csv",
        index=False,
    )
    unknown_patterns.to_csv(
        EXPORT_DIR / "training_unknown_patterns.csv",
        index=False,
    )
    rare_categories.to_csv(
        EXPORT_DIR / "training_rare_categories.csv",
        index=False,
    )

    overall_rate = float(train["target"].mean())

    plot_training_patterns(
        categorical_response,
        numeric_response,
        overall_rate,
    )

    print("\nTRAINING RESPONSE RATE")
    print(f"{overall_rate:.2%}")

    print("\nPREVIOUS CAMPAIGN OUTCOME")
    print(
        categorical_response.loc[
            categorical_response["feature"] == "poutcome"
        ].to_string(
            index=False,
            formatters={
                "response_rate": lambda value: f"{value:.2%}",
            },
        )
    )

    print("\nAGE BANDS")
    print(
        numeric_response.loc[
            numeric_response["feature"] == "age_band"
        ].to_string(
            index=False,
            formatters={
                "response_rate": lambda value: f"{value:.2%}",
            },
        )
    )

    print("\nUNKNOWN-VALUE PATTERNS")
    print(
        unknown_patterns.head(10).to_string(
            index=False,
            formatters={
                "response_rate": lambda value: f"{value:.2%}",
            },
        )
    )

    print("\nRARE TRAINING CATEGORIES")
    if rare_categories.empty:
        print("None")
    else:
        print(
            rare_categories.to_string(
                index=False,
                formatters={
                    "response_rate": lambda value: f"{value:.2%}",
                },
            )
        )

    print("\nControl: validation and test data were not loaded.")


if __name__ == "__main__":
    main()