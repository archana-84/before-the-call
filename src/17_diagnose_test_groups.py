"""Post-test error and group diagnostics for the locked model.

This analysis is descriptive and post-test. It must not be used to
retune the locked model or revise the previously recorded test result.
"""

from pathlib import Path
import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "test.parquet"
)
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_no_economic_sigmoid.joblib"
)
GROUP_EXPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "exports"
    / "post_test_group_metrics.csv"
)
ERROR_EXPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "exports"
    / "post_test_capacity_error_summary.csv"
)
FIGURE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "11_post_test_group_diagnostics.png"
)

TARGET_COLUMN = "y"
CAPACITIES = [0.10, 0.20]

GROUP_VARIABLES = [
    "age_band",
    "job",
    "marital",
    "education",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]


def selection_mask(
    probabilities: np.ndarray,
    capacity: float,
) -> np.ndarray:
    """Select exactly the highest-ranked capacity share."""

    selected_count = math.ceil(
        len(probabilities) * capacity
    )
    ranked_positions = np.argsort(
        -probabilities,
        kind="mergesort",
    )

    mask = np.zeros(
        len(probabilities),
        dtype=bool,
    )
    mask[ranked_positions[:selected_count]] = True

    return mask


def safe_average_precision(
    outcomes: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Calculate AP only when both outcome classes exist."""

    unique_outcomes = np.unique(outcomes)

    if len(unique_outcomes) < 2:
        return np.nan

    return float(
        average_precision_score(
            outcomes,
            probabilities,
        )
    )


def group_metrics(
    data: pd.DataFrame,
    group_variable: str,
) -> list[dict]:
    """Calculate predictive and capacity metrics by group."""

    rows = []

    for group_value, group in data.groupby(
        group_variable,
        dropna=False,
        observed=True,
    ):
        observations = len(group)
        subscribers = int(group["outcome"].sum())

        row = {
            "group_variable": group_variable,
            "group_value": str(group_value),
            "observations": observations,
            "subscribers": subscribers,
            "observed_response_rate": (
                subscribers / observations
            ),
            "mean_predicted_probability": float(
                group["probability"].mean()
            ),
            "calibration_gap": float(
                group["probability"].mean()
                - group["outcome"].mean()
            ),
            "average_precision": safe_average_precision(
                group["outcome"].to_numpy(),
                group["probability"].to_numpy(),
            ),
            "small_group_warning": observations < 100,
        }

        for capacity in CAPACITIES:
            capacity_label = int(capacity * 100)
            selected_column = (
                f"selected_{capacity_label}"
            )

            selected = group[group[selected_column]]
            nonselected = group[
                ~group[selected_column]
            ]

            selected_subscribers = int(
                selected["outcome"].sum()
            )
            selected_non_subscribers = (
                len(selected) - selected_subscribers
            )
            missed_subscribers = int(
                nonselected["outcome"].sum()
            )

            row[
                f"selected_observations_{capacity_label}"
            ] = len(selected)
            row[
                f"selected_subscribers_{capacity_label}"
            ] = selected_subscribers
            row[
                f"selected_non_subscribers_{capacity_label}"
            ] = selected_non_subscribers
            row[
                f"missed_subscribers_{capacity_label}"
            ] = missed_subscribers

            row[
                f"selection_rate_{capacity_label}"
            ] = len(selected) / observations

            row[
                f"precision_at_global_{capacity_label}"
            ] = (
                selected_subscribers / len(selected)
                if len(selected) > 0
                else np.nan
            )

            row[
                f"group_recall_at_global_{capacity_label}"
            ] = (
                selected_subscribers / subscribers
                if subscribers > 0
                else np.nan
            )

        rows.append(row)

    return rows


def capacity_error_summary(
    outcomes: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Describe ranking decisions at each capacity."""

    rows = []
    total_subscribers = int(outcomes.sum())
    prevalence = float(outcomes.mean())

    for capacity in CAPACITIES:
        selected = selection_mask(
            probabilities,
            capacity,
        )

        selected_subscribers = int(
            ((outcomes == 1) & selected).sum()
        )
        false_prioritizations = int(
            ((outcomes == 0) & selected).sum()
        )
        missed_subscribers = int(
            ((outcomes == 1) & ~selected).sum()
        )
        correct_deprioritizations = int(
            ((outcomes == 0) & ~selected).sum()
        )

        selected_count = int(selected.sum())

        rows.append(
            {
                "capacity": capacity,
                "selected_observations": selected_count,
                "selected_subscribers": (
                    selected_subscribers
                ),
                "false_prioritizations": (
                    false_prioritizations
                ),
                "missed_subscribers": missed_subscribers,
                "correct_deprioritizations": (
                    correct_deprioritizations
                ),
                "precision": (
                    selected_subscribers
                    / selected_count
                ),
                "recall": (
                    selected_subscribers
                    / total_subscribers
                ),
                "lift": (
                    (
                        selected_subscribers
                        / selected_count
                    )
                    / prevalence
                ),
            }
        )

    return pd.DataFrame(rows)


def ordered_group(
    group_table: pd.DataFrame,
    variable: str,
    order: list[str],
) -> pd.DataFrame:
    """Return one group table in a specified display order."""

    result = group_table[
        group_table["group_variable"] == variable
    ].copy()

    result["display_order"] = pd.Categorical(
        result["group_value"],
        categories=order,
        ordered=True,
    )

    return result.sort_values("display_order")


def plot_group_comparison(
    axis,
    data: pd.DataFrame,
    title: str,
) -> None:
    """Plot observed and predicted rates for one grouping."""

    positions = np.arange(len(data))

    axis.bar(
        positions,
        data["observed_response_rate"] * 100,
        color="#148277",
        alpha=0.80,
        label="Observed response rate",
    )
    axis.plot(
        positions,
        data["mean_predicted_probability"] * 100,
        color="#34495e",
        marker="o",
        linewidth=2,
        label="Mean predicted probability",
    )

    axis.set_title(title)
    axis.set_ylabel("Rate (%)")
    axis.set_xticks(positions)
    axis.set_xticklabels(
        data["group_value"],
        rotation=35,
        ha="right",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend()


def main() -> None:
    GROUP_EXPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    test = pd.read_parquet(TEST_PATH)

    outcomes = (
        test[TARGET_COLUMN]
        .map({"no": 0, "yes": 1})
        .astype(int)
        .to_numpy()
    )

    features = test.drop(columns=[TARGET_COLUMN])

    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["base_pipeline"]
    calibrator = bundle["calibrator"]

    scores = pipeline.decision_function(features)
    probabilities = calibrator.predict_proba(
        scores.reshape(-1, 1)
    )[:, 1]

    diagnostic_data = test.copy()
    diagnostic_data["outcome"] = outcomes
    diagnostic_data["probability"] = probabilities

    diagnostic_data["age_band"] = pd.cut(
        diagnostic_data["age"],
        bins=[0, 24, 34, 44, 54, 64, np.inf],
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

    for capacity in CAPACITIES:
        capacity_label = int(capacity * 100)
        diagnostic_data[
            f"selected_{capacity_label}"
        ] = selection_mask(
            probabilities,
            capacity,
        )

    group_rows = []

    for variable in GROUP_VARIABLES:
        group_rows.extend(
            group_metrics(
                diagnostic_data,
                variable,
            )
        )

    group_table = pd.DataFrame(group_rows)
    error_table = capacity_error_summary(
        outcomes,
        probabilities,
    )

    group_table.to_csv(
        GROUP_EXPORT_PATH,
        index=False,
    )
    error_table.to_csv(
        ERROR_EXPORT_PATH,
        index=False,
    )

    age_data = ordered_group(
        group_table,
        "age_band",
        [
            "Under 25",
            "25-34",
            "35-44",
            "45-54",
            "55-64",
            "65+",
        ],
    )

    contact_data = ordered_group(
        group_table,
        "contact",
        ["telephone", "cellular"],
    )

    month_data = ordered_group(
        group_table,
        "month",
        [
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
        ],
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(16, 11),
    )

    plot_group_comparison(
        axes[0, 0],
        age_data,
        "Observed versus predicted response by age band",
    )

    plot_group_comparison(
        axes[0, 1],
        contact_data,
        "Observed versus predicted response by contact channel",
    )

    plot_group_comparison(
        axes[1, 0],
        month_data,
        "Observed versus predicted response by campaign month",
    )

    capacity_labels = [
        f"{int(value * 100)}%"
        for value in error_table["capacity"]
    ]
    positions = np.arange(len(error_table))
    width = 0.34

    precision_bars = axes[1, 1].bar(
        positions - width / 2,
        error_table["precision"] * 100,
        width,
        color="#34495e",
        label="Precision",
    )
    recall_bars = axes[1, 1].bar(
        positions + width / 2,
        error_table["recall"] * 100,
        width,
        color="#148277",
        label="Recall",
    )

    axes[1, 1].axhline(
        outcomes.mean() * 100,
        color="#c45a00",
        linestyle="--",
        linewidth=1.8,
        label="Overall response rate",
    )
    axes[1, 1].set_title(
        "Operational results at selected capacities"
    )
    axes[1, 1].set_xlabel("Contact capacity")
    axes[1, 1].set_ylabel("Metric (%)")
    axes[1, 1].set_xticks(positions)
    axes[1, 1].set_xticklabels(capacity_labels)
    axes[1, 1].set_ylim(0, 60)
    axes[1, 1].grid(axis="y", alpha=0.2)
    axes[1, 1].legend()

    axes[1, 1].bar_label(
        precision_bars,
        fmt="%.1f",
        padding=3,
    )
    axes[1, 1].bar_label(
        recall_bars,
        fmt="%.1f",
        padding=3,
    )

    figure.suptitle(
        "Post-test group and ranking-error diagnostics",
        fontsize=17,
        y=0.99,
    )
    figure.text(
        0.5,
        0.953,
        (
            "Descriptive audit only; results are not used "
            "to retune the locked model"
        ),
        ha="center",
        fontsize=10,
        color="#555555",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.93])

    figure.savefig(
        FIGURE_PATH,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    display_errors = error_table.copy()

    for column in [
        "capacity",
        "precision",
        "recall",
    ]:
        display_errors[column] = (
            display_errors[column].map(
                lambda value: f"{value:.2%}"
            )
        )

    display_errors["lift"] = display_errors[
        "lift"
    ].map(lambda value: f"{value:.2f}")

    print("CAPACITY-BASED ERROR SUMMARY")
    print(display_errors.to_string(index=False))

    print("\nCONTACT-CHANNEL GROUP SUMMARY")
    contact_print = group_table[
        group_table["group_variable"] == "contact"
    ].copy()

    print(
        contact_print[
            [
                "group_value",
                "observations",
                "subscribers",
                "observed_response_rate",
                "mean_predicted_probability",
                "calibration_gap",
                "average_precision",
                "selection_rate_20",
                "precision_at_global_20",
                "group_recall_at_global_20",
            ]
        ].to_string(
            index=False,
            formatters={
                "observed_response_rate": (
                    lambda value: f"{value:.2%}"
                ),
                "mean_predicted_probability": (
                    lambda value: f"{value:.2%}"
                ),
                "calibration_gap": (
                    lambda value: f"{value:.2%}"
                ),
                "average_precision": (
                    lambda value: f"{value:.4f}"
                ),
                "selection_rate_20": (
                    lambda value: f"{value:.2%}"
                ),
                "precision_at_global_20": (
                    lambda value: f"{value:.2%}"
                ),
                "group_recall_at_global_20": (
                    lambda value: f"{value:.2%}"
                ),
            },
        )
    )

    print("\nInterpretation controls:")
    print(
        "- A false prioritization is a selected historical "
        "non-subscriber."
    )
    print(
        "- A missed subscriber is a historical subscriber "
        "outside the selected capacity."
    )
    print(
        "- Group differences are descriptive and may reflect "
        "campaign selection, time, and other confounding."
    )
    print(
        "- Small groups are marked in the CSV and should not "
        "support strong conclusions."
    )
    print(
        "- This is not a causal-effect analysis or a formal "
        "fairness certification."
    )
    print(
        "- No model, threshold, or feature policy was changed."
    )

    print(f"\nGroup metrics saved to: {GROUP_EXPORT_PATH}")
    print(f"Error summary saved to: {ERROR_EXPORT_PATH}")
    print(f"Figure saved to: {FIGURE_PATH}")


if __name__ == "__main__":
    main()