"""Evaluate the locked final model on the untouched test partition.

The model and calibration choices were committed before this script
was run. Test results are for final reporting, not further tuning.
"""

from pathlib import Path
import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "test.parquet"
)
FINAL_MODEL_PATH = (
    PROJECT_ROOT / "models" / "final_no_economic_sigmoid.joblib"
)
EXPORT_DIRECTORY = PROJECT_ROOT / "data" / "exports"
FIGURE_DIRECTORY = PROJECT_ROOT / "reports" / "figures"

TARGET_COLUMN = "y"


def probability_metrics(
    variant: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """Calculate test probability and threshold metrics."""

    probabilities = np.clip(probabilities, 1e-10, 1 - 1e-10)
    predictions = (probabilities >= 0.5).astype(int)
    prevalence = float(y_true.mean())
    mean_probability = float(probabilities.mean())

    return {
        "variant": variant,
        "test_observations": len(y_true),
        "test_subscribers": int(y_true.sum()),
        "test_prevalence": prevalence,
        "mean_predicted_probability": mean_probability,
        "calibration_gap": mean_probability - prevalence,
        "average_precision": average_precision_score(
            y_true,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
        ),
        "precision_at_0_5": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall_at_0_5": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
    }


def capacity_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    capacities: np.ndarray,
) -> pd.DataFrame:
    """Calculate results when selecting the highest scores."""

    ranked_positions = np.argsort(
        -probabilities,
        kind="mergesort",
    )
    ranked_outcomes = y_true[ranked_positions]

    observations = len(y_true)
    total_subscribers = int(y_true.sum())
    prevalence = float(y_true.mean())

    rows = []

    for capacity in capacities:
        selected_observations = math.ceil(
            observations * capacity
        )
        selected_outcomes = ranked_outcomes[
            :selected_observations
        ]
        selected_subscribers = int(
            selected_outcomes.sum()
        )

        precision = (
            selected_subscribers / selected_observations
        )
        recall = (
            selected_subscribers / total_subscribers
            if total_subscribers > 0
            else np.nan
        )

        rows.append(
            {
                "capacity": capacity,
                "selected_observations": (
                    selected_observations
                ),
                "selected_subscribers": (
                    selected_subscribers
                ),
                "precision_at_capacity": precision,
                "recall_at_capacity": recall,
                "lift_at_capacity": (
                    precision / prevalence
                    if prevalence > 0
                    else np.nan
                ),
                "minimum_selected_probability": float(
                    probabilities[ranked_positions][
                        selected_observations - 1
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)


def ranked_decile_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Summarize response by descending score decile."""

    ranked_positions = np.argsort(
        -probabilities,
        kind="mergesort",
    )

    ranked = pd.DataFrame(
        {
            "outcome": y_true[ranked_positions],
            "probability": probabilities[
                ranked_positions
            ],
        }
    )

    ranked["score_decile"] = (
        np.floor(
            np.arange(len(ranked)) * 10 / len(ranked)
        ).astype(int)
        + 1
    )

    return (
        ranked.groupby(
            "score_decile",
            as_index=False,
        )
        .agg(
            observations=("outcome", "size"),
            subscribers=("outcome", "sum"),
            mean_predicted_probability=(
                "probability",
                "mean",
            ),
            minimum_probability=(
                "probability",
                "min",
            ),
            maximum_probability=(
                "probability",
                "max",
            ),
        )
        .assign(
            observed_response_rate=lambda data: (
                data["subscribers"]
                / data["observations"]
            )
        )
    )


def main() -> None:
    EXPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("Opening the locked test partition...")
    test = pd.read_parquet(TEST_PATH)

    x_test = test.drop(columns=[TARGET_COLUMN])
    y_test = (
        test[TARGET_COLUMN]
        .map({"no": 0, "yes": 1})
        .astype(int)
        .to_numpy()
    )

    final_bundle = joblib.load(FINAL_MODEL_PATH)
    base_pipeline = final_bundle["base_pipeline"]
    calibrator = final_bundle["calibrator"]

    raw_probabilities = base_pipeline.predict_proba(
        x_test
    )[:, 1]
    test_scores = base_pipeline.decision_function(x_test)
    calibrated_probabilities = calibrator.predict_proba(
        test_scores.reshape(-1, 1)
    )[:, 1]

    metrics = pd.DataFrame(
        [
            probability_metrics(
                "uncalibrated",
                y_test,
                raw_probabilities,
            ),
            probability_metrics(
                "sigmoid_calibrated",
                y_test,
                calibrated_probabilities,
            ),
        ]
    )

    capacities = np.linspace(0.01, 0.50, 50)

    capacity_results = capacity_metrics(
        y_test,
        calibrated_probabilities,
        capacities,
    )

    decile_results = ranked_decile_table(
        y_test,
        calibrated_probabilities,
    )

    calibration_rows = []

    calibration_predictions = {
        "uncalibrated": raw_probabilities,
        "sigmoid_calibrated": calibrated_probabilities,
    }

    for variant, probabilities in (
        calibration_predictions.items()
    ):
        observed_rate, mean_probability = calibration_curve(
            y_test,
            probabilities,
            n_bins=10,
            strategy="quantile",
        )

        for bin_number, (
            bin_probability,
            bin_observed_rate,
        ) in enumerate(
            zip(mean_probability, observed_rate),
            start=1,
        ):
            calibration_rows.append(
                {
                    "variant": variant,
                    "bin": bin_number,
                    "mean_predicted_probability": (
                        bin_probability
                    ),
                    "observed_response_rate": (
                        bin_observed_rate
                    ),
                }
            )

    calibration_results = pd.DataFrame(
        calibration_rows
    )

    metrics_path = (
        EXPORT_DIRECTORY / "final_test_metrics.csv"
    )
    capacity_path = (
        EXPORT_DIRECTORY
        / "final_test_capacity_metrics.csv"
    )
    calibration_path = (
        EXPORT_DIRECTORY
        / "final_test_calibration.csv"
    )
    decile_path = (
        EXPORT_DIRECTORY / "final_test_deciles.csv"
    )

    metrics.to_csv(metrics_path, index=False)
    capacity_results.to_csv(capacity_path, index=False)
    calibration_results.to_csv(
        calibration_path,
        index=False,
    )
    decile_results.to_csv(decile_path, index=False)

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(15, 11),
    )

    prevalence = float(y_test.mean())

    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(
        y_test,
        calibrated_probabilities,
    )
    average_precision = average_precision_score(
        y_test,
        calibrated_probabilities,
    )

    axes[0, 0].plot(
        recall,
        precision,
        color="#148277",
        linewidth=2.3,
        label=f"Final model (AP={average_precision:.3f})",
    )
    axes[0, 0].axhline(
        prevalence,
        color="#c45a00",
        linestyle="--",
        linewidth=1.8,
        label=f"Test prevalence ({prevalence:.1%})",
    )
    axes[0, 0].set_title("Final test precision-recall curve")
    axes[0, 0].set_xlabel("Recall")
    axes[0, 0].set_ylabel("Precision")
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(alpha=0.2)
    axes[0, 0].legend()

    # Capacity performance
    axes[0, 1].plot(
        capacity_results["capacity"] * 100,
        capacity_results["precision_at_capacity"] * 100,
        color="#34495e",
        linewidth=2.3,
        label="Precision at capacity",
    )
    axes[0, 1].plot(
        capacity_results["capacity"] * 100,
        capacity_results["recall_at_capacity"] * 100,
        color="#148277",
        linewidth=2.3,
        label="Recall at capacity",
    )
    axes[0, 1].axhline(
        prevalence * 100,
        color="#c45a00",
        linestyle="--",
        linewidth=1.8,
        label="Overall response rate",
    )
    axes[0, 1].axvline(
        10,
        color="#777777",
        linestyle=":",
    )
    axes[0, 1].axvline(
        20,
        color="#777777",
        linestyle=":",
    )
    axes[0, 1].set_title("Final test capacity performance")
    axes[0, 1].set_xlabel("Observations selected (%)")
    axes[0, 1].set_ylabel("Metric (%)")
    axes[0, 1].set_xlim(1, 50)
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend()

    # Calibration
    for variant, color in [
        ("uncalibrated", "#34495e"),
        ("sigmoid_calibrated", "#148277"),
    ]:
        variant_data = calibration_results[
            calibration_results["variant"] == variant
        ]
        axes[1, 0].plot(
            variant_data["mean_predicted_probability"],
            variant_data["observed_response_rate"],
            marker="o",
            linewidth=2.2,
            color=color,
            label=variant.replace("_", " ").title(),
        )

    axes[1, 0].plot(
        [0, 1],
        [0, 1],
        color="#c45a00",
        linestyle="--",
        linewidth=1.8,
        label="Perfect calibration",
    )
    axes[1, 0].set_title("Final test calibration")
    axes[1, 0].set_xlabel("Mean predicted probability")
    axes[1, 0].set_ylabel("Observed response rate")
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend()

    # Ranked deciles
    axes[1, 1].bar(
        decile_results["score_decile"],
        decile_results["observed_response_rate"] * 100,
        color="#148277",
    )
    axes[1, 1].axhline(
        prevalence * 100,
        color="#c45a00",
        linestyle="--",
        linewidth=1.8,
        label=f"Overall rate ({prevalence:.1%})",
    )
    axes[1, 1].set_title(
        "Observed response by ranked score decile"
    )
    axes[1, 1].set_xlabel(
        "Score decile (1 = highest ranked)"
    )
    axes[1, 1].set_ylabel("Observed response rate (%)")
    axes[1, 1].set_xticks(range(1, 11))
    axes[1, 1].grid(axis="y", alpha=0.2)
    axes[1, 1].legend()

    figure.suptitle(
        "Locked model performance on the untouched test period",
        fontsize=17,
        y=0.99,
    )
    figure.text(
        0.5,
        0.955,
        (
            "Historical Portuguese campaign response; "
            "not an estimate of causal contact uplift"
        ),
        ha="center",
        fontsize=10,
        color="#555555",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.94])

    figure_path = (
        FIGURE_DIRECTORY
        / "09_final_test_evaluation.png"
    )
    figure.savefig(
        figure_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    display_metrics = metrics.copy()

    for column in [
        "test_prevalence",
        "mean_predicted_probability",
        "calibration_gap",
    ]:
        display_metrics[column] = display_metrics[column].map(
            lambda value: f"{value:.2%}"
        )

    for column in [
        "average_precision",
        "brier_score",
        "log_loss",
        "precision_at_0_5",
        "recall_at_0_5",
    ]:
        display_metrics[column] = display_metrics[column].map(
            lambda value: f"{value:.4f}"
        )

    selected_capacity_rows = capacity_results[
        capacity_results["capacity"].round(2).isin(
            [0.10, 0.20]
        )
    ].copy()

    for column in [
        "capacity",
        "precision_at_capacity",
        "recall_at_capacity",
    ]:
        selected_capacity_rows[column] = (
            selected_capacity_rows[column].map(
                lambda value: f"{value:.2%}"
            )
        )

    selected_capacity_rows[
        "lift_at_capacity"
    ] = selected_capacity_rows[
        "lift_at_capacity"
    ].map(lambda value: f"{value:.2f}")

    selected_capacity_rows[
        "minimum_selected_probability"
    ] = selected_capacity_rows[
        "minimum_selected_probability"
    ].map(lambda value: f"{value:.4f}")

    print("\nFINAL TEST PROBABILITY METRICS")
    print(display_metrics.to_string(index=False))

    print("\nFINAL TEST CAPACITY METRICS")
    print(selected_capacity_rows.to_string(index=False))

    print("\nFinal evaluation controls:")
    print("- The model choice was locked before testing.")
    print("- The test partition was not used for fitting.")
    print("- No test result will be used for additional tuning.")
    print("- Capacity ranking is primary; 0.5 is diagnostic only.")
    print("- Results are predictive, not causal.")

    print(f"\nFigure saved to: {figure_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Capacity metrics saved to: {capacity_path}")
    print(f"Calibration saved to: {calibration_path}")
    print(f"Deciles saved to: {decile_path}")


if __name__ == "__main__":
    main()