"""Compare finalist models on validation data only.

This script visualizes:
1. Precision-recall performance
2. Precision under different contact capacities
3. Cumulative recall under different contact capacities
4. Probability calibration

The untouched test set is not loaded.
"""

from pathlib import Path
import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.parquet"
MODEL_DIRECTORY = PROJECT_ROOT / "models"
EXPORT_DIRECTORY = PROJECT_ROOT / "data" / "exports"
FIGURE_DIRECTORY = PROJECT_ROOT / "reports" / "figures"

TARGET_COLUMN = "y"

FINALISTS = {
    "Full scheduled": {
        "path": MODEL_DIRECTORY / "logistic_full_scheduled.joblib",
        "color": "#34495e",
    },
    "No economic": {
        "path": MODEL_DIRECTORY / "logistic_no_economic.joblib",
        "color": "#148277",
    },
}


def capacity_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    capacities: np.ndarray,
) -> pd.DataFrame:
    """Calculate precision and recall at each contact-capacity level."""

    ranked_positions = np.argsort(-probabilities, kind="mergesort")
    ranked_outcomes = y_true[ranked_positions]

    total_positives = int(y_true.sum())
    observations = len(y_true)

    rows = []

    for capacity in capacities:
        selected_count = math.ceil(observations * capacity)
        selected_outcomes = ranked_outcomes[:selected_count]
        selected_positives = int(selected_outcomes.sum())

        precision = selected_positives / selected_count
        recall = (
            selected_positives / total_positives
            if total_positives > 0
            else np.nan
        )

        rows.append(
            {
                "capacity": capacity,
                "selected_observations": selected_count,
                "selected_subscribers": selected_positives,
                "precision": precision,
                "recall": recall,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    EXPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    validation = pd.read_parquet(VALIDATION_PATH)

    y_validation = (
        validation[TARGET_COLUMN]
        .map({"no": 0, "yes": 1})
        .astype(int)
        .to_numpy()
    )

    x_validation = validation.drop(columns=[TARGET_COLUMN])
    prevalence = y_validation.mean()

    capacities = np.linspace(0.01, 0.50, 50)

    prediction_results = {}
    capacity_results = []
    calibration_results = []

    for model_name, model_details in FINALISTS.items():
        print(f"Loading {model_name}...")

        model_bundle = joblib.load(model_details["path"])
        model = model_bundle["pipeline"]
        probabilities = model.predict_proba(x_validation)[:, 1]

        average_precision = average_precision_score(
            y_validation,
            probabilities,
        )

        precision, recall, _ = precision_recall_curve(
            y_validation,
            probabilities,
        )

        model_capacity = capacity_curve(
            y_validation,
            probabilities,
            capacities,
        )
        model_capacity.insert(0, "model", model_name)
        capacity_results.append(model_capacity)

        observed_rate, mean_probability = calibration_curve(
            y_validation,
            probabilities,
            n_bins=10,
            strategy="quantile",
        )

        model_calibration = pd.DataFrame(
            {
                "model": model_name,
                "mean_predicted_probability": mean_probability,
                "observed_response_rate": observed_rate,
            }
        )
        calibration_results.append(model_calibration)

        prediction_results[model_name] = {
            "probabilities": probabilities,
            "average_precision": average_precision,
            "precision": precision,
            "recall": recall,
            "color": model_details["color"],
        }

    capacity_table = pd.concat(capacity_results, ignore_index=True)
    calibration_table = pd.concat(calibration_results, ignore_index=True)

    capacity_table.to_csv(
        EXPORT_DIRECTORY / "validation_capacity_curves.csv",
        index=False,
    )
    calibration_table.to_csv(
        EXPORT_DIRECTORY / "validation_calibration_points.csv",
        index=False,
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(15, 11),
    )

    # Panel 1: precision-recall curve
    precision_recall_axis = axes[0, 0]

    for model_name, result in prediction_results.items():
        precision_recall_axis.plot(
            result["recall"],
            result["precision"],
            linewidth=2.3,
            color=result["color"],
            label=(
                f"{model_name} "
                f"(AP={result['average_precision']:.3f})"
            ),
        )

    precision_recall_axis.axhline(
        prevalence,
        color="#c45a00",
        linestyle="--",
        linewidth=1.8,
        label=f"Validation prevalence ({prevalence:.1%})",
    )
    precision_recall_axis.set_title("Precision-recall performance")
    precision_recall_axis.set_xlabel("Recall")
    precision_recall_axis.set_ylabel("Precision")
    precision_recall_axis.set_xlim(0, 1)
    precision_recall_axis.set_ylim(0, 1)
    precision_recall_axis.grid(alpha=0.2)
    precision_recall_axis.legend()

    # Panel 2: precision by capacity
    precision_capacity_axis = axes[0, 1]

    for model_name, result in prediction_results.items():
        model_capacity = capacity_table[
            capacity_table["model"] == model_name
        ]

        precision_capacity_axis.plot(
            model_capacity["capacity"] * 100,
            model_capacity["precision"] * 100,
            linewidth=2.3,
            color=result["color"],
            label=model_name,
        )

    precision_capacity_axis.axhline(
        prevalence * 100,
        color="#c45a00",
        linestyle="--",
        linewidth=1.8,
        label=f"Random-ranking reference ({prevalence:.1%})",
    )
    precision_capacity_axis.axvline(
        10,
        color="#777777",
        linestyle=":",
        linewidth=1.2,
    )
    precision_capacity_axis.axvline(
        20,
        color="#777777",
        linestyle=":",
        linewidth=1.2,
    )
    precision_capacity_axis.set_title("Precision under limited capacity")
    precision_capacity_axis.set_xlabel("Observations selected (%)")
    precision_capacity_axis.set_ylabel("Subscribers among selected (%)")
    precision_capacity_axis.set_xlim(1, 50)
    precision_capacity_axis.grid(alpha=0.2)
    precision_capacity_axis.legend()

    # Panel 3: cumulative recall by capacity
    recall_capacity_axis = axes[1, 0]

    for model_name, result in prediction_results.items():
        model_capacity = capacity_table[
            capacity_table["model"] == model_name
        ]

        recall_capacity_axis.plot(
            model_capacity["capacity"] * 100,
            model_capacity["recall"] * 100,
            linewidth=2.3,
            color=result["color"],
            label=model_name,
        )

    recall_capacity_axis.plot(
        capacities * 100,
        capacities * 100,
        color="#c45a00",
        linestyle="--",
        linewidth=1.8,
        label="Random-ranking expectation",
    )
    recall_capacity_axis.axvline(
        10,
        color="#777777",
        linestyle=":",
        linewidth=1.2,
    )
    recall_capacity_axis.axvline(
        20,
        color="#777777",
        linestyle=":",
        linewidth=1.2,
    )
    recall_capacity_axis.set_title("Subscribers captured by capacity")
    recall_capacity_axis.set_xlabel("Observations selected (%)")
    recall_capacity_axis.set_ylabel("All subscribers captured (%)")
    recall_capacity_axis.set_xlim(1, 50)
    recall_capacity_axis.set_ylim(0, 100)
    recall_capacity_axis.grid(alpha=0.2)
    recall_capacity_axis.legend()

    # Panel 4: calibration curve
    calibration_axis = axes[1, 1]

    maximum_calibration_value = max(
        calibration_table["mean_predicted_probability"].max(),
        calibration_table["observed_response_rate"].max(),
    )
    calibration_limit = min(
        1.0,
        max(0.15, maximum_calibration_value * 1.10),
    )

    calibration_axis.plot(
        [0, calibration_limit],
        [0, calibration_limit],
        color="#c45a00",
        linestyle="--",
        linewidth=1.8,
        label="Perfect calibration",
    )

    for model_name, result in prediction_results.items():
        model_calibration = calibration_table[
            calibration_table["model"] == model_name
        ]

        calibration_axis.plot(
            model_calibration["mean_predicted_probability"],
            model_calibration["observed_response_rate"],
            marker="o",
            linewidth=2.3,
            color=result["color"],
            label=model_name,
        )

    calibration_axis.set_title("Probability calibration by score decile")
    calibration_axis.set_xlabel("Mean predicted probability")
    calibration_axis.set_ylabel("Observed response rate")
    calibration_axis.set_xlim(0, calibration_limit)
    calibration_axis.set_ylim(0, calibration_limit)
    calibration_axis.grid(alpha=0.2)
    calibration_axis.legend()

    figure.suptitle(
        "Validation performance of finalist feature policies",
        fontsize=17,
        y=0.99,
    )
    figure.text(
        0.5,
        0.955,
        (
            "Historical response prediction only; "
            "results do not estimate causal campaign uplift"
        ),
        ha="center",
        fontsize=10,
        color="#555555",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.94])

    figure_path = (
        FIGURE_DIRECTORY / "07_validation_model_comparison.png"
    )
    figure.savefig(
        figure_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("\nVALIDATION FINALIST SUMMARY")
    for model_name, result in prediction_results.items():
        model_capacity = capacity_table[
            capacity_table["model"] == model_name
        ]

        ten_percent = model_capacity.iloc[
            (model_capacity["capacity"] - 0.10).abs().argmin()
        ]
        twenty_percent = model_capacity.iloc[
            (model_capacity["capacity"] - 0.20).abs().argmin()
        ]

        print(f"\n{model_name}")
        print(
            f"  Average precision: "
            f"{result['average_precision']:.4f}"
        )
        print(
            f"  Precision at 10% capacity: "
            f"{ten_percent['precision']:.2%}"
        )
        print(
            f"  Recall at 10% capacity: "
            f"{ten_percent['recall']:.2%}"
        )
        print(
            f"  Precision at 20% capacity: "
            f"{twenty_percent['precision']:.2%}"
        )
        print(
            f"  Recall at 20% capacity: "
            f"{twenty_percent['recall']:.2%}"
        )

    print(f"\nFigure saved to: {figure_path}")
    print(
        "Capacity curves saved to: "
        f"{EXPORT_DIRECTORY / 'validation_capacity_curves.csv'}"
    )
    print(
        "Calibration points saved to: "
        f"{EXPORT_DIRECTORY / 'validation_calibration_points.csv'}"
    )

    print("\nControls:")
    print("- Only validation data was evaluated.")
    print("- The test dataset was not loaded.")
    print("- No model was refitted by this script.")


if __name__ == "__main__":
    main()