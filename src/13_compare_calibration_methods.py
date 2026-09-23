"""Compare temporal probability-calibration methods.

The base models remain fixed. The validation period is divided
chronologically:

- Earlier 60%: calibration fitting
- Later 40%: calibration assessment

The final test dataset is not loaded.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.parquet"
MODEL_DIRECTORY = PROJECT_ROOT / "models"
EXPORT_DIRECTORY = PROJECT_ROOT / "data" / "exports"
FIGURE_DIRECTORY = PROJECT_ROOT / "reports" / "figures"

TARGET_COLUMN = "y"
CALIBRATION_SHARE = 0.60

FINALISTS = {
    "full_scheduled": {
        "label": "Full scheduled",
        "path": MODEL_DIRECTORY / "logistic_full_scheduled.joblib",
        "color": "#34495e",
    },
    "no_economic": {
        "label": "No economic",
        "path": MODEL_DIRECTORY / "logistic_no_economic.joblib",
        "color": "#148277",
    },
}

METHOD_STYLES = {
    "uncalibrated": "-",
    "sigmoid": "--",
    "isotonic": ":",
}


def target_to_binary(data: pd.DataFrame) -> np.ndarray:
    """Convert the yes/no target to 1/0."""

    return (
        data[TARGET_COLUMN]
        .map({"no": 0, "yes": 1})
        .astype(int)
        .to_numpy()
    )


def evaluate_probabilities(
    model_name: str,
    method: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """Return discrimination and calibration metrics."""

    probabilities = np.clip(probabilities, 1e-10, 1 - 1e-10)
    prevalence = float(y_true.mean())
    mean_probability = float(probabilities.mean())

    return {
        "model": model_name,
        "calibration_method": method,
        "assessment_observations": len(y_true),
        "assessment_prevalence": prevalence,
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
    }


def main() -> None:
    EXPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    validation = pd.read_parquet(VALIDATION_PATH)

    split_position = int(
        len(validation) * CALIBRATION_SHARE
    )

    calibration_data = validation.iloc[:split_position].copy()
    assessment_data = validation.iloc[split_position:].copy()

    x_calibration = calibration_data.drop(columns=[TARGET_COLUMN])
    y_calibration = target_to_binary(calibration_data)

    x_assessment = assessment_data.drop(columns=[TARGET_COLUMN])
    y_assessment = target_to_binary(assessment_data)

    print("TEMPORAL CALIBRATION SPLIT")
    print(
        f"Calibration fitting rows: "
        f"{len(calibration_data):,} "
        f"({y_calibration.mean():.2%} positive)"
    )
    print(
        f"Later assessment rows: "
        f"{len(assessment_data):,} "
        f"({y_assessment.mean():.2%} positive)"
    )

    metric_rows = []
    curve_rows = []

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, 6),
        sharex=True,
        sharey=True,
    )

    for axis, (model_name, model_details) in zip(
        axes,
        FINALISTS.items(),
    ):
        print(f"\nEvaluating {model_details['label']}...")

        model_bundle = joblib.load(model_details["path"])
        pipeline = model_bundle["pipeline"]

        calibration_probabilities = pipeline.predict_proba(
            x_calibration
        )[:, 1]
        assessment_probabilities = pipeline.predict_proba(
            x_assessment
        )[:, 1]

        calibration_scores = pipeline.decision_function(
            x_calibration
        )
        assessment_scores = pipeline.decision_function(
            x_assessment
        )

        # Sigmoid/Platt-style calibration using the base model's
        # decision score as a single input.
        sigmoid = LogisticRegression(
            C=1_000_000,
            solver="lbfgs",
            max_iter=5000,
        )
        sigmoid.fit(
            calibration_scores.reshape(-1, 1),
            y_calibration,
        )
        sigmoid_probabilities = sigmoid.predict_proba(
            assessment_scores.reshape(-1, 1)
        )[:, 1]

        # Isotonic calibration is flexible but can overfit when
        # calibration data is limited.
        isotonic = IsotonicRegression(
            out_of_bounds="clip",
        )
        isotonic.fit(
            calibration_probabilities,
            y_calibration,
        )
        isotonic_probabilities = isotonic.predict(
            assessment_probabilities
        )

        method_predictions = {
            "uncalibrated": assessment_probabilities,
            "sigmoid": sigmoid_probabilities,
            "isotonic": isotonic_probabilities,
        }

        for method, probabilities in method_predictions.items():
            metric_rows.append(
                evaluate_probabilities(
                    model_name=model_name,
                    method=method,
                    y_true=y_assessment,
                    probabilities=probabilities,
                )
            )

            observed_rate, mean_probability = calibration_curve(
                y_assessment,
                probabilities,
                n_bins=8,
                strategy="quantile",
            )

            for bin_number, (
                bin_probability,
                bin_observed_rate,
            ) in enumerate(
                zip(mean_probability, observed_rate),
                start=1,
            ):
                curve_rows.append(
                    {
                        "model": model_name,
                        "calibration_method": method,
                        "bin": bin_number,
                        "mean_predicted_probability": (
                            bin_probability
                        ),
                        "observed_response_rate": (
                            bin_observed_rate
                        ),
                    }
                )

            axis.plot(
                mean_probability,
                observed_rate,
                marker="o",
                linestyle=METHOD_STYLES[method],
                linewidth=2,
                label=method.capitalize(),
            )

        axis.plot(
            [0, 1],
            [0, 1],
            color="#c45a00",
            linestyle="--",
            linewidth=1.5,
            label="Perfect calibration",
        )
        axis.set_title(model_details["label"])
        axis.set_xlabel("Mean predicted probability")
        axis.set_xlim(0, 0.27)
        axis.set_ylim(0, 0.27)
        axis.grid(alpha=0.2)
        axis.legend()
        
    axes[0].set_ylabel("Observed response rate")

    figure.suptitle(
        "Temporal calibration assessment on later validation rows",
        fontsize=16,
    )
    figure.text(
        0.5,
        0.925,
        (
            "Calibration mappings fitted on the earlier 60% "
            "of validation; assessed on the later 40%"
        ),
        ha="center",
        fontsize=10,
        color="#555555",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.90])

    figure_path = (
        FIGURE_DIRECTORY
        / "08_temporal_calibration_comparison.png"
    )
    figure.savefig(
        figure_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    metrics = pd.DataFrame(metric_rows)
    curves = pd.DataFrame(curve_rows)

    metrics_path = (
        EXPORT_DIRECTORY
        / "temporal_calibration_metrics.csv"
    )
    curves_path = (
        EXPORT_DIRECTORY
        / "temporal_calibration_curves.csv"
    )

    metrics.to_csv(metrics_path, index=False)
    curves.to_csv(curves_path, index=False)

    display_metrics = metrics.copy()

    for column in [
        "assessment_prevalence",
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
    ]:
        display_metrics[column] = display_metrics[column].map(
            lambda value: f"{value:.4f}"
        )

    print("\nLATER-VALIDATION CALIBRATION RESULTS")
    print(display_metrics.to_string(index=False))

    print("\nInterpretation controls:")
    print("- Lower Brier score and log loss are better.")
    print("- Calibration gap closer to zero is better.")
    print("- Calibration was not fitted on assessment rows.")
    print("- Base models were not retrained.")
    print("- Test data was not loaded.")

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Calibration curves saved to: {curves_path}")
    print(f"Figure saved to: {figure_path}")


if __name__ == "__main__":
    main()