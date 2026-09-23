"""Evaluate a training-prior baseline on validation data."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modeling_config import PRIMARY_FEATURES, TARGET_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXPORT_DIR = PROJECT_ROOT / "data" / "exports"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
MODEL_DIR = PROJECT_ROOT / "models"

CAPACITY_LEVELS = [0.10, 0.20]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and validation data without reading test data."""
    train_path = PROCESSED_DIR / "train.parquet"
    validation_path = PROCESSED_DIR / "validation.parquet"

    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError(
            "Training or validation data is missing. "
            "Run src/05_create_chronological_splits.py first."
        )

    return (
        pd.read_parquet(train_path),
        pd.read_parquet(validation_path),
    )


def build_capacity_reference(
    validation_size: int,
    validation_prevalence: float,
) -> pd.DataFrame:
    """Calculate expected random-ranking performance by capacity."""
    rows = []

    for capacity in CAPACITY_LEVELS:
        selected_count = math.ceil(validation_size * capacity)
        selected_fraction = selected_count / validation_size

        rows.append(
            {
                "baseline": "random ranking expectation",
                "capacity": capacity,
                "validation_observations": validation_size,
                "selected_observations": selected_count,
                "expected_selected_subscribers": (
                    selected_count * validation_prevalence
                ),
                "expected_precision_at_capacity": (
                    validation_prevalence
                ),
                "expected_recall_at_capacity": selected_fraction,
                "expected_lift_at_capacity": 1.0,
            }
        )

    return pd.DataFrame(rows)


def plot_prevalence_comparison(
    training_prevalence: float,
    validation_prevalence: float,
    mean_prediction: float,
) -> None:
    """Visualize the effect of prevalence shift on the baseline."""
    labels = [
        "Training\nobserved",
        "Validation\nobserved",
        "Baseline\nmean prediction",
    ]
    values = [
        training_prevalence,
        validation_prevalence,
        mean_prediction,
    ]
    colors = ["#475569", "#0F766E", "#B45309"]

    figure, axis = plt.subplots(figsize=(9, 5.5))

    bars = axis.bar(
        labels,
        [value * 100 for value in values],
        color=colors,
        width=0.6,
    )

    axis.bar_label(
        bars,
        labels=[f"{value:.2%}" for value in values],
        padding=5,
    )
    axis.set_title("Training-prior baseline under prevalence shift")
    axis.set_ylabel("Positive outcome rate or probability (%)")
    axis.set_ylim(
        0,
        max(values) * 100 * 1.25,
    )
    axis.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "06_baseline_prevalence_shift.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    """Fit and evaluate the no-skill prior baseline."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train, validation = load_data()

    X_train = train[PRIMARY_FEATURES]
    y_train = train[TARGET_COLUMN]

    X_validation = validation[PRIMARY_FEATURES]
    y_validation = validation[TARGET_COLUMN]

    baseline = DummyClassifier(strategy="prior")
    baseline.fit(X_train, y_train)

    validation_probability = baseline.predict_proba(
        X_validation
    )[:, 1]
    validation_prediction = baseline.predict(X_validation)

    training_prevalence = float(y_train.mean())
    validation_prevalence = float(y_validation.mean())
    mean_prediction = float(validation_probability.mean())

    metrics = pd.DataFrame(
        [
            {
                "model": "dummy_training_prior",
                "training_observations": len(train),
                "validation_observations": len(validation),
                "training_prevalence": training_prevalence,
                "validation_prevalence": validation_prevalence,
                "mean_predicted_probability": mean_prediction,
                "average_precision": average_precision_score(
                    y_validation,
                    validation_probability,
                ),
                "precision_at_0_5": precision_score(
                    y_validation,
                    validation_prediction,
                    zero_division=0,
                ),
                "recall_at_0_5": recall_score(
                    y_validation,
                    validation_prediction,
                    zero_division=0,
                ),
                "brier_score": brier_score_loss(
                    y_validation,
                    validation_probability,
                ),
                "calibration_gap": (
                    mean_prediction - validation_prevalence
                ),
            }
        ]
    )

    capacity_reference = build_capacity_reference(
        validation_size=len(validation),
        validation_prevalence=validation_prevalence,
    )

    metrics.to_csv(
        EXPORT_DIR / "baseline_validation_metrics.csv",
        index=False,
    )
    capacity_reference.to_csv(
        EXPORT_DIR / "baseline_capacity_reference.csv",
        index=False,
    )

    joblib.dump(
        {
            "model": baseline,
            "features": PRIMARY_FEATURES,
            "training_prevalence": training_prevalence,
        },
        MODEL_DIR / "baseline_prior.joblib",
    )

    plot_prevalence_comparison(
        training_prevalence,
        validation_prevalence,
        mean_prediction,
    )

    print("\nBASELINE VALIDATION METRICS")
    print(
        metrics.to_string(
            index=False,
            formatters={
                "training_prevalence": (
                    lambda value: f"{value:.2%}"
                ),
                "validation_prevalence": (
                    lambda value: f"{value:.2%}"
                ),
                "mean_predicted_probability": (
                    lambda value: f"{value:.2%}"
                ),
                "average_precision": (
                    lambda value: f"{value:.4f}"
                ),
                "precision_at_0_5": (
                    lambda value: f"{value:.4f}"
                ),
                "recall_at_0_5": (
                    lambda value: f"{value:.4f}"
                ),
                "brier_score": (
                    lambda value: f"{value:.4f}"
                ),
                "calibration_gap": (
                    lambda value: f"{value:.2%}"
                ),
            },
        )
    )

    print("\nRANDOM-RANKING CAPACITY REFERENCE")
    print(
        capacity_reference.to_string(
            index=False,
            formatters={
                "capacity": lambda value: f"{value:.0%}",
                "expected_selected_subscribers": (
                    lambda value: f"{value:.1f}"
                ),
                "expected_precision_at_capacity": (
                    lambda value: f"{value:.2%}"
                ),
                "expected_recall_at_capacity": (
                    lambda value: f"{value:.2%}"
                ),
                "expected_lift_at_capacity": (
                    lambda value: f"{value:.2f}"
                ),
            },
        )
    )

    print("\nInterpretation:")
    print(
        "- The baseline assigns the training prevalence to every "
        "validation observation."
    )
    print(
        "- Constant scores cannot rank observations meaningfully."
    )
    print(
        "- Capacity results are theoretical random-selection "
        "expectations because every baseline score is tied."
    )
    print("- Test data was not loaded or evaluated.")


if __name__ == "__main__":
    main()