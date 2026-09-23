"""Train and evaluate random forest model variants."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation_utils import (
    calibration_table,
    capacity_metrics,
    probability_metrics,
)
from modeling_config import (
    PRIMARY_CATEGORICAL_FEATURES,
    PRIMARY_FEATURES,
    PRIMARY_NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXPORT_DIR = PROJECT_ROOT / "data" / "exports"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_CONFIGURATIONS = {
    "random_forest_unweighted": None,
    "random_forest_balanced": "balanced_subsample",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and validation data without test data."""
    train_path = PROCESSED_DIR / "train.parquet"
    validation_path = PROCESSED_DIR / "validation.parquet"

    return (
        pd.read_parquet(train_path),
        pd.read_parquet(validation_path),
    )


def build_pipeline(class_weight: str | None) -> Pipeline:
    """Build a random forest pipeline."""
    preprocessor = build_preprocessor(
        categorical_features=PRIMARY_CATEGORICAL_FEATURES,
        numeric_features=PRIMARY_NUMERIC_FEATURES,
        scale_numeric=False,
    )

    classifier = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight=class_weight,
        n_jobs=-1,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def main() -> None:
    """Train and evaluate random forest variants."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train, validation = load_data()

    X_train = train[PRIMARY_FEATURES]
    y_train = train[TARGET_COLUMN]

    X_validation = validation[PRIMARY_FEATURES]
    y_validation = validation[TARGET_COLUMN]

    metric_rows = []
    capacity_frames = []
    calibration_frames = []

    for model_name, class_weight in MODEL_CONFIGURATIONS.items():
        print(f"Training {model_name}...")

        pipeline = build_pipeline(class_weight)
        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(
            X_validation
        )[:, 1]

        metrics = probability_metrics(
            y_true=y_validation,
            probabilities=probabilities,
            threshold=0.50,
        )
        metrics["model"] = model_name
        metrics["class_weight"] = (
            "none"
            if class_weight is None
            else class_weight
        )
        metric_rows.append(metrics)

        capacity_frames.append(
            capacity_metrics(
                y_true=y_validation,
                probabilities=probabilities,
                model_name=model_name,
            )
        )

        calibration_frames.append(
            calibration_table(
                y_true=y_validation,
                probabilities=probabilities,
                model_name=model_name,
            )
        )

        joblib.dump(
            pipeline,
            MODEL_DIR / f"{model_name}.joblib",
        )

    metric_results = pd.DataFrame(metric_rows)
    capacity_results = pd.concat(
        capacity_frames,
        ignore_index=True,
    )
    calibration_results = pd.concat(
        calibration_frames,
        ignore_index=True,
    )

    metric_results.to_csv(
        EXPORT_DIR / "tree_validation_metrics.csv",
        index=False,
    )
    capacity_results.to_csv(
        EXPORT_DIR / "tree_capacity_metrics.csv",
        index=False,
    )
    calibration_results.to_csv(
        EXPORT_DIR / "tree_calibration_table.csv",
        index=False,
    )

    print("\nTREE VALIDATION METRICS")
    print(
        metric_results[
            [
                "model",
                "class_weight",
                "prevalence",
                "mean_predicted_probability",
                "average_precision",
                "precision_at_threshold",
                "recall_at_threshold",
                "brier_score",
                "calibration_gap",
            ]
        ].to_string(
            index=False,
            formatters={
                "prevalence": lambda value: f"{value:.2%}",
                "mean_predicted_probability": (
                    lambda value: f"{value:.2%}"
                ),
                "average_precision": (
                    lambda value: f"{value:.4f}"
                ),
                "precision_at_threshold": (
                    lambda value: f"{value:.4f}"
                ),
                "recall_at_threshold": (
                    lambda value: f"{value:.4f}"
                ),
                "brier_score": lambda value: f"{value:.4f}",
                "calibration_gap": (
                    lambda value: f"{value:.2%}"
                ),
            },
        )
    )

    print("\nTREE CAPACITY METRICS")
    print(
        capacity_results.to_string(
            index=False,
            formatters={
                "capacity": lambda value: f"{value:.0%}",
                "precision_at_capacity": (
                    lambda value: f"{value:.2%}"
                ),
                "recall_at_capacity": (
                    lambda value: f"{value:.2%}"
                ),
                "lift_at_capacity": (
                    lambda value: f"{value:.2f}"
                ),
                "minimum_selected_probability": (
                    lambda value: f"{value:.4f}"
                ),
            },
        )
    )

    print("\nControls:")
    print("- Preprocessing fitted only on training data.")
    print("- Duration and campaign excluded.")
    print("- Test data not loaded.")


if __name__ == "__main__":
    main()