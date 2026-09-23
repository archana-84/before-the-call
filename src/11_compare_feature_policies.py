"""Compare logistic regression feature policies on validation data."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation_utils import capacity_metrics, probability_metrics
from modeling_config import (
    NO_ECONOMIC_CATEGORICAL_FEATURES,
    NO_ECONOMIC_NUMERIC_FEATURES,
    PRIMARY_CATEGORICAL_FEATURES,
    PRIMARY_NUMERIC_FEATURES,
    STRICT_CATEGORICAL_FEATURES,
    STRICT_NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXPORT_DIR = PROJECT_ROOT / "data" / "exports"
MODEL_DIR = PROJECT_ROOT / "models"

FEATURE_POLICIES = {
    "full_scheduled": {
        "categorical": PRIMARY_CATEGORICAL_FEATURES,
        "numeric": PRIMARY_NUMERIC_FEATURES,
    },
    "no_economic": {
        "categorical": NO_ECONOMIC_CATEGORICAL_FEATURES,
        "numeric": NO_ECONOMIC_NUMERIC_FEATURES,
    },
    "no_scheduling": {
        "categorical": STRICT_CATEGORICAL_FEATURES,
        "numeric": STRICT_NUMERIC_FEATURES,
    },
    "core_only": {
        "categorical": STRICT_CATEGORICAL_FEATURES,
        "numeric": NO_ECONOMIC_NUMERIC_FEATURES,
    },
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and validation without test data."""
    return (
        pd.read_parquet(PROCESSED_DIR / "train.parquet"),
        pd.read_parquet(PROCESSED_DIR / "validation.parquet"),
    )


def build_pipeline(
    categorical_features: list[str],
    numeric_features: list[str],
) -> Pipeline:
    """Build an unweighted logistic pipeline."""
    preprocessor = build_preprocessor(
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        scale_numeric=True,
    )

    classifier = LogisticRegression(
        class_weight=None,
        C=1.0,
        max_iter=5000,
        solver="liblinear",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def main() -> None:
    """Fit and compare feature-policy models."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train, validation = load_data()

    y_train = train[TARGET_COLUMN]
    y_validation = validation[TARGET_COLUMN]

    metric_rows = []
    capacity_frames = []

    for policy_name, feature_policy in FEATURE_POLICIES.items():
        categorical = feature_policy["categorical"]
        numeric = feature_policy["numeric"]
        features = categorical + numeric

        print(f"Training policy: {policy_name}...")

        pipeline = build_pipeline(
            categorical_features=categorical,
            numeric_features=numeric,
        )

        pipeline.fit(
            train[features],
            y_train,
        )

        probabilities = pipeline.predict_proba(
            validation[features]
        )[:, 1]

        metrics = probability_metrics(
            y_true=y_validation,
            probabilities=probabilities,
            threshold=0.50,
        )
        metrics["feature_policy"] = policy_name
        metrics["source_feature_count"] = len(features)
        metric_rows.append(metrics)

        capacity_frames.append(
            capacity_metrics(
                y_true=y_validation,
                probabilities=probabilities,
                model_name=policy_name,
            )
        )

        joblib.dump(
            {
                "pipeline": pipeline,
                "feature_policy": policy_name,
                "features": features,
                "categorical_features": categorical,
                "numeric_features": numeric,
            },
            MODEL_DIR / f"logistic_{policy_name}.joblib",
        )

    metric_results = pd.DataFrame(metric_rows)
    capacity_results = pd.concat(
        capacity_frames,
        ignore_index=True,
    )

    metric_results.to_csv(
        EXPORT_DIR / "logistic_feature_policy_metrics.csv",
        index=False,
    )
    capacity_results.to_csv(
        EXPORT_DIR / "logistic_feature_policy_capacity.csv",
        index=False,
    )

    print("\nFEATURE-POLICY VALIDATION METRICS")
    print(
        metric_results[
            [
                "feature_policy",
                "source_feature_count",
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

    print("\nFEATURE-POLICY CAPACITY METRICS")
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
    print("- All policies use the same training and validation rows.")
    print("- No test data was loaded.")
    print("- Duration and campaign remain excluded.")


if __name__ == "__main__":
    main()