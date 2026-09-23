"""Finalize the selected model without opening the test dataset.

Selected model:
- Base model: no-economic logistic regression
- Calibration: sigmoid mapping
- Base model fitted on training data
- Calibrator fitted on the complete validation data

The test dataset remains locked.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_PATH = (
    PROJECT_ROOT / "data" / "processed" / "validation.parquet"
)
BASE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "logistic_no_economic.joblib"
)
FINAL_MODEL_PATH = (
    PROJECT_ROOT / "models" / "final_no_economic_sigmoid.joblib"
)
MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "exports" / "final_model_selection.json"
)

TARGET_COLUMN = "y"


def main() -> None:
    validation = pd.read_parquet(VALIDATION_PATH)

    x_validation = validation.drop(columns=[TARGET_COLUMN])
    y_validation = (
        validation[TARGET_COLUMN]
        .map({"no": 0, "yes": 1})
        .astype(int)
        .to_numpy()
    )

    base_bundle = joblib.load(BASE_MODEL_PATH)
    base_pipeline = base_bundle["pipeline"]

    validation_scores = base_pipeline.decision_function(
        x_validation
    )

    sigmoid_calibrator = LogisticRegression(
        C=1_000_000,
        solver="lbfgs",
        max_iter=5000,
    )
    sigmoid_calibrator.fit(
        validation_scores.reshape(-1, 1),
        y_validation,
    )

    calibrated_probabilities = sigmoid_calibrator.predict_proba(
        validation_scores.reshape(-1, 1)
    )[:, 1]

    final_bundle = {
        "base_pipeline": base_pipeline,
        "calibrator": sigmoid_calibrator,
        "base_model_name": "logistic_no_economic",
        "calibration_method": "sigmoid",
        "features": base_bundle["features"],
        "categorical_features": base_bundle[
            "categorical_features"
        ],
        "numeric_features": base_bundle["numeric_features"],
        "decision_time": (
            "Immediately before a scheduled marketing call, "
            "after contact channel and call date are known."
        ),
        "excluded_features": [
            "duration",
            "campaign",
            "emp.var.rate",
            "cons.price.idx",
            "cons.conf.idx",
            "euribor3m",
            "nr.employed",
        ],
        "probability_note": (
            "Probabilities describe historical response patterns. "
            "They do not estimate causal campaign uplift."
        ),
    }

    joblib.dump(final_bundle, FINAL_MODEL_PATH)

    manifest = {
        "selection_status": "locked_before_test_evaluation",
        "selected_base_model": "logistic_no_economic",
        "selected_calibration_method": "sigmoid",
        "source_feature_count": len(base_bundle["features"]),
        "base_model_training_partition": "train",
        "calibration_partition": "validation",
        "test_partition_status": "untouched",
        "selection_basis": [
            (
                "More stable probabilities after removing economic "
                "variables that were outside their training ranges "
                "during validation."
            ),
            (
                "Best later-validation sigmoid Brier score among "
                "the finalist models."
            ),
            (
                "Best later-validation sigmoid log loss among "
                "the finalist models."
            ),
            (
                "Competitive top-10-percent and top-20-percent "
                "ranking performance."
            ),
        ],
        "capacity_policy": {
            "selection_method": (
                "Rank observations by calibrated probability and "
                "select the highest-scoring observations."
            ),
            "reported_capacities": [0.10, 0.20],
            "fixed_probability_threshold": None,
            "reason": (
                "Operational capacity determines queue size; "
                "a default 0.5 threshold is not appropriate."
            ),
        },
        "validation_fit_summary": {
            "observations": int(len(validation)),
            "observed_response_rate": float(
                y_validation.mean()
            ),
            "mean_calibrated_probability": float(
                calibrated_probabilities.mean()
            ),
            "average_precision": float(
                average_precision_score(
                    y_validation,
                    calibrated_probabilities,
                )
            ),
            "brier_score": float(
                brier_score_loss(
                    y_validation,
                    calibrated_probabilities,
                )
            ),
            "log_loss": float(
                log_loss(
                    y_validation,
                    calibrated_probabilities,
                    labels=[0, 1],
                )
            ),
            "warning": (
                "These calibration metrics are in-sample because "
                "the final calibrator was fitted on the complete "
                "validation partition. They are not final "
                "generalization estimates."
            ),
        },
        "claim_boundaries": [
            (
                "The model predicts historical subscription "
                "response, not the causal effect of contacting "
                "an observation."
            ),
            (
                "The dataset does not identify unique customers, "
                "so repeated-customer leakage cannot be ruled out."
            ),
            (
                "Month-level chronology is inferred from documented "
                "row ordering; exact contact dates are unavailable."
            ),
            (
                "Results come from historical Portuguese banking "
                "campaigns and do not establish current US-market "
                "performance."
            ),
            (
                "No revenue or profit claim is made without an "
                "explicitly labeled assumed-value scenario."
            ),
        ],
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    with MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("FINAL MODEL LOCKED")
    print("Base model: logistic_no_economic")
    print("Calibration method: sigmoid")
    print(f"Features: {len(base_bundle['features'])}")
    print(
        "Validation observations used for calibration: "
        f"{len(validation):,}"
    )
    print(
        "Validation response rate: "
        f"{y_validation.mean():.2%}"
    )
    print(
        "Mean calibrated probability: "
        f"{calibrated_probabilities.mean():.2%}"
    )
    print(
        "Important: final validation calibration metrics are "
        "in-sample and are not generalization estimates."
    )
    print(f"\nModel saved to: {FINAL_MODEL_PATH}")
    print(f"Selection manifest saved to: {MANIFEST_PATH}")

    print("\nControls:")
    print("- The base model was not retrained.")
    print("- The calibrator used validation data only.")
    print("- The test dataset was not loaded.")
    print("- The model choice is now locked before testing.")


if __name__ == "__main__":
    main()