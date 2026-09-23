from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
)


def probability_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, float | int]:
    """Calculate probability and threshold-based metrics."""
    y_array = np.asarray(y_true)
    probability_array = np.asarray(probabilities)
    predictions = (probability_array >= threshold).astype(int)

    prevalence = float(y_array.mean())
    mean_probability = float(probability_array.mean())

    return {
        "observations": int(len(y_array)),
        "positive_outcomes": int(y_array.sum()),
        "prevalence": prevalence,
        "mean_predicted_probability": mean_probability,
        "average_precision": float(
            average_precision_score(
                y_array,
                probability_array,
            )
        ),
        "precision_at_threshold": float(
            precision_score(
                y_array,
                predictions,
                zero_division=0,
            )
        ),
        "recall_at_threshold": float(
            recall_score(
                y_array,
                predictions,
                zero_division=0,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_array,
                probability_array,
            )
        ),
        "calibration_gap": mean_probability - prevalence,
        "threshold": threshold,
    }


def capacity_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    model_name: str,
    capacities: tuple[float, ...] = (0.10, 0.20),
) -> pd.DataFrame:
    """Evaluate the highest-ranked observations at each capacity."""
    y_array = np.asarray(y_true)
    probability_array = np.asarray(probabilities)

    ranking = np.argsort(
        -probability_array,
        kind="mergesort",
    )

    total_positives = int(y_array.sum())
    prevalence = float(y_array.mean())
    rows = []

    for capacity in capacities:
        selected_count = math.ceil(len(y_array) * capacity)
        selected_indices = ranking[:selected_count]

        selected_positives = int(
            y_array[selected_indices].sum()
        )
        precision = selected_positives / selected_count
        recall = (
            selected_positives / total_positives
            if total_positives > 0
            else 0.0
        )
        lift = (
            precision / prevalence
            if prevalence > 0
            else float("nan")
        )

        rows.append(
            {
                "model": model_name,
                "capacity": capacity,
                "selected_observations": selected_count,
                "selected_subscribers": selected_positives,
                "precision_at_capacity": precision,
                "recall_at_capacity": recall,
                "lift_at_capacity": lift,
                "minimum_selected_probability": float(
                    probability_array[selected_indices].min()
                ),
            }
        )

    return pd.DataFrame(rows)


def calibration_table(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    model_name: str,
    number_of_bins: int = 10,
) -> pd.DataFrame:
    """Create an equal-frequency calibration table."""
    observed_rate, mean_probability = calibration_curve(
        y_true,
        probabilities,
        n_bins=number_of_bins,
        strategy="quantile",
    )

    return pd.DataFrame(
        {
            "model": model_name,
            "bin": range(1, len(observed_rate) + 1),
            "mean_predicted_probability": mean_probability,
            "observed_response_rate": observed_rate,
            "calibration_difference": (
                mean_probability - observed_rate
            ),
        }
    )