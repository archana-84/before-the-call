"""Check consistency between demo scores and report exports."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIRECTORY = PROJECT_ROOT / "data" / "exports"


def test_probability_report_matches_demo_scores() -> None:
    """Published metrics must match the tracked scores."""

    demo = pd.read_csv(
        EXPORT_DIRECTORY / "demo_scored_test.csv"
    )
    metrics = pd.read_csv(
        EXPORT_DIRECTORY / "final_test_metrics.csv"
    )

    calibrated = metrics[
        metrics["variant"] == "sigmoid_calibrated"
    ].iloc[0]

    outcomes = demo[
        "historical_subscribed"
    ].to_numpy()
    probabilities = demo[
        "predicted_probability"
    ].to_numpy()

    assert np.isclose(
        calibrated["test_prevalence"],
        outcomes.mean(),
        atol=1e-8,
    )
    assert np.isclose(
        calibrated["mean_predicted_probability"],
        probabilities.mean(),
        atol=1e-8,
    )
    assert np.isclose(
        calibrated["average_precision"],
        average_precision_score(
            outcomes,
            probabilities,
        ),
        atol=1e-8,
    )
    assert np.isclose(
        calibrated["brier_score"],
        brier_score_loss(
            outcomes,
            probabilities,
        ),
        atol=1e-8,
    )
    assert np.isclose(
        calibrated["log_loss"],
        log_loss(
            outcomes,
            probabilities,
            labels=[0, 1],
        ),
        atol=1e-8,
    )


def test_python_and_sql_capacity_exports_agree() -> None:
    """Python and SQL results must match at 10% and 20%."""

    python_results = pd.read_csv(
        EXPORT_DIRECTORY
        / "final_test_capacity_metrics.csv"
    )
    sql_results = pd.read_csv(
        EXPORT_DIRECTORY
        / "sql_test_capacity_summary.csv"
    )

    for capacity in [0.10, 0.20]:
        python_row = python_results[
            np.isclose(
                python_results["capacity"],
                capacity,
            )
        ].iloc[0]

        sql_row = sql_results[
            np.isclose(
                sql_results["capacity"],
                capacity,
            )
        ].iloc[0]

        assert (
            int(python_row["selected_observations"])
            == int(sql_row["selected_observations"])
        )
        assert (
            int(python_row["selected_subscribers"])
            == int(sql_row["selected_subscribers"])
        )
        assert np.isclose(
            python_row["precision_at_capacity"],
            sql_row["precision_at_capacity"],
        )
        assert np.isclose(
            python_row["recall_at_capacity"],
            sql_row["recall_at_capacity"],
        )
        assert np.isclose(
            python_row["lift_at_capacity"],
            sql_row["lift_at_capacity"],
        )