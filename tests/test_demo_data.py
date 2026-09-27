"""Tests for the Streamlit demonstration dataset."""

from pathlib import Path
import math

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = (
    PROJECT_ROOT
    / "data"
    / "exports"
    / "demo_scored_test.csv"
)


def load_demo() -> pd.DataFrame:
    """Load the tracked demo file."""

    return pd.read_csv(DEMO_PATH)


def test_demo_data_contract() -> None:
    """The demo file must match the locked test partition."""

    data = load_demo()

    assert len(data) == 4_964
    assert data["historical_subscribed"].sum() == 2_209
    assert data["observation_id"].is_unique
    assert data["score_rank"].is_unique
    assert data["predicted_probability"].notna().all()

    assert data["predicted_probability"].between(
        0,
        1,
        inclusive="both",
    ).all()

    expected_ranks = np.arange(1, len(data) + 1)

    np.testing.assert_array_equal(
        data.sort_values("score_rank")[
            "score_rank"
        ].to_numpy(),
        expected_ranks,
    )


def test_demo_contains_no_leakage_columns() -> None:
    """The app must not expose excluded call information."""

    data = load_demo()

    assert "duration" not in data.columns
    assert "campaign" not in data.columns
    assert "y" not in data.columns


def test_locked_capacity_counts() -> None:
    """The demo must reproduce final 10% and 20% results."""

    data = load_demo().sort_values("score_rank")

    expected = {
        0.10: {
            "selected_observations": 497,
            "selected_subscribers": 214,
        },
        0.20: {
            "selected_observations": 993,
            "selected_subscribers": 484,
        },
    }

    for capacity, expected_values in expected.items():
        selected_count = math.ceil(
            len(data) * capacity
        )
        selected = data.iloc[:selected_count]

        assert (
            selected_count
            == expected_values[
                "selected_observations"
            ]
        )
        assert (
            selected["historical_subscribed"].sum()
            == expected_values[
                "selected_subscribers"
            ]
        )