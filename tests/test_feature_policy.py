"""Tests for leakage controls and the locked feature policy."""

from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from modeling_config import (  # noqa: E402
    EXCLUDED_FEATURES,
    PRIMARY_FEATURES,
    NO_ECONOMIC_CATEGORICAL_FEATURES,
    NO_ECONOMIC_NUMERIC_FEATURES,
)


ECONOMIC_FEATURES = {
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
}


def test_hard_leakage_features_are_excluded() -> None:
    """Duration and campaign must not enter valid models."""

    assert "duration" in EXCLUDED_FEATURES
    assert "campaign" in EXCLUDED_FEATURES
    assert "duration" not in PRIMARY_FEATURES
    assert "campaign" not in PRIMARY_FEATURES


def test_no_economic_policy_contains_13_features() -> None:
    """The locked base policy should contain 13 inputs."""

    selected_features = (
        NO_ECONOMIC_CATEGORICAL_FEATURES
        + NO_ECONOMIC_NUMERIC_FEATURES
    )

    assert len(selected_features) == 13
    assert len(selected_features) == len(
        set(selected_features)
    )
    assert ECONOMIC_FEATURES.isdisjoint(
        selected_features
    )


def test_model_was_locked_before_test() -> None:
    """The tracked manifest must preserve selection timing."""

    manifest_path = (
        PROJECT_ROOT
        / "data"
        / "exports"
        / "final_model_selection.json"
    )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert (
        manifest["selection_status"]
        == "locked_before_test_evaluation"
    )
    assert (
        manifest["selected_base_model"]
        == "logistic_no_economic"
    )
    assert (
        manifest["selected_calibration_method"]
        == "sigmoid"
    )
    assert (
        manifest["test_partition_status"]
        == "untouched"
    )