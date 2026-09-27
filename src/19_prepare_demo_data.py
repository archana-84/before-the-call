"""Create a compact historical scoring file for the Streamlit demo.

The file contains anonymized test-period observations, locked model
scores, and historical outcomes. It supports retrospective capacity
simulation only; it is not current customer or US-market data.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "test.parquet"
)
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_no_economic_sigmoid.joblib"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "exports"
    / "demo_scored_test.csv"
)

TARGET_COLUMN = "y"
TEST_STARTING_SOURCE_ROW = 36_225


def main() -> None:
    test = pd.read_parquet(TEST_PATH)
    bundle = joblib.load(MODEL_PATH)

    features = bundle["features"]
    pipeline = bundle["base_pipeline"]
    calibrator = bundle["calibrator"]

    model_input = test.drop(columns=[TARGET_COLUMN])

    decision_scores = pipeline.decision_function(
        model_input
    )
    probabilities = calibrator.predict_proba(
        decision_scores.reshape(-1, 1)
    )[:, 1]

    outcomes = (
        test[TARGET_COLUMN]
        .map({"no": 0, "yes": 1})
        .astype(int)
        .to_numpy()
    )

    ranked_positions = np.argsort(
        -probabilities,
        kind="mergesort",
    )

    ranks = np.empty(len(test), dtype=int)
    ranks[ranked_positions] = np.arange(
        1,
        len(test) + 1,
    )

    demo = test[features].copy()

    demo.insert(
        0,
        "observation_id",
        np.arange(
            TEST_STARTING_SOURCE_ROW,
            TEST_STARTING_SOURCE_ROW + len(test),
        ),
    )

    demo["historical_subscribed"] = outcomes
    demo["predicted_probability"] = probabilities
    demo["score_rank"] = ranks
    demo["score_percentile"] = (
        1 - (ranks - 1) / len(test)
    )
    demo["score_decile"] = (
        np.floor(
            (ranks - 1) * 10 / len(test)
        ).astype(int)
        + 1
    )

    demo = demo.sort_values(
        ["score_rank", "observation_id"],
    ).reset_index(drop=True)

    if len(demo) != 4_964:
        raise ValueError(
            f"Expected 4,964 rows; found {len(demo):,}."
        )

    if demo["historical_subscribed"].sum() != 2_209:
        raise ValueError(
            "Historical subscriber count does not match "
            "the locked test result."
        )

    if demo["score_rank"].nunique() != len(demo):
        raise ValueError("Score ranks are not unique.")

    if demo["predicted_probability"].isna().any():
        raise ValueError(
            "Demo file contains missing probabilities."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    demo.to_csv(
        OUTPUT_PATH,
        index=False,
        float_format="%.8f",
    )

    print("STREAMLIT DEMO DATA CREATED")
    print(f"Observations: {len(demo):,}")
    print(
        "Historical subscribers: "
        f"{demo['historical_subscribed'].sum():,}"
    )
    print(
        "Historical response rate: "
        f"{demo['historical_subscribed'].mean():.2%}"
    )
    print(
        "Mean predicted probability: "
        f"{demo['predicted_probability'].mean():.2%}"
    )
    print(
        "Unique score ranks: "
        f"{demo['score_rank'].nunique():,}"
    )
    print(f"\nSaved to: {OUTPUT_PATH}")

    print("\nUsage boundary:")
    print(
        "- These are anonymized historical Portuguese "
        "campaign observations."
    )
    print(
        "- The file supports retrospective capacity "
        "demonstration only."
    )
    print(
        "- It does not contain current customers or "
        "current US-market predictions."
    )


if __name__ == "__main__":
    main()
    