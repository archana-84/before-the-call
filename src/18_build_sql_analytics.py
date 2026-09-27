"""Build the local SQLite analytics layer.

The database contains the three chronological partitions and the
locked final-model scores for the test period. SQL views reproduce
key model-monitoring and capacity results.

This is post-test reporting and does not change the model.
"""

from pathlib import Path
import math
import sqlite3

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_ROOT / "data" / "processed" / "train.parquet"
)
VALIDATION_PATH = (
    PROJECT_ROOT / "data" / "processed" / "validation.parquet"
)
TEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "test.parquet"
)
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_no_economic_sigmoid.joblib"
)
SQL_PATH = (
    PROJECT_ROOT / "sql" / "analytics_views.sql"
)
DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "before_the_call.sqlite"
)
EXPORT_DIRECTORY = PROJECT_ROOT / "data" / "exports"

TARGET_COLUMN = "y"


def binary_target(data: pd.DataFrame) -> np.ndarray:
    """Convert the yes/no target to integers."""

    return (
        data[TARGET_COLUMN]
        .map({"no": 0, "yes": 1})
        .astype(int)
        .to_numpy()
    )


def selection_mask(
    probabilities: np.ndarray,
    capacity: float,
) -> np.ndarray:
    """Select the exact highest-ranked capacity share."""

    selected_count = math.ceil(
        len(probabilities) * capacity
    )
    ranked_positions = np.argsort(
        -probabilities,
        kind="mergesort",
    )

    selected = np.zeros(
        len(probabilities),
        dtype=int,
    )
    selected[ranked_positions[:selected_count]] = 1

    return selected


def prepare_partition(
    data: pd.DataFrame,
    partition_name: str,
    features: list[str],
) -> pd.DataFrame:
    """Prepare one partition for SQLite."""

    prepared = data[features].copy()
    prepared["dataset_split"] = partition_name
    prepared["subscribed"] = binary_target(data)
    prepared["calibrated_probability"] = np.nan
    prepared["selected_10"] = 0
    prepared["selected_20"] = 0

    return prepared


def main() -> None:
    EXPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading chronological partitions...")

    train = pd.read_parquet(TRAIN_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)
    test = pd.read_parquet(TEST_PATH)

    bundle = joblib.load(MODEL_PATH)
    features = bundle["features"]
    pipeline = bundle["base_pipeline"]
    calibrator = bundle["calibrator"]

    test_scores = pipeline.decision_function(
        test.drop(columns=[TARGET_COLUMN])
    )
    test_probabilities = calibrator.predict_proba(
        test_scores.reshape(-1, 1)
    )[:, 1]

    train_table = prepare_partition(
        train,
        "train",
        features,
    )
    validation_table = prepare_partition(
        validation,
        "validation",
        features,
    )
    test_table = prepare_partition(
        test,
        "test",
        features,
    )

    test_table["calibrated_probability"] = (
        test_probabilities
    )
    test_table["selected_10"] = selection_mask(
        test_probabilities,
        0.10,
    )
    test_table["selected_20"] = selection_mask(
        test_probabilities,
        0.20,
    )

    observations = pd.concat(
        [
            train_table,
            validation_table,
            test_table,
        ],
        ignore_index=True,
    )

    observations.insert(
        0,
        "observation_id",
        np.arange(1, len(observations) + 1),
    )

    column_order = [
        "observation_id",
        "dataset_split",
        *features,
        "subscribed",
        "calibrated_probability",
        "selected_10",
        "selected_20",
    ]
    observations = observations[column_order]

    print("Building SQLite database...")

    with sqlite3.connect(DATABASE_PATH) as connection:
        observations.to_sql(
            "campaign_observations",
            connection,
            if_exists="replace",
            index=False,
        )

        connection.executescript(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
                idx_campaign_observation_id
            ON campaign_observations(observation_id);

            CREATE INDEX IF NOT EXISTS
                idx_campaign_split
            ON campaign_observations(dataset_split);

            CREATE INDEX IF NOT EXISTS
                idx_campaign_test_probability
            ON campaign_observations(
                dataset_split,
                calibrated_probability DESC
            );

            CREATE INDEX IF NOT EXISTS
                idx_campaign_contact
            ON campaign_observations(contact);

            CREATE INDEX IF NOT EXISTS
                idx_campaign_month
            ON campaign_observations(month);
            """
        )

        sql_script = SQL_PATH.read_text(
            encoding="utf-8"
        )
        connection.executescript(sql_script)

        exports = {
            "sql_split_summary.csv": """
                SELECT *
                FROM vw_split_summary
                ORDER BY
                    CASE dataset_split
                        WHEN 'train' THEN 1
                        WHEN 'validation' THEN 2
                        WHEN 'test' THEN 3
                    END;
            """,
            "sql_test_capacity_summary.csv": """
                SELECT *
                FROM vw_test_capacity_summary
                ORDER BY capacity;
            """,
            "sql_test_score_deciles.csv": """
                SELECT *
                FROM vw_test_score_deciles
                ORDER BY score_decile;
            """,
            "sql_test_contact_audit.csv": """
                SELECT *
                FROM vw_test_contact_audit
                ORDER BY contact;
            """,
            "sql_test_month_audit.csv": """
                SELECT *
                FROM vw_test_month_audit
                ORDER BY
                    CASE month
                        WHEN 'mar' THEN 1
                        WHEN 'apr' THEN 2
                        WHEN 'may' THEN 3
                        WHEN 'jun' THEN 4
                        WHEN 'jul' THEN 5
                        WHEN 'aug' THEN 6
                        WHEN 'sep' THEN 7
                        WHEN 'oct' THEN 8
                        WHEN 'nov' THEN 9
                        WHEN 'dec' THEN 10
                    END;
            """,
        }

        exported_tables = {}

        for filename, query in exports.items():
            result = pd.read_sql_query(
                query,
                connection,
            )
            result.to_csv(
                EXPORT_DIRECTORY / filename,
                index=False,
            )
            exported_tables[filename] = result

        table_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM campaign_observations;
            """
        ).fetchone()[0]

        view_names = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'view'
              AND name LIKE 'vw_%'
            ORDER BY name;
            """,
            connection,
        )

    if table_count != 41_188:
        raise ValueError(
            "SQLite row count does not match source data: "
            f"{table_count:,}"
        )

    capacity_result = exported_tables[
        "sql_test_capacity_summary.csv"
    ]

    expected_capacity_results = {
        0.10: {
            "selected_observations": 497,
            "selected_subscribers": 214,
        },
        0.20: {
            "selected_observations": 993,
            "selected_subscribers": 484,
        },
    }

    for capacity, expected in (
        expected_capacity_results.items()
    ):
        row = capacity_result[
            np.isclose(
                capacity_result["capacity"],
                capacity,
            )
        ].iloc[0]

        for metric, expected_value in expected.items():
            actual_value = int(row[metric])

            if actual_value != expected_value:
                raise ValueError(
                    "SQL capacity result does not match "
                    f"the locked Python result for "
                    f"{capacity:.0%} {metric}: "
                    f"expected {expected_value}; "
                    f"found {actual_value}."
                )

    print("\nSQL SPLIT SUMMARY")
    print(
        exported_tables[
            "sql_split_summary.csv"
        ].to_string(
            index=False,
            formatters={
                "response_rate": (
                    lambda value: f"{value:.2%}"
                )
            },
        )
    )

    print("\nSQL TEST CAPACITY SUMMARY")
    print(
        capacity_result.to_string(
            index=False,
            formatters={
                "capacity": (
                    lambda value: f"{value:.0%}"
                ),
                "precision_at_capacity": (
                    lambda value: f"{value:.2%}"
                ),
                "recall_at_capacity": (
                    lambda value: f"{value:.2%}"
                ),
                "lift_at_capacity": (
                    lambda value: f"{value:.2f}"
                ),
            },
        )
    )

    print("\nSQL OBJECTS CREATED")
    print(f"Rows: {table_count:,}")
    print(
        "Views: "
        + ", ".join(view_names["name"].tolist())
    )

    print(f"\nDatabase saved to: {DATABASE_PATH}")
    print(f"Exports saved to: {EXPORT_DIRECTORY}")

    print("\nControls:")
    print(
        "- SQL reproduced the locked 10% and 20% "
        "capacity results."
    )
    print(
        "- The SQLite database is generated locally "
        "and remains ignored by Git."
    )
    print(
        "- No model or probability was refitted."
    )
    print(
        "- SQL results are post-test reporting, "
        "not model tuning."
    )


if __name__ == "__main__":
    main()