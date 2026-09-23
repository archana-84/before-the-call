"""Shared feature policies and preprocessing for model pipelines."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CLIENT_CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
]

SCHEDULING_CATEGORICAL_FEATURES = [
    "contact",
    "month",
    "day_of_week",
]

HISTORY_CATEGORICAL_FEATURES = [
    "poutcome",
]

CLIENT_NUMERIC_FEATURES = [
    "age",
]

HISTORY_NUMERIC_FEATURES = [
    "pdays",
    "previous",
]

ECONOMIC_NUMERIC_FEATURES = [
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

PRIMARY_CATEGORICAL_FEATURES = (
    CLIENT_CATEGORICAL_FEATURES
    + SCHEDULING_CATEGORICAL_FEATURES
    + HISTORY_CATEGORICAL_FEATURES
)

PRIMARY_NUMERIC_FEATURES = (
    CLIENT_NUMERIC_FEATURES
    + HISTORY_NUMERIC_FEATURES
    + ECONOMIC_NUMERIC_FEATURES
)

PRIMARY_FEATURES = (
    PRIMARY_CATEGORICAL_FEATURES
    + PRIMARY_NUMERIC_FEATURES
)

STRICT_CATEGORICAL_FEATURES = (
    CLIENT_CATEGORICAL_FEATURES
    + HISTORY_CATEGORICAL_FEATURES
)

STRICT_NUMERIC_FEATURES = (
    CLIENT_NUMERIC_FEATURES
    + HISTORY_NUMERIC_FEATURES
    + ECONOMIC_NUMERIC_FEATURES
)

NO_ECONOMIC_CATEGORICAL_FEATURES = PRIMARY_CATEGORICAL_FEATURES

NO_ECONOMIC_NUMERIC_FEATURES = (
    CLIENT_NUMERIC_FEATURES
    + HISTORY_NUMERIC_FEATURES
)

EXCLUDED_FEATURES = [
    "duration",
    "campaign",
]

TARGET_COLUMN = "target"


def build_preprocessor(
    categorical_features: list[str],
    numeric_features: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Build preprocessing that is fitted only through a model pipeline.

    Unknown strings remain ordinary categories. Categories absent from
    training are ignored during validation and test transformation.
    """
    numeric_transformer = (
        StandardScaler()
        if scale_numeric
        else "passthrough"
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                categorical_features,
            ),
            (
                "numeric",
                numeric_transformer,
                numeric_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )