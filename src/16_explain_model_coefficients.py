"""Extract and visualize final logistic model coefficients.

These coefficients describe model associations, not causal effects.
The test dataset is not needed or loaded.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_no_economic_sigmoid.joblib"
)
EXPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "exports"
    / "final_model_coefficients.csv"
)
FIGURE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "10_feature_contributions.png"
)


def identify_source_feature(
    transformed_feature: str,
    source_features: list[str],
) -> str:
    """Map an encoded feature back to its source column."""

    for feature in sorted(
        source_features,
        key=len,
        reverse=True,
    ):
        if transformed_feature == feature:
            return feature

        if transformed_feature.startswith(f"{feature}_"):
            return feature

    return "unmapped"


def display_label(
    transformed_feature: str,
    source_feature: str,
    numeric_features: list[str],
) -> str:
    """Create a readable chart label."""

    if source_feature in numeric_features:
        return f"{source_feature} (per 1 SD)"

    category = transformed_feature[
        len(source_feature) + 1 :
    ]

    return f"{source_feature} = {category}"


def main() -> None:
    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(MODEL_PATH)

    pipeline = bundle["base_pipeline"]
    calibrator = bundle["calibrator"]

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    transformed_features = (
        preprocessor.get_feature_names_out()
    )

    base_coefficients = classifier.coef_.ravel()
    base_intercept = float(classifier.intercept_[0])

    calibration_slope = float(
        calibrator.coef_.ravel()[0]
    )
    calibration_intercept = float(
        calibrator.intercept_[0]
    )

    final_coefficients = (
        base_coefficients * calibration_slope
    )
    final_intercept = (
        base_intercept * calibration_slope
        + calibration_intercept
    )

    source_features = bundle["features"]
    numeric_features = bundle["numeric_features"]

    coefficient_table = pd.DataFrame(
        {
            "transformed_feature": transformed_features,
            "base_coefficient": base_coefficients,
            "calibrated_coefficient": final_coefficients,
        }
    )

    coefficient_table["source_feature"] = (
        coefficient_table["transformed_feature"].map(
            lambda feature: identify_source_feature(
                feature,
                source_features,
            )
        )
    )

    coefficient_table["feature_type"] = (
        coefficient_table["source_feature"].map(
            lambda feature: (
                "numeric_standardized"
                if feature in numeric_features
                else "categorical_indicator"
            )
        )
    )

    coefficient_table["display_label"] = (
        coefficient_table.apply(
            lambda row: display_label(
                row["transformed_feature"],
                row["source_feature"],
                numeric_features,
            ),
            axis=1,
        )
    )

    coefficient_table["odds_ratio"] = np.exp(
        coefficient_table["calibrated_coefficient"]
    )

    coefficient_table[
        "absolute_calibrated_coefficient"
    ] = coefficient_table[
        "calibrated_coefficient"
    ].abs()

    coefficient_table = coefficient_table.sort_values(
        "calibrated_coefficient",
        ascending=False,
    ).reset_index(drop=True)

    coefficient_table.to_csv(
        EXPORT_PATH,
        index=False,
    )

    positive = (
    coefficient_table[
        coefficient_table["calibrated_coefficient"] > 0
    ]
    .nlargest(12, "calibrated_coefficient")
    .sort_values("calibrated_coefficient")
    )

    negative = (
        coefficient_table[
            coefficient_table["calibrated_coefficient"] < 0
        ]
        .nsmallest(12, "calibrated_coefficient")
        .sort_values(
            "calibrated_coefficient",
            ascending=False,
        )
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, 8),
    )

    axes[0].barh(
        positive["display_label"],
        positive["calibrated_coefficient"],
        color="#148277",
    )
    axes[0].axvline(
        0,
        color="#555555",
        linewidth=1,
    )
    axes[0].set_title(
        "Largest positive model coefficients"
    )
    axes[0].set_xlabel(
        "Contribution to calibrated log-odds"
    )
    axes[0].grid(axis="x", alpha=0.2)

    axes[1].barh(
        negative["display_label"],
        negative["calibrated_coefficient"],
        color="#9c4f4f",
    )
    axes[1].axvline(
        0,
        color="#555555",
        linewidth=1,
    )
    axes[1].set_title(
        "Largest negative model coefficients"
    )
    axes[1].set_xlabel(
        "Contribution to calibrated log-odds"
    )
    axes[1].grid(axis="x", alpha=0.2)

    figure.suptitle(
        "Feature contributions in the locked logistic model",
        fontsize=17,
        y=0.99,
    )
    figure.text(
        0.5,
        0.945,
        (
            "Coefficients are conditional model associations; "
            "they are not causal effects or campaign uplift"
        ),
        ha="center",
        fontsize=10,
        color="#555555",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.92])

    figure.savefig(
        FIGURE_PATH,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("FINAL MODEL STRUCTURE")
    print(
        f"Encoded features: "
        f"{len(transformed_features)}"
    )
    print(
        f"Sigmoid calibration slope: "
        f"{calibration_slope:.4f}"
    )
    print(
        f"Sigmoid calibration intercept: "
        f"{calibration_intercept:.4f}"
    )
    print(
        f"Combined final intercept: "
        f"{final_intercept:.4f}"
    )

    print("\nLARGEST POSITIVE COEFFICIENTS")
    print(
        positive[
            [
                "display_label",
                "calibrated_coefficient",
                "odds_ratio",
            ]
        ]
        .sort_values(
            "calibrated_coefficient",
            ascending=False,
        )
        .to_string(
            index=False,
            formatters={
                "calibrated_coefficient": (
                    lambda value: f"{value:.4f}"
                ),
                "odds_ratio": (
                    lambda value: f"{value:.3f}"
                ),
            },
        )
    )

    print("\nLARGEST NEGATIVE COEFFICIENTS")
    print(
        negative[
            [
                "display_label",
                "calibrated_coefficient",
                "odds_ratio",
            ]
        ]
        .sort_values(
            "calibrated_coefficient"
        )
        .to_string(
            index=False,
            formatters={
                "calibrated_coefficient": (
                    lambda value: f"{value:.4f}"
                ),
                "odds_ratio": (
                    lambda value: f"{value:.3f}"
                ),
            },
        )
    )

    print("\nInterpretation controls:")
    print(
        "- Positive coefficients increase the model score; "
        "negative coefficients decrease it."
    )
    print(
        "- Numeric coefficients represent a one-training-"
        "standard-deviation increase."
    )
    print(
        "- One-hot coefficients are regularized category "
        "associations, not simple unadjusted response rates."
    )
    print(
        "- Coefficient magnitude does not guarantee stability "
        "in later periods."
    )
    print(
        "- No causal or fairness conclusion should be drawn "
        "from a coefficient alone."
    )
    print("- Test data was not loaded.")

    print(f"\nCoefficient table saved to: {EXPORT_PATH}")
    print(f"Figure saved to: {FIGURE_PATH}")


if __name__ == "__main__":
    main()