"""Analyze target imbalance and chronological response patterns."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank-additional-full.csv"
EXPORT_DIR = PROJECT_ROOT / "data" / "exports"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

MONTH_NUMBER = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def load_data() -> pd.DataFrame:
    """Load the verified raw dataset."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Run src/01_download_data.py first."
        )

    return pd.read_csv(RAW_DATA_PATH, sep=";")


def infer_month_periods(data: pd.DataFrame) -> pd.Series:
    """
    Infer year-month labels from UCI's documented chronological ordering.

    The source contains month names but not complete dates or years. UCI
    documents that the file is ordered from May 2008 through November 2010.
    A year is advanced whenever the month number moves backward.
    """
    current_year = 2008
    previous_month_number: int | None = None
    inferred_periods: list[str] = []

    for month_name in data["month"]:
        current_month_number = MONTH_NUMBER[month_name]

        if (
            previous_month_number is not None
            and current_month_number < previous_month_number
        ):
            current_year += 1

        inferred_periods.append(
            f"{current_year:04d}-{current_month_number:02d}"
        )
        previous_month_number = current_month_number

    return pd.Series(
        inferred_periods,
        index=data.index,
        name="inferred_period",
    )


def create_target_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize the binary target distribution."""
    summary = (
        data["y"]
        .value_counts()
        .reindex(["no", "yes"])
        .rename_axis("outcome")
        .reset_index(name="observations")
    )

    summary["share"] = summary["observations"] / len(data)
    return summary


def create_period_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize observations and outcomes by inferred month."""
    summary = (
        data.groupby("inferred_period", sort=False)
        .agg(
            observations=("y", "size"),
            subscribers=(
                "y",
                lambda values: int((values == "yes").sum()),
            ),
        )
        .reset_index()
    )

    summary["non_subscribers"] = (
        summary["observations"] - summary["subscribers"]
    )
    summary["response_rate"] = (
        summary["subscribers"] / summary["observations"]
    )

    return summary


def create_chronological_fifths(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Compare target prevalence across consecutive fifths."""
    labels = [
        "1 - earliest",
        "2",
        "3",
        "4",
        "5 - latest",
    ]

    working = data.copy()
    working["chronological_fifth"] = pd.qcut(
        working["observation_order"],
        q=5,
        labels=labels,
    )

    summary = (
        working.groupby(
            "chronological_fifth",
            observed=True,
            sort=False,
        )
        .agg(
            observations=("y", "size"),
            subscribers=(
                "y",
                lambda values: int((values == "yes").sum()),
            ),
            first_period=("inferred_period", "first"),
            last_period=("inferred_period", "last"),
        )
        .reset_index()
    )

    summary["response_rate"] = (
        summary["subscribers"] / summary["observations"]
    )

    return summary


def plot_target_distribution(
    target_summary: pd.DataFrame,
) -> None:
    """Save a target-distribution chart."""
    colors = ["#334155", "#0F766E"]

    figure, axis = plt.subplots(figsize=(8, 5))

    bars = axis.bar(
        target_summary["outcome"],
        target_summary["observations"],
        color=colors,
        width=0.6,
    )

    labels = [
        f"{count:,}\n({share:.1%})"
        for count, share in zip(
            target_summary["observations"],
            target_summary["share"],
            strict=True,
        )
    ]

    axis.bar_label(bars, labels=labels, padding=5)
    axis.set_title("Historical subscription outcomes")
    axis.set_xlabel("Outcome")
    axis.set_ylabel("Observations")
    axis.set_ylim(
        0,
        target_summary["observations"].max() * 1.15,
    )
    axis.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "01_target_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_response_over_time(
    period_summary: pd.DataFrame,
    overall_response_rate: float,
) -> None:
    """Save observation-volume and response-rate charts."""
    x_positions = range(len(period_summary))

    figure, (volume_axis, rate_axis) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.4]},
    )

    volume_axis.bar(
        x_positions,
        period_summary["observations"],
        color="#64748B",
    )
    volume_axis.set_title(
        "Observation volume and historical response rate by inferred month"
    )
    volume_axis.set_ylabel("Observations")
    volume_axis.spines[["top", "right"]].set_visible(False)

    rate_axis.plot(
        x_positions,
        period_summary["response_rate"] * 100,
        color="#0F766E",
        marker="o",
        linewidth=2,
    )
    rate_axis.axhline(
        overall_response_rate * 100,
        color="#B45309",
        linestyle="--",
        linewidth=1.5,
        label=f"Overall rate: {overall_response_rate:.1%}",
    )

    rate_axis.set_ylabel("Historical response rate (%)")
    rate_axis.set_xlabel("Inferred month")
    rate_axis.set_xticks(list(x_positions))
    rate_axis.set_xticklabels(
        period_summary["inferred_period"],
        rotation=45,
        ha="right",
    )
    rate_axis.legend()
    rate_axis.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "02_response_rate_by_period.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    """Create target and chronological EDA outputs."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    data["observation_order"] = range(1, len(data) + 1)
    data["inferred_period"] = infer_month_periods(data)

    target_summary = create_target_summary(data)
    period_summary = create_period_summary(data)
    fifth_summary = create_chronological_fifths(data)

    overall_response_rate = float((data["y"] == "yes").mean())

    target_summary.to_csv(
        EXPORT_DIR / "target_distribution.csv",
        index=False,
    )
    period_summary.to_csv(
        EXPORT_DIR / "chronological_response_summary.csv",
        index=False,
    )
    fifth_summary.to_csv(
        EXPORT_DIR / "chronological_fifths_summary.csv",
        index=False,
    )

    plot_target_distribution(target_summary)
    plot_response_over_time(
        period_summary,
        overall_response_rate,
    )

    print("\nTARGET DISTRIBUTION")
    print(target_summary.to_string(index=False))

    print("\nCHRONOLOGICAL FIFTHS")
    print(
        fifth_summary.to_string(
            index=False,
            formatters={
                "response_rate": lambda value: f"{value:.2%}",
            },
        )
    )

    print("\nINFERRED MONTH SUMMARY")
    print(
        period_summary.to_string(
            index=False,
            formatters={
                "response_rate": lambda value: f"{value:.2%}",
            },
        )
    )

    print("\nImportant interpretation:")
    print(
        "The inferred periods are month-level labels derived from UCI's "
        "documented row ordering. They are not exact contact dates."
    )
    print(
        "The changing response rate indicates temporal distribution shift. "
        "A random train-test split would mix these historical regimes."
    )

    print("\nEDA exports and figures created successfully.")


if __name__ == "__main__":
    main()