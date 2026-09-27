"""Before the Call Streamlit demonstration.

This app is a retrospective capacity simulator using historical,
anonymized UCI Bank Marketing test-period observations.

It is not a live decision system and does not estimate causal uplift.
"""

from pathlib import Path
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIRECTORY = PROJECT_ROOT / "data" / "exports"

DEMO_DATA_PATH = (
    EXPORT_DIRECTORY / "demo_scored_test.csv"
)
TEST_METRICS_PATH = (
    EXPORT_DIRECTORY / "final_test_metrics.csv"
)
TEST_CALIBRATION_PATH = (
    EXPORT_DIRECTORY / "final_test_calibration.csv"
)
TEST_DECILES_PATH = (
    EXPORT_DIRECTORY / "final_test_deciles.csv"
)

PRIMARY_COLOR = "#148277"
SECONDARY_COLOR = "#34495e"
REFERENCE_COLOR = "#c45a00"


st.set_page_config(
    page_title="Before the Call",
    page_icon="☎️",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1350px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .project-subtitle {
        color: #566573;
        font-size: 1.15rem;
        margin-top: -0.8rem;
        margin-bottom: 1.4rem;
    }

    .context-box {
    background: #f4f7f7;
    border-left: 5px solid #148277;
    border-radius: 0.35rem;
    padding: 0.9rem 1.1rem;
    margin: 0.8rem 0 1.2rem 0;
    color: #1f2933;
    }

    .context-box strong {
        color: #0f4c46;
    }

    .warning-box {
    background: #fff4d6;
    border-left: 5px solid #c45a00;
    border-radius: 0.35rem;
    padding: 0.9rem 1.1rem;
    margin: 0.8rem 0 1.2rem 0;
    color: #5d3a00;
    }

    .danger-box {
        background: #fdecec;
        border-left: 5px solid #b42318;
        border-radius: 0.35rem;
        padding: 0.9rem 1.1rem;
        margin: 0.8rem 0 1.2rem 0;
        color: #641e16;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_demo_data() -> pd.DataFrame:
    """Load the historical scored test observations."""

    data = pd.read_csv(DEMO_DATA_PATH)

    return data.sort_values(
        ["score_rank", "observation_id"]
    ).reset_index(drop=True)


@st.cache_data
def load_report_tables() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load final evaluation results."""

    metrics = pd.read_csv(TEST_METRICS_PATH)
    calibration = pd.read_csv(
        TEST_CALIBRATION_PATH
    )
    deciles = pd.read_csv(TEST_DECILES_PATH)

    return metrics, calibration, deciles


def build_capacity_curve(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate historical results from 1% to 50% capacity."""

    total_observations = len(data)
    total_subscribers = int(
        data["historical_subscribed"].sum()
    )
    prevalence = (
        total_subscribers / total_observations
    )

    rows = []

    for capacity_percent in range(1, 51):
        capacity = capacity_percent / 100
        selected_count = math.ceil(
            total_observations * capacity
        )
        selected = data.iloc[:selected_count]
        selected_subscribers = int(
            selected["historical_subscribed"].sum()
        )

        precision = (
            selected_subscribers / selected_count
        )
        recall = (
            selected_subscribers / total_subscribers
        )

        rows.append(
            {
                "capacity_percent": capacity_percent,
                "selected_observations": selected_count,
                "selected_subscribers": (
                    selected_subscribers
                ),
                "precision": precision,
                "recall": recall,
                "lift": precision / prevalence,
                "random_expected_subscribers": (
                    selected_count * prevalence
                ),
            }
        )

    return pd.DataFrame(rows)


def capacity_figure(
    curve: pd.DataFrame,
    selected_capacity: int,
    prevalence: float,
) -> go.Figure:
    """Create the interactive capacity chart."""

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=curve["capacity_percent"],
            y=curve["precision"] * 100,
            mode="lines",
            name="Historical precision",
            line={
                "color": SECONDARY_COLOR,
                "width": 3,
            },
        )
    )

    figure.add_trace(
        go.Scatter(
            x=curve["capacity_percent"],
            y=curve["recall"] * 100,
            mode="lines",
            name="Historical recall",
            line={
                "color": PRIMARY_COLOR,
                "width": 3,
            },
        )
    )

    figure.add_hline(
        y=prevalence * 100,
        line_dash="dash",
        line_color=REFERENCE_COLOR,
        annotation_text=(
            f"Overall response rate: "
            f"{prevalence:.1%}"
        ),
    )

    figure.add_vline(
        x=selected_capacity,
        line_dash="dot",
        line_color="#777777",
        annotation_text=(
            f"Selected: {selected_capacity}%"
        ),
    )

    figure.update_layout(
        title="How performance changes with contact capacity",
        xaxis_title="Observations selected (%)",
        yaxis_title="Metric (%)",
        xaxis={"range": [1, 50]},
        yaxis={"range": [0, 100]},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        margin={"l": 30, "r": 30, "t": 90, "b": 40},
    )

    return figure


def decile_figure(
    deciles: pd.DataFrame,
    prevalence: float,
) -> go.Figure:
    """Create the observed response-by-decile chart."""

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=deciles["score_decile"],
            y=deciles[
                "observed_response_rate"
            ]
            * 100,
            name="Observed response rate",
            marker_color=PRIMARY_COLOR,
        )
    )

    figure.add_hline(
        y=prevalence * 100,
        line_dash="dash",
        line_color=REFERENCE_COLOR,
        annotation_text=(
            f"Overall rate: {prevalence:.1%}"
        ),
    )

    figure.update_layout(
        title=(
            "Observed response by ranked score decile"
        ),
        xaxis_title=(
            "Score decile (1 = highest ranked)"
        ),
        yaxis_title="Historical response rate (%)",
        xaxis={"dtick": 1},
        margin={"l": 30, "r": 30, "t": 70, "b": 40},
    )

    return figure


def calibration_figure(
    calibration: pd.DataFrame,
) -> go.Figure:
    """Create the final test calibration chart."""

    figure = go.Figure()

    colors = {
        "uncalibrated": SECONDARY_COLOR,
        "sigmoid_calibrated": PRIMARY_COLOR,
    }

    for variant, variant_data in calibration.groupby(
        "variant"
    ):
        figure.add_trace(
            go.Scatter(
                x=variant_data[
                    "mean_predicted_probability"
                ]
                * 100,
                y=variant_data[
                    "observed_response_rate"
                ]
                * 100,
                mode="lines+markers",
                name=variant.replace(
                    "_",
                    " ",
                ).title(),
                line={
                    "color": colors.get(
                        variant,
                        "#777777",
                    ),
                    "width": 3,
                },
            )
        )

    figure.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            name="Perfect calibration",
            line={
                "color": REFERENCE_COLOR,
                "dash": "dash",
            },
        )
    )

    figure.update_layout(
        title="Calibration in the untouched test period",
        xaxis_title="Mean predicted probability (%)",
        yaxis_title="Observed response rate (%)",
        xaxis={"range": [0, 100]},
        yaxis={"range": [0, 100]},
        margin={"l": 30, "r": 30, "t": 70, "b": 40},
    )

    return figure


data = load_demo_data()
metrics, calibration, deciles = load_report_tables()
capacity_curve = build_capacity_curve(data)

total_observations = len(data)
total_subscribers = int(
    data["historical_subscribed"].sum()
)
prevalence = total_subscribers / total_observations


st.title("☎️ Before the Call")
st.markdown(
    '<p class="project-subtitle">'
    "Predicting campaign response before outreach"
    "</p>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="context-box">
    <strong>Retrospective historical demonstration:</strong>
    this app uses anonymized Portuguese bank-marketing
    observations from 2009–2010. It demonstrates how limited
    contact capacity changes a ranked queue. It is not a
    deployment-ready system or evidence that contacting an
    observation causes subscription.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="warning-box">
    <strong>Model-risk warning:</strong>
    the locked model showed weak temporal generalization in the
    untouched test period. At 10% capacity, lift was 0.97; at
    20%, lift was 1.10. Treat this as a decision-analysis
    demonstration—not an operational tool.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Simulation controls")

    capacity_percent = st.slider(
        "Contact capacity",
        min_value=1,
        max_value=50,
        value=20,
        step=1,
        help=(
            "The share of historical observations that "
            "can be included in the ranked contact queue."
        ),
    )

    show_historical_outcomes = st.toggle(
        "Show historical evaluation",
        value=True,
        help=(
            "Turn this off to imitate the decision-time view, "
            "when outcomes would not yet be known."
        ),
    )

    st.divider()

    st.markdown("**Dataset context**")
    st.caption(
        "UCI Bank Marketing — Portuguese direct-marketing "
        "campaigns, May 2008 to November 2010."
    )

    st.markdown("**Decision time**")
    st.caption(
        "Immediately before a scheduled call, after channel "
        "and call date are assumed to be known."
    )

    st.markdown("**Excluded leakage**")
    st.caption(
        "`duration` and `campaign` are excluded. Economic "
        "indicators were also removed from the selected model "
        "because of temporal extrapolation."
    )


selected_count = math.ceil(
    total_observations * capacity_percent / 100
)
selected = data.iloc[:selected_count].copy()

selected_subscribers = int(
    selected["historical_subscribed"].sum()
)
precision = selected_subscribers / selected_count
recall = selected_subscribers / total_subscribers
lift = precision / prevalence
random_expected_subscribers = (
    selected_count * prevalence
)
minimum_selected_probability = float(
    selected["predicted_probability"].min()
)


simulator_tab, monitoring_tab, methods_tab = st.tabs(
    [
        "Capacity simulator",
        "Model monitoring",
        "Method and limitations",
    ]
)


with simulator_tab:
    st.subheader(
        f"Highest-ranked {capacity_percent}% "
        "of historical observations"
    )

    if show_historical_outcomes:
        metric_columns = st.columns(5)

        metric_columns[0].metric(
            "Selected observations",
            f"{selected_count:,}",
        )
        metric_columns[1].metric(
            "Historical subscribers",
            f"{selected_subscribers:,}",
            delta=(
                f"{selected_subscribers - random_expected_subscribers:+.1f} "
                "vs random expectation"
            ),
        )
        metric_columns[2].metric(
            "Historical precision",
            f"{precision:.1%}",
            delta=(
                f"{(precision - prevalence) * 100:+.1f} pp "
                "vs overall"
            ),
        )
        metric_columns[3].metric(
            "Historical recall",
            f"{recall:.1%}",
            delta=(
                f"{(recall - capacity_percent / 100) * 100:+.1f} pp "
                "vs random"
            ),
        )
        metric_columns[4].metric(
            "Lift",
            f"{lift:.2f}",
        )

        st.plotly_chart(
            capacity_figure(
                capacity_curve,
                capacity_percent,
                prevalence,
            ),
            width="stretch",
        )
    else:
        metric_columns = st.columns(4)

        metric_columns[0].metric(
            "Selected observations",
            f"{selected_count:,}",
        )
        metric_columns[1].metric(
            "Capacity",
            f"{capacity_percent}%",
        )
        metric_columns[2].metric(
            "Mean selected score",
            (
                f"{selected['predicted_probability'].mean():.1%}"
            ),
        )
        metric_columns[3].metric(
            "Minimum selected score",
            f"{minimum_selected_probability:.1%}",
        )

        st.info(
            "Decision-time view: historical outcomes are hidden. "
            "In a real campaign, these outcomes would not yet "
            "be available."
        )

    st.subheader("Ranked contact queue")

    queue_columns = [
        "observation_id",
        "predicted_probability",
        "age",
        "job",
        "contact",
        "month",
        "day_of_week",
        "previous",
        "poutcome",
    ]

    if show_historical_outcomes:
        queue_columns.append(
            "historical_subscribed"
        )

    queue = selected[queue_columns].copy()
    queue["predicted_probability"] = (
        queue["predicted_probability"] * 100
    )

    if show_historical_outcomes:
        queue["historical_subscribed"] = (
            queue["historical_subscribed"].map(
                {
                    0: "No",
                    1: "Yes",
                }
            )
        )

    queue = queue.rename(
        columns={
            "observation_id": "Observation",
            "predicted_probability": "Model score (%)",
            "age": "Age",
            "job": "Job",
            "contact": "Channel",
            "month": "Month",
            "day_of_week": "Day",
            "previous": "Previous contacts",
            "poutcome": "Previous outcome",
            "historical_subscribed": (
                "Historical subscription"
            ),
        }
    )

    st.dataframe(
        queue,
        width="stretch",
        hide_index=True,
        height=430,
        column_config={
            "Model score (%)": st.column_config.NumberColumn(
                format="%.2f"
            )
        },
    )

    download_data = queue.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download selected historical queue",
        data=download_data,
        file_name=(
            f"historical_ranked_queue_"
            f"{capacity_percent}_percent.csv"
        ),
        mime="text/csv",
    )

    st.caption(
        "Observation numbers are synthetic row identifiers, "
        "not customer IDs. Repeated customers cannot be "
        "identified in the source data."
    )


with monitoring_tab:
    calibrated_metrics = metrics[
        metrics["variant"]
        == "sigmoid_calibrated"
    ].iloc[0]

    st.subheader("Untouched test-period result")

    monitoring_columns = st.columns(4)

    monitoring_columns[0].metric(
        "Test prevalence",
        f"{calibrated_metrics['test_prevalence']:.1%}",
    )
    monitoring_columns[1].metric(
        "Average precision",
        f"{calibrated_metrics['average_precision']:.3f}",
        delta=(
            f"{calibrated_metrics['average_precision'] - calibrated_metrics['test_prevalence']:+.3f} "
            "above no-skill AP"
        ),
    )
    monitoring_columns[2].metric(
        "Mean predicted probability",
        (
            f"{calibrated_metrics['mean_predicted_probability']:.1%}"
        ),
    )
    monitoring_columns[3].metric(
        "Calibration gap",
        (
            f"{calibrated_metrics['calibration_gap']:.1%}"
        ),
    )

    left_chart, right_chart = st.columns(2)

    with left_chart:
        st.plotly_chart(
            decile_figure(
                deciles,
                prevalence,
            ),

            width="stretch",
        )

    with right_chart:
        st.plotly_chart(
            calibration_figure(calibration),
            width="stretch",
        )

    st.markdown(
    """
    <div class="danger-box">
    <strong>Deployment recommendation:</strong>
    do not deploy this model without newer representative data,
    verified customer identifiers, rolling temporal validation,
    recalibration, and ongoing drift monitoring.
    </div>
    """,
    unsafe_allow_html=True,
    )


with methods_tab:
    st.subheader("Prediction design")

    st.markdown(
        """
        - **Target:** whether the historical observation ended
          in a term-deposit subscription.
        - **Unit:** one campaign/contact observation—not a
          verified unique customer.
        - **Decision time:** immediately before a scheduled call.
        - **Selected model:** regularized logistic regression
          without economic indicators, followed by sigmoid
          calibration.
        - **Operational rule:** rank scores and select only the
          number allowed by contact capacity.
        """
    )

    st.subheader("Chronological evaluation")

    split_table = pd.DataFrame(
        {
            "Partition": [
                "Training",
                "Validation",
                "Untouched test",
            ],
            "Observations": [
                27_680,
                8_544,
                4_964,
            ],
            "Historical response rate": [
                "4.83%",
                "12.79%",
                "44.50%",
            ],
            "Use": [
                "Fit preprocessing and base models",
                "Choose policy and fit calibration",
                "One-time final evaluation",
            ],
        }
    )

    st.dataframe(
        split_table,
        width="stretch",
        hide_index=True,
    )

    st.subheader("Claim boundaries")

    st.markdown(
        """
        - Historical response prediction does **not** estimate
          the causal effect of making a call.
        - The project does not claim campaign uplift, increased
          revenue, or current US-market performance.
        - Exact contact dates and customer identifiers are not
          available.
        - Repeated-customer leakage cannot be ruled out.
        - Month-level chronology is inferred from the documented
          ordering of the UCI dataset.
        - Demographic patterns are descriptive model
          associations, not causal explanations or fairness
          certification.
        """
    )

    with st.expander("Why exclude call duration?"):
        st.write(
            "Call duration is only known after the conversation "
            "has occurred. Using it for a pre-call decision would "
            "leak future information and create an unrealistically "
            "strong model."
        )

    with st.expander("Why did the final test result weaken?"):
        st.write(
            "The response rate and feature relationships changed "
            "substantially over time. The model learned patterns "
            "from an early campaign regime that did not transfer "
            "reliably to the much later test period. Calibration "
            "changed the probability scale but could not repair "
            "the unstable ranking."
        )

    with st.expander("Why use capacity instead of a 0.5 threshold?"):
        st.write(
            "The business constraint is the number of calls that "
            "can be made. Ranking the available observations and "
            "selecting the top capacity share directly matches "
            "that decision. A universal 0.5 cutoff does not."
        )


st.divider()
st.markdown(
    """
    <p class="small-note">
    Source: UCI Bank Marketing dataset. Historical Portuguese
    direct-marketing observations, 2008–2010. This educational
    portfolio demo is not financial advice, a production model,
    or a causal campaign-impact estimate.
    </p>
    """,
    unsafe_allow_html=True,
)