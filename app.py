import pandas as pd
import streamlit as st

from src.windows import get_monitoring_windows
from src.outlier_detection import detect_outliers
from src.gemini_review import run_gemini_review

st.set_page_config(
    page_title="Non-Response Outlier Monitor",
    layout="wide",
)
MAX_GEMINI_CALLS_PER_SESSION = 2
MAX_GEMINI_MESSAGES_PER_CALL = 20

if "gemini_calls_used" not in st.session_state:
    st.session_state.gemini_calls_used = 0


st.title("AI Support Non-Response Monitoring")

st.caption(
    "Operational analytics workflow for monitoring & investigating **AI support assistant intent categorization failures and blind spots**."
)

st.markdown("""
This project demonstrates an operational analytics workflow for monitoring and investigating
**AI support assistant intent categorization failures and blind spots**.

The workflow detects abnormal spikes in AI assistant non-response behavior,
surfaces raw investigation samples, and generates LLM-assisted operational summaries.
""")

with st.expander("Project background and methodology"):
    st.markdown("""
The demo simulates a production environment where an AI support assistant listens to high-volume community messages and selectively decides when not to answer.

A non-response category means the assistant logged the user message but intentionally did not generate a reply because the message was classified into categories such as:

- unclear requests
- missing context
- unsupported topics
- spam or malicious content
- fallback categories

Monitoring these categories helps detect when:

- eligible user questions may be ignored
- documentation coverage may be missing
- routing or taxonomy quality may degrade
- fallback categories absorb too many requests
- moderation or spam events affect support traffic

The workflow compares a selected detection week against a historical baseline and flags category/date combinations with unusually high behavior.

The demo also showcases:

- statistical outlier detection
- raw-message investigation
- LLM-assisted review
- operational alert summarization

All data shown in this demo is synthetic and anonymized.

In production, this workflow is scheduled to run automatically once per week. Relevant teams receive a Slack summary message (see the final section) with links to the supporting raw investigation data.
""")

st.info(
    "This portfolio demo uses synthetic data inspired by real monitoring patterns. "
    "No production messages, user identifiers, internal links, or proprietary datasets are included."
)

daily_totals_df = pd.read_csv("data/synthetic_daily_totals.csv")
daily_totals_df["message_date"] = pd.to_datetime(daily_totals_df["message_date"]).dt.date

min_date = daily_totals_df["message_date"].min()
max_date = daily_totals_df["message_date"].max()

st.sidebar.header("Monitoring settings")

detection_start = st.sidebar.date_input(
    "Detection start date",
    value=pd.to_datetime("2026-05-24").date(),
    min_value=min_date,
    max_value=max_date,
)

detection_end = st.sidebar.date_input(
    "Detection end date",
    value=pd.to_datetime("2026-05-30").date(),
    min_value=min_date,
    max_value=max_date,
)

baseline_days = st.sidebar.slider(
    "Baseline window days",
    min_value=30,
    max_value=90,
    value=90,
    step=10,
)

baseline_df, detection_df, metadata = get_monitoring_windows(
    df=daily_totals_df,
    detection_start_date=str(detection_start),
    detection_end_date=str(detection_end),
    baseline_days=baseline_days,
    minimum_baseline_days=30,
)

st.subheader("Monitoring window selection")
st.markdown("""
The **detection window** is the week being evaluated.

The **baseline window** is the historical period before the detection week.
The baseline does not include the detection week, which prevents an abnormal spike from influencing its own threshold.
""")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Baseline start", str( metadata["baseline_start"]))
col2.metric("Baseline end", str(metadata["baseline_end"]))
col3.metric("Detection start", str(metadata["detection_start"]))
col4.metric("Detection end", str(metadata["detection_end"]))

col5, col6 = st.columns(2)

col5.metric("Baseline days available", metadata["actual_baseline_days"])
col6.metric("Detection days available", metadata["actual_detection_days"])

if not metadata["has_enough_baseline"]:
    st.error("Not enough baseline data to run detection.")
    st.stop()

if not metadata["has_detection_data"]:
    st.error("No detection data found for this date range.")
    st.stop()

outliers_df = detect_outliers(
    baseline_df=baseline_df,
    detection_df=detection_df,
)

raw_messages_df = pd.read_csv("data/synthetic_raw_messages.csv")
raw_messages_df["request_date"] = pd.to_datetime(raw_messages_df["request_date"]).dt.date

pregenerated_reviews_df = pd.read_csv("data/pregenerated_llm_reviews.csv")

st.markdown("""
###Detection-week traffic

This table shows daily message volume and non-response category counts for the selected week.

The main metric is:

`category_share = category_count / non_response_total`

Shares are used instead of raw counts because total traffic can change from day to day.
""")

st.subheader("Detection week daily totals")
st.dataframe(detection_df, use_container_width=True, height = 300)

st.markdown("""
### Historical baseline

For each category, the system learns a separate historical baseline.

This matters because categories behave differently:
- some categories are common
- some are rare
- some naturally fluctuate more than others

The outlier logic uses historical average, standard deviation, and percentile thresholds for each category.
""")

st.subheader("Baseline sample")
with st.expander("View baseline data sample"):
    st.dataframe(
        baseline_df,
        use_container_width=True,
        height=300,
    )



st.subheader("Outlier detection results")
st.markdown("""
A category/date is flagged when its share or count is unusually high compared with its own historical baseline.

Raw messages export and review is suggested only when the outlier is both:
`high severity` and `review eligible`.
""")

with st.expander("How the outlier logic works"):
    st.markdown("""
This demo uses the same statistical logic as the production monitoring workflow, adapted to synthetic data and public-safe category names.

The main metric is:

`category_share = category_count / non_response_total`

Using share helps normalize for changing community volume.  
For example, 30 messages in one category may be normal on a very busy day, but unusual on a quieter day.

The system does **not** use one global threshold for all categories.  
Different categories can indicate different operational issues, so the workflow combines several detection methods instead of relying on one fixed threshold.

Examples:
- average-based thresholds detect sudden deviations from a category’s usual behavior
- percentile thresholds detect unusually high behavior compared to historical patterns
- count thresholds help avoid overreacting to tiny-volume spikes
- sudden-appearance checks help detect entirely new failure patterns

For each category, the baseline window calculates:

- historical average share
- historical standard deviation
- 90th percentile share
- 95th percentile share
- historical average count

A category/date can be flagged by either share-based or count-based rules.

**Medium severity**
- current share > baseline average + 2.5 × standard deviation
- current share > baseline 90th percentile
- current count >= max(5, baseline average count × 2.5)

**High severity**
- current share > baseline average + 3 × standard deviation
- current share > baseline 95th percentile
- current count >= max(10, baseline average count × 3)
- category appeared after being absent for the last 30 baseline days

This combination helps the monitor work across different community volumes and category behaviors.

A flagged outlier means **needs review**, not “definitely broken.”  
Raw message review is suggested only when the outlier is both high severity and review-eligible.
""")


if outliers_df.empty:
    st.success("No outliers detected for the selected window.")
else:
    high_count = len(outliers_df[outliers_df["severity"] == "high"])
    medium_count = len(outliers_df[outliers_df["severity"] == "medium"])
    export_count = len(outliers_df[outliers_df["should_export"] == True])

    c1, c2, c3 = st.columns(3)
    c1.metric("High severity", high_count)
    c2.metric("Medium severity", medium_count)
    c3.metric("Raw exports suggested", export_count)

    display_cols = [
        "message_date",
        "category",
        "severity",
        "category_share",
        "baseline_mean_share",
        "baseline_p95",
        "severity_reasons",
        "should_export",
    ]
    outliers_display = outliers_df.rename(columns={
        "message_date": "Date",
        "category": "Category",
        "severity": "Severity",
        "category_share": "Current Share",
        "baseline_mean_share": "Baseline Avg",
        "baseline_p95": "High Threshold (P95)",
        "severity_reasons": "Why flagged",
        "should_export": "Needs review",
    })
    st.dataframe(
    outliers_display[
        [  # IMPORTANT: renamed column names here
            "Date",
            "Category",
            "Severity",
            "Current Share",
            "Baseline Avg",
            "High Threshold (P95)",
            "Why flagged",
            "Needs review",
        ]
    ],
    use_container_width=True,
    )

    csv = outliers_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download outlier report CSV",
        data=csv,
        file_name="outlier_report.csv",
        mime="text/csv",
    )
    st.markdown("""
### Raw message review

For high-severity, review-eligible outliers, the workflow surfaces example messages from the affected category/date.

In production, this helps analysts quickly inspect whether the spike reflects:
- expected behavior
- classification issues
- missing knowledge
- taxonomy gaps
- spam or moderation activity
""")

exportable_outliers_df = outliers_df[outliers_df["should_export"] == True].copy()

if exportable_outliers_df.empty:
    st.info("No high-severity review-eligible outliers for this window.")
else:
    exportable_outliers_df["label"] = (
        exportable_outliers_df["message_date"].astype(str)
        + " — "
        + exportable_outliers_df["category"]
    )

    selected_outlier_label = st.selectbox(
        "Select an outlier to review",
        exportable_outliers_df["label"].tolist(),
    )

    selected_outlier = exportable_outliers_df[
        exportable_outliers_df["label"] == selected_outlier_label
    ].iloc[0]

    selected_date = selected_outlier["message_date"]
    selected_category = selected_outlier["category"]

    review_messages_df = raw_messages_df[
        (raw_messages_df["request_date"] == selected_date)
        & (raw_messages_df["non_response_category"] == selected_category)
    ].copy()

    st.write(
        f"Showing raw message examples for **{selected_category}** on **{selected_date}**."
    )

    st.caption(
    "The raw message examples below are synthetic and anonymized. "
    "They are designed to simulate realistic investigation workflows without exposing real community data."
)

    review_cols = [
        "timestamp_utc",
        "channel_name",
        "demo_message_link",
        "user_request",
        "language",
        "bot_response",
    ]

    st.dataframe(
        review_messages_df[review_cols].head(50),
        use_container_width=True,
    )

    raw_csv = review_messages_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download raw review sample CSV",
        data=raw_csv,
        file_name=f"raw_review_{selected_date}_{selected_category}.csv",
        mime="text/csv",
    )

    st.markdown("###LLM investigation")

st.markdown("""
The app can use two investigation modes:

1. **Pre-generated demo review** — available for the default demo week.
2. **Live Gemini review** — generated on demand for the selected outlier.

The pre-generated review keeps the hosted demo usable without requiring API calls on every page load.

In a production version, this step is always handled by a live LLM call.
""")

matching_review_df = pregenerated_reviews_df[
    (pregenerated_reviews_df["message_date"] == str(selected_date))
    & (pregenerated_reviews_df["category"] == selected_category)
]

if matching_review_df.empty:
    st.info("No pre-generated review is available for this selected outlier. "
        "This is expected outside the default demo week. "
        "Use the live Gemini investigation button below to generate a fresh review.")
else:
    review = matching_review_df.iloc[0]

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Classification quality**")
        st.info(review["classification_quality"])

    with c2:
        st.markdown("**Sample size**")
        st.info(str(int(review["sample_size"])))

    st.markdown("**Main themes**")
    st.write(review["main_themes"])

    st.markdown("**Possible misclassifications to inspect**")
    st.write(review["possible_misclassifications"])

    st.markdown("**Cases where the bot may have been expected to answer**")
    st.write(review["should_bot_have_answered"])

    st.markdown("**Recommended action**")
    st.write(review["recommended_action"])

    st.markdown("**Slack-style summary**")
    st.code(review["slack_summary"])

st.markdown("### Optional — Live LLM-assisted investigation")

calls_left = MAX_GEMINI_CALLS_PER_SESSION - st.session_state.gemini_calls_used

st.caption(
    f"This makes a live Gemini API call for the selected outlier. "
    f"Calls left this session: {calls_left}."
)

run_live_gemini = st.button(
    "Run live Gemini investigation",
    disabled=calls_left <= 0,
)

if run_live_gemini:
    if calls_left <= 0:
        st.warning("Live Gemini investigation limit reached for this session.")
    else:
        with st.spinner("Running Gemini investigation..."):
            try:
                live_review = run_gemini_review(
                    raw_messages_df=review_messages_df,
                    category=selected_category,
                    message_date=selected_date,
                    max_messages=MAX_GEMINI_MESSAGES_PER_CALL,
                )

                st.session_state.gemini_calls_used += 1

                st.success("Live Gemini investigation completed.")

                st.markdown("**Classification quality**")
                st.info(live_review["classification_quality"])

                st.markdown("**Main themes**")
                st.write(live_review.get("main_themes", []))

                st.markdown("**Possible misclassifications**")
                st.dataframe(
                    pd.DataFrame(live_review.get("possible_misclassifications", [])),
                    use_container_width=True,
                )

                st.markdown("**Cases where the assistant may have been expected to answer**")
                st.dataframe(
                    pd.DataFrame(live_review.get("should_bot_have_answered", [])),
                    use_container_width=True,
                )

                st.markdown("**Recommended action**")
                st.write(live_review.get("recommended_action"))

                st.markdown("**Slack-style summary**")
                st.code(live_review.get("slack_summary", ""))

                st.caption(
                    f"Source: {live_review.get('review_source')} | "
                    f"Model: {live_review.get('model_name')} | "
                    f"Sample size: {live_review.get('sample_size')}"
                )

            except Exception as error:
                st.error(f"Live Gemini investigation failed: {error}")