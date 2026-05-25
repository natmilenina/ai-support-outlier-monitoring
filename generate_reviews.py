from pathlib import Path

import pandas as pd

from src.outlier_detection import detect_outliers
from src.pregenerated_reviews import build_pregenerated_review
from src.windows import get_monitoring_windows


DATA_DIR = Path("data")

daily_totals_df = pd.read_csv(DATA_DIR / "synthetic_daily_totals.csv")
raw_messages_df = pd.read_csv(DATA_DIR / "synthetic_raw_messages.csv")

raw_messages_df["request_date"] = pd.to_datetime(raw_messages_df["request_date"]).dt.date

baseline_df, detection_df, metadata = get_monitoring_windows(
    df=daily_totals_df,
    detection_start_date="2026-05-24",
    detection_end_date="2026-05-30",
    baseline_days=90,
    minimum_baseline_days=30,
)

if not metadata["has_enough_baseline"]:
    raise ValueError("Not enough baseline data.")

outliers_df = detect_outliers(
    baseline_df=baseline_df,
    detection_df=detection_df,
)

exportable_outliers_df = outliers_df[outliers_df["should_export"] == True].copy()

review_rows = []

for _, outlier in exportable_outliers_df.iterrows():
    message_date = outlier["message_date"]
    category = outlier["category"]

    matching_raw_messages_df = raw_messages_df[
        (raw_messages_df["request_date"] == message_date)
        & (raw_messages_df["non_response_category"] == category)
    ]

    review_rows.append(
        build_pregenerated_review(
            raw_messages_df=matching_raw_messages_df,
            message_date=message_date,
            category=category,
        )
    )

reviews_df = pd.DataFrame(review_rows)
reviews_df.to_csv(DATA_DIR / "pregenerated_llm_reviews.csv", index=False)

print("Pre-generated reviews created:")
print(f"- reviews: {len(reviews_df)}")
print(DATA_DIR / "pregenerated_llm_reviews.csv")
