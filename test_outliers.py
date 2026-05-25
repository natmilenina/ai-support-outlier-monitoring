import pandas as pd

from src.windows import get_monitoring_windows
from src.outlier_detection import detect_outliers


df = pd.read_csv("data/synthetic_daily_totals.csv")

baseline_df, detection_df, metadata = get_monitoring_windows(
    df=df,
    detection_start_date="2026-05-24",
    detection_end_date="2026-05-30",
    baseline_days=90,
    minimum_baseline_days=30,
)

print("Window metadata:")
for key, value in metadata.items():
    print(f"- {key}: {value}")

print("\nBaseline rows:", len(baseline_df))
print("Detection rows:", len(detection_df))

if not metadata["has_enough_baseline"]:
    raise ValueError("Not enough baseline data to run detection.")

if not metadata["has_detection_data"]:
    raise ValueError("No detection data found.")

outliers_df = detect_outliers(
    baseline_df=baseline_df,
    detection_df=detection_df,
)

print("\nOutliers found:", len(outliers_df))

if outliers_df.empty:
    print("No outliers detected.")
else:
    print(
        outliers_df[
            [
                "message_date",
                "category",
                "severity",
                "category_count",
                "category_share",
                "baseline_mean_share",
                "baseline_p95",
                "severity_reasons",
                "should_export",
            ]
        ].to_string(index=False)
    )
