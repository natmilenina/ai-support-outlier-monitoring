from datetime import timedelta

import pandas as pd


def get_monitoring_windows(
    df: pd.DataFrame,
    detection_start_date: str,
    detection_end_date: str,
    baseline_days: int = 90,
    minimum_baseline_days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = df.copy()
    df["message_date"] = pd.to_datetime(df["message_date"]).dt.date

    detection_start = pd.to_datetime(detection_start_date).date()
    detection_end = pd.to_datetime(detection_end_date).date()

    baseline_end = detection_start - timedelta(days=1)
    baseline_start = detection_start - timedelta(days=baseline_days)

    baseline_df = df[
        (df["message_date"] >= baseline_start)
        & (df["message_date"] <= baseline_end)
    ].copy()

    detection_df = df[
        (df["message_date"] >= detection_start)
        & (df["message_date"] <= detection_end)
    ].copy()

    actual_baseline_days = baseline_df["message_date"].nunique()
    actual_detection_days = detection_df["message_date"].nunique()

    metadata = {
        "baseline_start": baseline_start,
        "baseline_end": baseline_end,
        "detection_start": detection_start,
        "detection_end": detection_end,
        "requested_baseline_days": baseline_days,
        "actual_baseline_days": actual_baseline_days,
        "actual_detection_days": actual_detection_days,
        "minimum_baseline_days": minimum_baseline_days,
        "has_enough_baseline": actual_baseline_days >= minimum_baseline_days,
        "has_detection_data": actual_detection_days > 0,
    }

    return baseline_df, detection_df, metadata
