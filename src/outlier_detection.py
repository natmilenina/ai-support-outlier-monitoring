import pandas as pd

from src.schema import NON_RESPONSE_CATEGORY_COLUMNS, EXPORT_ELIGIBLE_CATEGORIES


def prepare_category_daily_data(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    df = df.copy()
    df["message_date"] = pd.to_datetime(df["message_date"]).dt.date

    for _, row in df.iterrows():
        total = int(row["non_response_total"])

        for column_name, category_name in NON_RESPONSE_CATEGORY_COLUMNS.items():
            count = int(row[column_name])
            share = count / total if total > 0 else 0

            records.append(
                {
                    "message_date": row["message_date"],
                    "category": category_name,
                    "category_count": count,
                    "non_response_total": total,
                    "category_share": share,
                }
            )

    return pd.DataFrame(records)


def detect_outliers(
    baseline_df: pd.DataFrame,
    detection_df: pd.DataFrame,
) -> pd.DataFrame:
    baseline_long = prepare_category_daily_data(baseline_df)
    detection_long = prepare_category_daily_data(detection_df)

    results = []

    for _, row in detection_long.iterrows():
        category = row["category"]
        current_date = row["message_date"]
        current_count = row["category_count"]
        current_share = row["category_share"]

        baseline_cat = baseline_long[baseline_long["category"] == category]

        if baseline_cat.empty:
            continue

        mean_share = baseline_cat["category_share"].mean()
        std_share = baseline_cat["category_share"].std() or 0
        p90 = baseline_cat["category_share"].quantile(0.90)
        p95 = baseline_cat["category_share"].quantile(0.95)
        mean_count = baseline_cat["category_count"].mean()

        medium_share_threshold = mean_share + 2.5 * std_share
        high_share_threshold = mean_share + 3 * std_share

        medium_count_threshold = max(5, mean_count * 2.5)
        high_count_threshold = max(10, mean_count * 3)

        last_30 = baseline_cat.sort_values("message_date").tail(30)
        was_zero_last_30_days = (last_30["category_count"] == 0).all()

        medium_reasons = []
        high_reasons = []

        if current_share > high_share_threshold:
            high_reasons.append("share_gt_mean_3std")
        elif current_share > medium_share_threshold:
            medium_reasons.append("share_gt_mean_2_5std")

        if current_share > p95:
            high_reasons.append("share_gt_p95")
        elif current_share > p90:
            medium_reasons.append("share_gt_p90")

        if current_count >= high_count_threshold:
            high_reasons.append("count_gt_high_threshold")
        elif current_count >= medium_count_threshold:
            medium_reasons.append("count_gt_medium_threshold")

        if current_count > 0 and was_zero_last_30_days:
            high_reasons.append("sudden_appearance")

        if not high_reasons and not medium_reasons:
            continue

        severity = "high" if high_reasons else "medium"
        reasons = high_reasons + medium_reasons

        export_eligible = category in EXPORT_ELIGIBLE_CATEGORIES
        should_export = severity == "high" and export_eligible

        results.append(
            {
                "message_date": current_date,
                "category": category,
                "severity": severity,
                "category_count": current_count,
                "category_share": round(current_share, 4),
                "baseline_mean_share": round(mean_share, 4),
                "baseline_std_share": round(std_share, 4),
                "baseline_p90": round(p90, 4),
                "baseline_p95": round(p95, 4),
                "baseline_mean_count": round(mean_count, 2),
                "medium_share_threshold": round(medium_share_threshold, 4),
                "high_share_threshold": round(high_share_threshold, 4),
                "medium_count_threshold": round(medium_count_threshold, 2),
                "high_count_threshold": round(high_count_threshold, 2),
                "was_zero_last_30_days": was_zero_last_30_days,
                "severity_reasons": ", ".join(reasons),
                "export_eligible": export_eligible,
                "should_export": should_export,
            }
        )

    outliers_df = pd.DataFrame(results)

    if not outliers_df.empty:
        severity_order = {"high": 1, "medium": 2}
        outliers_df["sort"] = outliers_df["severity"].map(severity_order)

        outliers_df = (
            outliers_df.sort_values(
                by=["sort", "message_date", "category"],
                ascending=[True, False, True],
            )
            .drop(columns=["sort"])
            .reset_index(drop=True)
        )

    return outliers_df
