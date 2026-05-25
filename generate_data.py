from pathlib import Path

from src.synthetic_data import generate_all_synthetic_data


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

daily_totals_df, daily_by_channel_df, raw_messages_df = generate_all_synthetic_data()

daily_totals_df.to_csv(DATA_DIR / "synthetic_daily_totals.csv", index=False)
daily_by_channel_df.to_csv(DATA_DIR / "synthetic_daily_by_channel.csv", index=False)
raw_messages_df.to_csv(DATA_DIR / "synthetic_raw_messages.csv", index=False)

print("Synthetic data created:")
print(f"- daily totals: {len(daily_totals_df)} rows")
print(f"- daily by channel: {len(daily_by_channel_df)} rows")
print(f"- raw messages: {len(raw_messages_df)} rows")
