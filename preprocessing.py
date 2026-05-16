"""
preprocessing.py

Purpose:
--------
Create clean, modeling-ready datasets focused on
short-term learning dynamics (within-day behavior).

Supports:
- spacing effect analysis
- fatigue analysis
- at-risk learner modeling

Outputs:
--------
outputs/processed_duolingo.csv
outputs/tables/spacing_bins.csv
outputs/tables/user_early_features.csv
"""

import os
import pandas as pd
import numpy as np

RAW_PATH = "data/duolingo_data.csv"
OUT_PATH = "outputs/processed_duolingo.csv"
TABLE_PATH = "outputs/tables"

os.makedirs(TABLE_PATH, exist_ok=True)

print("Loading raw data...")
df = pd.read_csv(RAW_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
df = df.sort_values(["user_id", "timestamp"])

df["session_count"] = (
    df.groupby("user_id")
      .cumcount() + 1
)

df["prev_time"] = df.groupby("user_id")["timestamp"].shift(1)

df["delta"] = (
    df["timestamp"] - df["prev_time"]
).dt.total_seconds()

df["delta"] = df["delta"].fillna(0)

df["log_delta"] = np.log1p(df["delta"])

df["next_p_recall"] = (
    df.groupby(["user_id", "lexeme_id"])["p_recall"]
      .shift(-1)
)

df["next_recall_flag"] = (df["next_p_recall"] < 0.8).astype("float")

df["difficulty_ratio"] = (
    df["history_correct"] /
    df["history_seen"].replace(0, np.nan)
)

df["difficulty_ratio"] = df["difficulty_ratio"].fillna(0)

bins = [0, 60, 300, 1800, 3600, 14400, 86400, 604800]
labels = ["<1m","1-5m","5-30m","30m-1h","1-4h","4-24h","1-7d"]
df["spacing_bin"] = pd.cut(df["delta"], bins=bins, labels=labels, right=True)


spacing_table = (
    df.groupby("spacing_bin")["next_p_recall"]
      .mean()
      .reset_index()
)

spacing_table.to_csv(
    f"{TABLE_PATH}/spacing_bins.csv",
    index=False
)

EARLY_N = 20
early = (
    df[df["session_count"] <= EARLY_N]
      .groupby("user_id")
      .agg(
          early_avg_recall=("p_recall", "mean"),
          early_avg_delta=("delta", "mean"),
          early_std_delta=("delta", "std"),
          early_avg_difficulty=("difficulty_ratio", "mean")
      )
      .reset_index()
)

early.to_csv(
    f"{TABLE_PATH}/user_early_features.csv",
    index=False
)


# Save
df.drop(columns=["prev_time"]).to_csv(OUT_PATH, index=False)

print("Preprocessing complete.")
print(f"Saved: {OUT_PATH}")
print(f"Saved tables in: {TABLE_PATH}")
