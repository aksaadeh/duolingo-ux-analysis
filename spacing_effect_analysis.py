"""
RQ1 Analysis:
----------------------

1) Spacing curve with confidence intervals
2) Diminishing returns (effect size)
3) Spacing × difficulty interaction

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Paths
DATA_PATH = "outputs/processed_duolingo.csv"
TABLE_PATH = "outputs/tables"
PLOT_PATH = "outputs/plots"

os.makedirs(TABLE_PATH, exist_ok=True)
os.makedirs(PLOT_PATH, exist_ok=True)

# Loading the data
print("Loading processed data...")
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

# True lexeme-level spacing
df = df.sort_values(["user_id", "lexeme_id", "timestamp"])

df["prev_time_lexeme"] = (
    df.groupby(["user_id", "lexeme_id"])["timestamp"]
      .shift(1)
)

df["delta_lexeme"] = (
    df["timestamp"] - df["prev_time_lexeme"]
).dt.total_seconds()

df_spacing = df[df["delta_lexeme"].notna()].copy() # remove the first attempt

# creating the Spacing bins
bins = [0, 60, 300, 1800, 3600, 14400, 86400]
labels = ["<1m","1-5m","5-30m","30m-1h","1-4h","4-24h"]

df_spacing["spacing_bin"] = pd.cut(
    df_spacing["delta_lexeme"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# 1) SPACING CURVE + CONFIDENCE INTERVALS

grouped = df_spacing.groupby("spacing_bin")["next_p_recall"]

spacing_curve = grouped.agg(
    mean="mean",
    std="std",
    count="count"
).reset_index()

spacing_curve["ci_95"] = 1.96 * (
    spacing_curve["std"] / np.sqrt(spacing_curve["count"])
)

spacing_curve.to_csv(
    f"{TABLE_PATH}/rq1_spacing_curve_ci.csv",
    index=False
)

# Plot with error bars
plt.figure(figsize=(8,5))
plt.errorbar(
    spacing_curve["spacing_bin"],
    spacing_curve["mean"],
    yerr=spacing_curve["ci_95"],
    marker="o",
    capsize=5
)
plt.title("RQ1: Spacing vs Recall (95% CI)")
plt.xlabel("Spacing Interval")
plt.ylabel("Avg Next Recall")
plt.ylim(0.8, 1.0)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{PLOT_PATH}/rq1_spacing_curve_ci.png")
plt.close()


# 2) DIMINISHING RETURNS (Effect Size)

baseline = spacing_curve.loc[
    spacing_curve["spacing_bin"] == "<1m", "mean"
].values[0]

spacing_curve["absolute_gain"] = (
    spacing_curve["mean"] - baseline
)

spacing_curve["percent_gain"] = (
    spacing_curve["absolute_gain"] / baseline
) * 100

spacing_curve.to_csv(
    f"{TABLE_PATH}/rq1_spacing_effect_size.csv",
    index=False
)

plt.figure(figsize=(8,5))
plt.plot(
    spacing_curve["spacing_bin"],
    spacing_curve["percent_gain"],
    marker="o"
)
plt.axhline(0)
plt.title("RQ1: Percent Improvement vs Immediate Repetition")
plt.xlabel("Spacing Interval")
plt.ylabel("Percent Improvement (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{PLOT_PATH}/rq1_spacing_effect_size.png")
plt.close()

# 3) SPACING × DIFFICULTY INTERACTION

# Difficulty tiers
df_spacing["difficulty_tier"] = pd.cut(
    df_spacing["difficulty_ratio"],
    bins=[0, 0.75, 0.9, 1.0],
    labels=["Hard", "Medium", "Easy"]
)

difficulty_curve = (
    df_spacing
    .groupby(["spacing_bin","difficulty_tier"])
    ["next_p_recall"]
    .mean()
    .reset_index()
)

difficulty_curve.to_csv(
    f"{TABLE_PATH}/rq1_spacing_by_difficulty.csv",
    index=False
)

plt.figure(figsize=(8,5))
for tier in ["Hard","Medium","Easy"]:
    subset = difficulty_curve[
        difficulty_curve["difficulty_tier"] == tier
    ]
    plt.plot(
        subset["spacing_bin"],
        subset["next_p_recall"],
        marker="o",
        label=tier
    )

plt.legend()
plt.title("RQ1: Spacing Effect by Difficulty")
plt.xlabel("Spacing Interval")
plt.ylabel("Avg Next Recall")
plt.ylim(0.8, 1.0)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{PLOT_PATH}/rq1_spacing_by_difficulty.png")
plt.close()
