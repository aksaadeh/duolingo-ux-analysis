"""
RQ2: Diminishing Returns Within a 24-Hour Period

Research Question:
Does increasing daily practice intensity within a 24-hour period
exhibit diminishing returns in recall performance?
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

DATA_PATH = "outputs/processed_duolingo.csv"
FIG_PATH = "outputs/plots"
os.makedirs(FIG_PATH, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Aggregate per user (24h window)

print("Aggregating per user...")

daily = (
    df.groupby("user_id")
      .agg(
          total_attempts=("p_recall", "count"),
          mean_recall=("p_recall", "mean")
      )
      .reset_index()
)

daily = daily[
    daily["total_attempts"] <= daily["total_attempts"].quantile(0.99)
]


# GRAPH 1: Scatter + LOWESS Smooth
print("Creating scatter + LOWESS plot...")

lowess = sm.nonparametric.lowess(
    daily["mean_recall"],
    daily["total_attempts"],
    frac=0.3
)

plt.figure()
plt.scatter(daily["total_attempts"], daily["mean_recall"], alpha=0.1)
plt.plot(lowess[:, 0], lowess[:, 1])
plt.xlabel("Total Attempts (24h)")
plt.ylabel("Mean Recall Probability")
plt.title("RQ2: Practice Intensity vs Recall (LOWESS)")
plt.tight_layout()
plt.savefig(f"{FIG_PATH}/rq2_lowess_scatter.png")
plt.close()


# GRAPH 2: Binned Dose-Response Curve
print("Creating binned dose-response plot...")

bin_width = 10
bins = np.arange(0, daily["total_attempts"].max() + bin_width, bin_width)
daily["attempt_bin"] = pd.cut(daily["total_attempts"], bins=bins)

binned = (
    daily.groupby("attempt_bin")
         .agg(
             mean_attempts=("total_attempts", "mean"),
             mean_recall=("mean_recall", "mean"),
             count=("mean_recall", "count"),
             std=("mean_recall", "std")
         )
         .dropna()
         .reset_index()
)

binned["se"] = binned["std"] / np.sqrt(binned["count"])
binned["ci_lower"] = binned["mean_recall"] - 1.96 * binned["se"]
binned["ci_upper"] = binned["mean_recall"] + 1.96 * binned["se"]

plt.figure()
plt.plot(binned["mean_attempts"], binned["mean_recall"])
plt.fill_between(
    binned["mean_attempts"],
    binned["ci_lower"],
    binned["ci_upper"],
    alpha=0.2
)
plt.xlabel("Total Attempts (24h)")
plt.ylabel("Mean Recall Probability")
plt.title("RQ2: Binned Dose–Response with 95% CI")
plt.tight_layout()
plt.savefig(f"{FIG_PATH}/rq2_binned_curve.png")
plt.close()

# GRAPH 3: Marginal Returns (Slope Change)
print("Computing marginal returns...")
binned = binned.sort_values("mean_attempts")
binned["marginal_return"] = (
    binned["mean_recall"].diff() /
    binned["mean_attempts"].diff()
)

plt.figure()
plt.plot(binned["mean_attempts"], binned["marginal_return"])
plt.axhline(0)
plt.xlabel("Total Attempts (24h)")
plt.ylabel("Marginal Change in Recall")
plt.title("RQ2: Marginal Returns Across Practice Intensity")
plt.tight_layout()
plt.savefig(f"{FIG_PATH}/rq2_marginal_returns.png")
plt.close()

print("RQ2 diminishing returns analysis complete.")
print(f"Figures saved to: {FIG_PATH}")

# Quantitative Analysis 

print("\nQuantitative analysis of diminishing returns...")

# Linear regression

X = sm.add_constant(daily["total_attempts"])
y = daily["mean_recall"]

model = sm.OLS(y, X).fit()

print("\nLinear Effect of Practice Intensity")
print("-----------------------------------")
print(model.summary())

beta = model.params["total_attempts"]

print(f"\nEffect per 10 additional attempts: {beta * 10:.5f}")

# Low vs High Practice Groups

low_group = daily[daily["total_attempts"] <= 50]
high_group = daily[daily["total_attempts"] >= 150]

low_mean = low_group["mean_recall"].mean()
high_mean = high_group["mean_recall"].mean()

print("\nLow vs High Practice Comparison")
print("--------------------------------")

print(f"Low practice mean recall:  {low_mean:.4f}")
print(f"High practice mean recall: {high_mean:.4f}")
print(f"Difference:                {high_mean - low_mean:.4f}")

# Plateau Detection
# compute slope of LOWESS curve
x_lowess = lowess[:, 0]
y_lowess = lowess[:, 1]

slopes = np.diff(y_lowess) / np.diff(x_lowess)

# plateau defined when slope becomes very small
threshold = 0.00001

plateau_index = np.where(np.abs(slopes) < threshold)[0]

if len(plateau_index) > 0:
    plateau_attempt = x_lowess[plateau_index[0]]
    print("\nPlateau Detection")
    print("-----------------")
    print(f"Recall performance plateaus around ~{int(plateau_attempt)} attempts.")
else:
    print("\nNo clear plateau detected.")

print("\nRQ2 analysis complete.")
