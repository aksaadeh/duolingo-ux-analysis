"""
rq3_risk_prediction.py

RQ3 — Same-Day Risk Prediction

Goal:
Predict whether a learner will experience low recall later in the same day
using early-attempt behavioral signals.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


DATA_PATH = "outputs/processed_duolingo.csv"
TABLE_PATH = "outputs/tables"
FIG_PATH = "outputs/plots"

os.makedirs(FIG_PATH, exist_ok=True)


print("Loading data...")
df = pd.read_csv(DATA_PATH)

EARLY_N = 20

early = df[df["session_count"] <= EARLY_N]
late = df[df["session_count"] > EARLY_N]


# Aggregate EARLY features (already done, but we recompute cleanly)
early_features = (
    early.groupby("user_id")
    .agg(
        early_avg_recall=("p_recall", "mean"),
        early_avg_delta=("delta", "mean"),
        early_std_delta=("delta", "std"),
        early_avg_difficulty=("difficulty_ratio", "mean")
    )
    .reset_index()
)


# Aggregate LATE performance
late_performance = (
    late.groupby("user_id")
    .agg(
        late_avg_recall=("p_recall", "mean")
    )
    .reset_index()
)
data = pd.merge(early_features, late_performance, on="user_id")

# Define Target

THRESHOLD = 0.85

data["at_risk"] = (data["late_avg_recall"] < THRESHOLD).astype(int)

print("At-risk rate:", data["at_risk"].mean())

# Features & Split
X = data[
    [
        "early_avg_recall",
        "early_avg_delta",
        "early_std_delta",
        "early_avg_difficulty"
    ]
]

y = data["at_risk"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


# GRAPH 1 — ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("RQ3: ROC Curve (Risk Prediction)")
plt.savefig(f"{FIG_PATH}/rq3_roc_curve.png")
plt.close()


# GRAPH 2 — Feature Importance
short_names = [
    "Early Recall",
    "Avg Spacing",
    "Spacing Var",
    "Difficulty"
]

coeffs = model.coef_[0]

plt.figure()
plt.bar(short_names, coeffs)
plt.xticks(rotation=20)
plt.title("RQ3: Feature Importance")
plt.tight_layout()
plt.savefig(f"{FIG_PATH}/rq3_feature_importance.png")
plt.close()


# GRAPH 3 — Early Recall vs Risk
plt.figure()
plt.scatter(
    data["early_avg_recall"],
    data["late_avg_recall"],
    alpha=0.3
)
plt.xlabel("Early Avg Recall")
plt.ylabel("Late Avg Recall")
plt.title("RQ3: Early vs Late Recall")
plt.savefig(f"{FIG_PATH}/rq3_early_vs_late.png")
plt.close()


print("RQ3 analysis complete.")