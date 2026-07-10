"""
Hybrid Model: Isolation Forest (Unsupervised) + XGBoost (Supervised)
======================================================================
WHY HYBRID, NOT JUST ONE MODEL:

Most student projects pick ONE algorithm. In the real world, utility
companies don't have perfectly labeled theft data (theft is often
UNDETECTED and unlabeled -- you only know about the thefts you caught).
So a purely supervised model trained only on known theft cases will miss
NEW/unseen theft patterns.

Our hybrid approach:
1. Isolation Forest (unsupervised) scores EVERY meter for how "weird"
   its consumption pattern is vs the general population -- catches
   novel/unseen theft patterns without needing labels.
2. XGBoost (supervised) learns the SPECIFIC signatures of the 4 known
   theft types from labeled data -- catches known patterns with high
   precision.
3. We combine both scores into a final ensemble risk score.

This mirrors how real fraud-detection systems in banking/utilities work:
supervised model for known fraud patterns + unsupervised model as a
safety net for new/emerging fraud patterns.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, precision_recall_curve, roc_auc_score,
    confusion_matrix, f1_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

FEATURE_COLS = [
    "avg_daily_consumption", "std_daily_consumption", "cv_consumption",
    "night_to_day_ratio", "morning_evening_peak_ratio", "weekday_weekend_ratio",
    "zero_reading_pct", "flat_line_pct", "rolling_std_of_daily_sum",
    "max_single_reading", "min_single_reading"
]

df = pd.read_csv("./meter_features.csv")
df = df.fillna(0)

X = df[FEATURE_COLS]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# MODEL 1: Isolation Forest (unsupervised anomaly detector)
# Trained WITHOUT labels -- only learns what "normal" looks like
# ---------------------------------------------------------------
honest_mask_train = (y_train == 0)
iso_forest = IsolationForest(
    n_estimators=200, contamination=0.15, random_state=42
)
iso_forest.fit(X_train_scaled[honest_mask_train.values])

# anomaly_score: higher = more anomalous (we flip sklearn's sign convention)
iso_scores_test = -iso_forest.score_samples(X_test_scaled)
iso_scores_test_norm = (iso_scores_test - iso_scores_test.min()) / (
    iso_scores_test.max() - iso_scores_test.min() + 1e-9
)

# ---------------------------------------------------------------
# MODEL 2: XGBoost (supervised classifier)
# ---------------------------------------------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=250, max_depth=4, learning_rate=0.05,
    scale_pos_weight=scale_pos_weight, eval_metric="logloss",
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)
xgb_probs_test = xgb_model.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------------
# ENSEMBLE: weighted combination of both scores
# ---------------------------------------------------------------
ALPHA = 0.7  # weight for supervised model (it's more precise on known patterns)
ensemble_score = ALPHA * xgb_probs_test + (1 - ALPHA) * iso_scores_test_norm
ensemble_pred = (ensemble_score > 0.5).astype(int)

print("=" * 60)
print("XGBoost only performance:")
print(classification_report(y_test, (xgb_probs_test > 0.5).astype(int)))
print("ROC-AUC:", roc_auc_score(y_test, xgb_probs_test))

print("=" * 60)
print("Hybrid Ensemble performance:")
print(classification_report(y_test, ensemble_pred))
print("ROC-AUC:", roc_auc_score(y_test, ensemble_score))
print("Confusion Matrix:\n", confusion_matrix(y_test, ensemble_pred))

# Save everything needed for the dashboard
joblib.dump(scaler, "./scaler.joblib")
joblib.dump(iso_forest, "./iso_forest.joblib")
joblib.dump(xgb_model, "./xgb_model.joblib")
joblib.dump(FEATURE_COLS, "./feature_cols.joblib")

print("\nModels saved.")
