"""
Explainability with SHAP
=========================
A "black box says 87% theft probability" is not useful to a field
inspector who has to decide whether to send a technician to a house.
SHAP tells us WHICH features drove that specific prediction, so the
output can say: "flagged mainly due to unusually high zero-reading
percentage and abnormal night-time consumption pattern."

This turns the project from "just a classifier" into a decision-support
tool -- a much stronger resume/interview talking point.
"""

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

df = pd.read_csv("./meter_features.csv").fillna(0)
scaler = joblib.load("./scaler.joblib")
xgb_model = joblib.load("./xgb_model.joblib")
FEATURE_COLS = joblib.load("./feature_cols.joblib")

X = df[FEATURE_COLS]
X_scaled = scaler.transform(X)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_scaled)

# Global feature importance plot
plt.figure()
shap.summary_plot(shap_values, X, feature_names=FEATURE_COLS, show=False)
plt.tight_layout()
plt.savefig("./shap_summary.png", dpi=150)
plt.close()
print("Saved global SHAP summary plot -> shap_summary.png")

# Save the explainer for use in the Streamlit app
joblib.dump(explainer, "./shap_explainer.joblib")
print("Saved SHAP explainer for dashboard use.")
