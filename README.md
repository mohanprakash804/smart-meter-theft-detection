# ⚡ Smart Meter Electricity Theft Detection (Hybrid ML)

A machine learning system that detects electricity theft from smart meter
consumption data using a **hybrid Isolation Forest + XGBoost ensemble**,
with **SHAP explainability** and an interactive **Streamlit dashboard**.

## Why this project is different

Most student projects in this space download a ready-made Kaggle dataset
and train a single classifier. This project instead:

1. **Simulates realistic smart meter data from scratch** — 480 households,
   365 days, 48 half-hourly readings/day (~8.4M records), with 4 distinct,
   *subtly* injected theft signatures (subtle on purpose — real theft
   doesn't look obviously different, which is what makes this a genuinely
   hard, interesting ML problem).
2. Uses a **hybrid model**: an unsupervised Isolation Forest (catches novel/
   unseen theft patterns without needing labels) combined with a supervised
   XGBoost classifier (precise on known theft signatures) — mirroring how
   real-world fraud detection systems in banking and utilities are designed.
3. Adds **SHAP explainability** so every flagged meter comes with a reason,
   not just a probability — turning it into a decision-support tool for
   field inspectors rather than a black box.
4. Ships as an **interactive Streamlit dashboard** for live demos.

## Theft types modeled

| Type | Real-world meaning | Data signature |
|---|---|---|
| METER_BYPASS | Wire routed around the meter for part of the day | Partial reading drop for a few hours |
| PARTIAL_TAMPER | Meter rigged to under-report | Consistently 75-88% of true usage |
| ILLEGAL_HOOKING | Neighbour's load hooked onto this line | Small random spikes off the household's normal profile |
| METER_REVERSAL | Meter tampered to freeze at night | Flat-lined readings during specific hours |

## Results

- **Hybrid ensemble F1-score:** ~0.99 on held-out test set
- **ROC-AUC:** ~0.997
- Full classification report and confusion matrix in `03_train_model.py` output

## Project structure

```
theft_detection/
├── 01_data_generator.py              # Simulates realistic smart meter data + theft injection
├── 02_feature_engineering.py         # Extracts theft-indicative statistical features
├── 03_train_model.py                 # Trains hybrid Isolation Forest + XGBoost ensemble
├── 04_explainability.py              # SHAP analysis for model interpretability
├── app.py                            # Streamlit interactive dashboard
├── Theft_Detection_Learning_Notebook.ipynb   # Full walkthrough with explanations (learn-by-doing)
└── README.md
```

## How to run

```bash
pip install pandas numpy scikit-learn xgboost shap streamlit matplotlib joblib pyarrow

# 1. Generate the synthetic dataset (~8.4M rows, takes a couple minutes)
python3 01_data_generator.py

# 2. Engineer features per meter
python3 02_feature_engineering.py

# 3. Train the hybrid model
python3 03_train_model.py

# 4. Generate SHAP explainability plots
python3 04_explainability.py

# 5. Launch the interactive dashboard
streamlit run app.py
```

Or open `Theft_Detection_Learning_Notebook.ipynb` to go through the entire
pipeline step-by-step with explanations of every concept used.

## Tech stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · SHAP · Streamlit · Matplotlib

## Resume description (suggested)

> Built a hybrid anomaly-detection + classification pipeline (Isolation
> Forest + XGBoost) to detect electricity theft from simulated smart meter
> time-series data (8M+ readings), engineered domain-specific features
> capturing 4 real-world theft signatures, added SHAP-based explainability,
> and deployed an interactive Streamlit dashboard. Achieved ~99% F1-score.

## Possible extensions (good "future work" talking points in interviews)

- Replace hand-crafted features with an LSTM/temporal CNN on raw time series
- Active learning loop: feed confirmed field-inspection results back into training
- Real dataset integration (e.g. SGCC State Grid Corporation of China dataset)
