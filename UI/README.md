# Hanoi Weather ML — Daily & Hourly Forecast UI

This project provides a **machine-learning–powered weather forecasting system for Hanoi**, featuring:

- **Daily forecasting** — predicts the next **5 days**  
- **Hourly forecasting** — predicts temperatures at **1h, 6h, 12h, 24h** into the future  

A Streamlit-based UI displays real historical temperatures, 5-day forecasts, and full hourly breakdowns for each predicted day.

---

## Project Structure

```bash
.
├── daily.py               # Full ML pipeline for daily forecasting + artifacts save/load
├── hourly.py              # Full ML pipeline for hourly forecasting + artifacts save/load
├── weather_backend.py     # Backend logic for UI (daily + hourly)
├── app_weather_ui.py      # Streamlit application
├── Hanoi Daily.csv        # (Optional) local daily dataset. If missing → auto-download
├── artifacts/             # Auto-generated model artifacts after training
│   ├── df_daily.parquet
│   ├── X_features.parquet
│   ├── lgbm_models.pkl
│   ├── meta.pkl
│   ├── df_hourly_clean.parquet
│   ├── lgbm_hourly_models.pkl
│   ├── hourly_ohe.pkl
│   └── hourly_meta.pkl
└── README.md

**Note:**

* `artifacts/` is created automatically after running `daily.py` and `hourly.py`.
* If `Hanoi Daily.csv` is present, the pipeline uses it. Otherwise it loads the dataset from GitHub.


---

## Model & Pipeline Overview

### **Daily Forecasting (daily.py)**

* *Target*: temp (daily mean temperature)
* *Forecast Horizon*: next *5 days*
* *Feature Engineering*:

  * Time features: year, month, day_of_year, day_of_week, quarter
  * Cyclical encoding: month, day_of_year, day_of_week
  * Lag features: lags over [1, 3, 5, 7]
  * Rolling windows: [7, 14, 28, 56, 84]
  * Derived features:

    * temp_range = tempmax - tempmin,
    * dewpoint_depression = temp - dew, etc.
* *Temporal splitting*:

  * Train 70% — Validation 15% — Test 15%
* *Model*:

  * One *LightGBM regressor per horizon* (t+1 → t+5)
  * Trained *strictly on the train split*
* *Artifacts*:

  * df_daily.parquet — cleaned dataset
  * X_features.parquet — full FE output (for backend)
  * lgbm_models.pkl — {h: model}
  * meta.pkl — metadata for inference

### Inference Behavior

predict_for_date() performs *fresh feature engineering* on the full dataset (without dropping edge dates), making it possible to forecast even for the first/last days that were removed during training.

---

### **Hourly Forecasting (hourly.py)**

* *Target*: temp (hourly temperature)
* *Horizons*: [1, 6, 12, 24] hours ahead
* *Feature Engineering*:

  * Cyclical: hour, day_of_week, day_of_year, month, winddir
  * Lag features: [1, 2, 3, 6, 24]
  * Rolling windows: [3, 6, 12, 24]
  * Derived features:

    * dewpoint_depression
    * wind_speed_squared
    * wind_chill
    * wind_ratio
    * severe_proxy
    * heat_index_approx
* *Data Cleaning*:

  * Add season
  * Impute windspeed, winddir
  * Fill precip and visibility
  * Smart solar variable imputation (night/day logic)
* *OHE*:

  * Encode icon and season if present
* *Model*:

  * One *LightGBM regressor per horizon*
* *Artifacts*:

  * df_hourly_clean.parquet
  * lgbm_hourly_models.pkl
  * hourly_ohe.pkl
  * hourly_meta.pkl

### Inference Logic

Hourly forecasting uses the exact window approach of the original notebook:

1. Extract last *ROWS_NEEDED* rows based on max lag/rolling window
2. Apply OHE → FE inference → take *last row* (tail(1))
3. Predict with models at [1, 6, 12, 24]

---

## Installation

### 1. Requirements

requirements.txt:

text
numpy
pandas
lightgbm
scikit-learn
joblib
streamlit
altair

Install dependencies:

pip install -r requirements.txt

---

## 🏋️‍♂️ Train the Models

Run:

python daily.py
python hourly.py

Artifacts will appear in artifacts/.

---

## Run the UI

Start Streamlit:

streamlit run app_weather_ui.py

### UI Features

* Select any valid historical date
* View *actual daily temperature*
* View *5-day forecast*
* Click a forecasted day to view:

  * Full *hourly* predicted temperature curve
  * X-axis always from *0 → 23* with proper units

---
