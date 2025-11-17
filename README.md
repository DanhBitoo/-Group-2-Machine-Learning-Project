Here is a comprehensive, professional **README.md** template written in English for your project. It is structured based on the code and datasets you provided (focusing on LightGBM, Optuna, ClearML, and ONNX integration).

You can copy the code block below directly into your GitHub repository.

-----

## Hanoi Temperature Forecasting Project 

   

###  Overview

This project applies Machine Learning techniques to forecast the temperature in **Hanoi, Vietnam**. It handles two distinct time-series forecasting tasks:

1.  **Daily Forecasting:** Predicting average daily temperatures for the next 1 to 5 days ($t+1$ to $t+5$).
2.  **Hourly Forecasting:** Predicting hourly temperature fluctuations.

The project demonstrates a complete MLOps workflow, including data preprocessing, feature engineering, hyperparameter tuning with **Optuna**, experiment tracking with **ClearML**, and model deployment optimization using **ONNX**.

###  Dataset

The models are trained on historical meteorological data for Hanoi (approx. 2015–2025).

  * **`Hanoi Daily.csv`**: Aggregated daily metrics including `temp`, `humidity`, `precip`, `windspeed`, `pressure`, etc.
  * **`Hanoi Hourly.csv`**: Granular hourly data with similar features.

###  Tech Stack

  * **Language:** Python
  * **Data Processing:** Pandas, NumPy
  * **Visualization:** Matplotlib, Seaborn
  * **Machine Learning:** LightGBM, XGBoost, Random Forest, Scikit-Learn
  * **Optimization & Tracking:** Optuna, ClearML
  * **Deployment/Inference:** ONNX (Open Neural Network Exchange)

###  Key Features & Methodology

#### 1\. Data Preprocessing

  * Handling missing values and outliers.
  * Datetime conversion and extraction of temporal features (day of year, month, hour, etc.).
  * Feature Engineering: Lag features, rolling windows, and statistical aggregations.

#### 2\. Model Training strategies

  * **Daily Prediction:**
      * Utilizes **LightGBM** as the primary regressor.
      * Multi-step forecasting strategy (predicting $t+1, t+2, ... t+5$).
      * **ONNX Export:** The trained LightGBM model is converted to ONNX format to ensure faster inference speeds and portability.
  * **Hourly Prediction:**
      * Compares multiple algorithms: **RandomForest, LightGBM, and XGBoost**.
      * Uses **Optuna** for automated hyperparameter optimization.
      * Integrates **ClearML** to track experiments, compare runs, and log metrics (MAE, RMSE, R2).

#### 3\. Evaluation

Models are evaluated using standard regression metrics:

  * **MAE** (Mean Absolute Error)
  * **RMSE** (Root Mean Squared Error)
  * **$R^2$ Score**

###  Results Highlight

  * **Daily Model:** The LightGBM model demonstrated high stability across all forecast horizons ($t+1$ to $t+5$).
  * **ONNX Conversion:** The ONNX version of the model retained near-identical accuracy to the original model (differences within negligible range), validating it for production use.

###  Installation & Usage

1.  **Clone the repository**

    ```bash
    git clone https://github.com/DanhBitoo/hanoi-temp-prediction.git
    cd hanoi-temp-prediction
    ```

2.  **Install dependencies**

    ```bash
    pip install pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn optuna clearml onnxruntime
    ```

3.  **Run the Notebooks**

      * Open `Hanoi_daily_temperature_prediction.ipynb` for daily forecasting and ONNX export.
      * Open `Hanoi_Hourly_temperature_prediction.ipynb` for hourly forecasting and Optuna/ClearML experiments.

###  Project Structure

```
├── data/
│   ├── Hanoi Hourly.csv
│   └── Hanoi Daily.csv
├── notebooks/
│   ├── Hanoi_Hourly_temperature_prediction.ipynb
│   └── Hanoi_daily_temperature_prediction.ipynb
├── models/
│   └── (Saved models or ONNX files will appear here)
├── README.md
└── requirements.txt
```

###  Future Work

  * Deploy the ONNX model as a REST API using FastAPI.
  * Integrate deep learning models (LSTM/GRU) for comparison.
  * Build a Streamlit dashboard to visualize real-time forecasts.

