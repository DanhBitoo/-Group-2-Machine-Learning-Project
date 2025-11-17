# Hanoi Weather ML – Daily & Hourly Forecast UI

Dự án này xây dựng một **ứng dụng dự báo thời tiết cho Hà Nội** sử dụng mô hình machine learning (LightGBM) cho cả:

- **Daily**: dự báo nhiệt độ trung bình của **5 ngày tiếp theo**  
- **Hourly**: dự báo nhiệt độ theo giờ ở các horizon **1h, 6h, 12h, 24h**  

Frontend là một **Streamlit app** với giao diện giống app thời tiết: hiển thị nhiệt độ thực tế của ngày được chọn, forecast 5 ngày sau, và khi click vào từng ngày forecast sẽ hiện chi tiết dự báo theo giờ.

---

##  Cấu trúc repo

```bash
.
├── daily.py               # Pipeline load/FE/train LightGBM cho daily + lưu artifacts
├── hourly.py              # Pipeline load/FE/train LightGBM cho hourly + lưu artifacts
├── weather_backend.py     # Backend dùng artifacts để phục vụ UI (daily + hourly)
├── app_weather_ui.py      # Streamlit app (UI)
├── Hanoi Daily.csv        # (tuỳ chọn) file dữ liệu daily local, nếu không có sẽ tải từ GitHub
├── artifacts/             # Thư mục chứa artifacts sau khi train (auto tạo)
│   ├── df_daily.parquet
│   ├── X_features.parquet
│   ├── lgbm_models.pkl
│   ├── meta.pkl
│   ├── df_hourly_clean.parquet
│   ├── lgbm_hourly_models.pkl
│   ├── hourly_ohe.pkl
│   └── hourly_meta.pkl
└── README.md

**Lưu ý:**

* `artifacts/` được tạo tự động sau khi chạy `daily.py` và `hourly.py`.
* Nếu tồn tại `Hanoi Daily.csv` trong thư mục gốc, `daily.py` sẽ ưu tiên dùng file này; nếu không sẽ tải từ GitHub.


---

##  Mô hình & pipeline

### Daily (daily.py)

* *Target*: temp (nhiệt độ trung bình ngày)
* *HORIZON*: 5 (dự báo 5 ngày tiếp theo)
* *Feature engineering*:

  * Time features: year, month, day_of_year, day_of_week, quarter
  * Cyclical encoding: month, day_of_year, day_of_week
  * Lag features: lags theo ngày cho các biến như humidity, dew, precip, windspeed,…
  * Rolling windows: mean/std cho các cửa sổ [7, 14, 28, 56, 84]
  * Derived features: temp_range, `dewpoint_depression`…
* *Split theo thời gian*:

  * Train ~70%, Val 15%, Test 15%
* *Model*:

  * 1 LGBMRegressor cho *mỗi horizon* t+1`…t+5`
  * Train *chỉ trên tập train*
* *Artifacts*:

  * df_daily.parquet: dữ liệu daily sau preprocess (index = datetime)
  * X_features.parquet: full features (train + val + test) cho mục đích backend/debug
  * lgbm_models.pkl: dict {h: model} với h = 1..5
  * meta.pkl: meta info (TARGET_COL, HORIZON, feature_cols, …)

Hàm quan trọng dùng cho UI:

* `load_daily_artifacts(...)`
* `predict_for_date(origin_date, horizon, artifact_dir=ARTIFACT_DIR)`
* `get_actual_and_forecast_for_ui(...)`


predict_for_date **FE lại trên toàn bộ df_daily** (không dropna, không tạo target) nên có thể dự báo cho *bất kỳ ngày nào có trong dữ liệu gốc*, kể cả những ngày đầu/cuối bị cắt khi train.

---

### Hourly (hourly.py)

* *Target*: temp (nhiệt độ theo giờ)
* *HORIZON*: [1, 6, 12, 24] (giờ phía trước)
* *Feature engineering*:

  * Time + cyclical: hour, day_of_week, day_of_year, month, winddir
  * Lag features: LAGS = [1, 2, 3, 6, 24]
  * Rolling windows: mean/std cho [3, 6, 12, 24]
  * Derived features: dewpoint_depression, wind_speed_squared, wind_chill, wind_ratio, severe_proxy, heat_index_approx
* *Xử lý dữ liệu & missing*:

  * Thêm season từ month
  * FFill windspeed, impute winddir theo (name, season, hour) nếu có
  * precip → fill 0; visibility → ffill + bfill
  * Impute solarradiation/solarenergy/uvindex theo logic đêm/ngày & cloudcover
  * Tạo precip_flag
* *One-hot encoding*:

  * OHE cho icon, season (nếu tồn tại)
* *Split theo thời gian*:

  * Train 70%, Val 15%, Test 15%
* *Model*:

  * 1 LGBMRegressor cho mỗi horizon trong [1, 6, 12, 24]
  * Train *chỉ trên tập train*
* *Artifacts*:

  * df_hourly_clean.parquet: dữ liệu hourly sau preprocess
  * lgbm_hourly_models.pkl: dict {h: model}
  * hourly_ohe.pkl: OneHotEncoder đã fit
  * hourly_meta.pkl: meta & metrics

Hàm quan trọng cho UI:

* `load_hourly_artifacts(...)`
* `predict_hourly_multi_horizon_for_timestamp(origin_ts, artifact_dir=...)`


Khi dự báo tại một origin_ts:

1. Lấy *history ≤ origin_ts*, giữ lại ROWS_NEEDED dòng cuối (dựa trên lag/rolling lớn nhất).
2. Áp dụng OHE + FE inference (create_hourly_features_for_predictions).
3. Lấy *hàng cuối cùng* làm input (giống đoạn X_pred_ready = df_pred.tail(1) trong notebook).
4. Chạy qua các model LightGBM ở horizon [1, 6, 12, 24].

---

## Cài đặt & chạy

### 1. Yêu cầu

* Python 3.10+ (khuyến nghị)
* Các thư viện chính:

  * numpy
  * pandas
  * lightgbm
  * scikit-learn
  * joblib
  * streamlit
  * altair

Bạn có thể tạo file requirements.txt như sau:

text
numpy
pandas
lightgbm
scikit-learn
joblib
streamlit
altair

Rồi cài:

pip install -r requirements.txt

---

### 2. Train & build artifacts

Trong thư mục repo:

# Train daily model + lưu artifacts
python daily.py

# Train hourly model + lưu artifacts
python hourly.py

Sau khi chạy xong, thư mục artifacts/ sẽ được tạo với đầy đủ file cần cho UI.

---

### 3. Chạy ứng dụng UI (Streamlit)

streamlit run app_weather_ui.py

Sau khi chạy:

* UI sẽ hiển thị màn hình *chọn ngày* (date picker).
* Ngày mặc định thường là *ngày cuối cùng* có dữ liệu trong df_daily.
* Khi chọn một ngày D:

  * Bên trái: *nhiệt độ thực tế* của ngày D (daily) + mô tả tổng quan.
  * Dưới: các *card dự báo 5 ngày tiếp theo* (D+1`…D+5`) với nhiệt độ dự báo.
  * Bên phải: biểu đồ hourly thật của ngày D (0–23h).
* Khi *click vào một card ngày forecast* (ví dụ D+3):

  * UI gọi backend hourly → hiển thị chi tiết *dự báo theo giờ* của ngày đó (biểu đồ 0–23h).

---

##  Lưu ý kỹ thuật

* *Train vs. Inference*:

  * Daily & hourly đều *train chỉ trên tập train*, validate/test riêng.
  * Khi inference, *không tái dùng trực tiếp X_train/X_full để tra index như trước*, mà FE lại toàn bộ (hoặc fe trên window lịch sử) để duy trì tính đúng cho cả những ngày/giờ vốn bị drop trong quá trình train.
* *Xử lý dtime*:

  * df_daily sau preprocess có index = datetime (kiểu Timestamp).
  * df_hourly_clean giữ cột datetime dạng datetime64[ns].
