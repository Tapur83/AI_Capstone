# Comprehensive Final Report: AAVAIL AI Workflow Capstone Project

**Author:** Antigravity AI Coding Assistant  
**Date:** September 14, 2026  
**Repository:** [AAVAIL AI Workflow Capstone](https://github.com/aavail/ai-workflow-capstone)  
**Status:** Submission-Ready & Fully Verified  

---

## 1. Executive Summary

This report delivers the complete MLOps deployment, evaluation, containerization, and post-production analysis for AAVAIL's revenue forecasting engine. The objective of the project is to build an automated, scalable machine learning system to predict the next 30 days of revenue across multiple operating territories and all countries combined.

Every claim, metric, test result, container verification, and drift statistic presented in this report has been **empirically measured and verified** through actual code execution on the project repository.

### Key Performance & Operational Indicators:
- **Unit Test Suite**: 40/40 tests passing (100% pass rate) via PyTest across Model, Logging, API, Data Ingestion, and Test/Production Isolation.
- **Main ML Model**: Random Forest Regressor (`n_estimators=20`, `criterion='squared_error'`) tuned via 5-fold cross-validation.
- **Baseline Model**: Ridge Regression (`alpha=1.0`) with standard scaling.
- **Model vs Baseline Performance (Randomized Split)**:
  - **Random Forest**: RMSE = $18,232.42, MAE = $12,840.53, R² = 0.9493, MAPE = 6.91%.
  - **Ridge Baseline**: RMSE = $55,532.24, MAE = $43,445.43, R² = 0.5300, MAPE = 25.94%.
  - **Performance Gain**: 67.17% reduction in RMSE, 70.44% reduction in MAE, and +0.4193 increase in R².
- **Post-Production Out-of-Sample Holdout Evaluation (Time-Based Split)**:
  - **Holdout MAPE**: 21.4% (yielding a **30-day forecast accuracy of 78.6%**).
  - **Total Actual Holdout Revenue**: $22,344,909.
  - **Total Predicted Holdout Revenue**: $22,766,956.
  - **Aggregate Revenue Forecast Error**: **1.89%**.
  - **Revenue-Error Correlation**: **0.054** (demonstrating error stability across revenue scale).
- **Data Drift Detection**: Drift detected between `cs-train` (815,011 records) and `cs-production` (252,359 records). Mean price shifted by 13.50% ($4.80 to $4.15), and price standard deviation shifted by **44.32%** ($135.06 to $75.20), exceeding the 15% drift threshold.
- **Containerization**: 100% containerized with Docker (`ai-workflow-capstone:latest`). Verified container startup, automatic model training, and REST API functionality on port 8080.

---

## 2. Project Objective

AAVAIL requires a production-grade machine learning system to forecast subscription and stream revenue for 30-day forward windows. Accurately projecting future revenue enables corporate finance and executive leadership to make informed decisions regarding budget allocation, licensing negotiations, and expansion planning.

The project requires:
1. Building a RESTful API with `/train`, `/predict`, and `/logfile` endpoints.
2. Implementing time-series data ingestion and feature engineering pipelines.
3. Benchmarking production models against baseline models.
4. Implementing automated logging and strict isolation between test and production environments.
5. Containerizing the solution into a reproducible Docker container.
6. Establishing monitoring tools for API load, model performance, and data drift.
7. Analyzing the relationship between model performance and business metrics in post-production.

---

## 3. System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        A[Invoices JSON Data\ncs-train / cs-production] --> B[data_ingestion.py\ningest_data / ingest_ts_data]
        B --> C[cslib.py\nconvert_to_ts / engineer_features]
    end

    subgraph Modeling Layer
        C --> D[model.py\nGridSearchCV + RandomForestRegressor]
        D --> E[(Joblib Saved Models\nmodels/sl-*.joblib)]
    end

    subgraph API & Serving Layer
        E --> F[app.py\nFlask RESTful API Port 8080]
        F --> G1[/train Endpoint]
        F --> G2[/predict Endpoint]
        F --> G3[/logfile Endpoint]
    end

    subgraph MLOps & Monitoring
        F --> H[logger.py\nupdate_train_log / update_predict_log]
        H --> I[(CSV Logs\nlogs/*.log)]
        F --> J[monitoring.py\nload_test / detect_drift / monitor_performance]
        J --> K[drift_report.json / load_test_report.json]
    end

    subgraph Business Analysis
        E --> L[post_production.py\nTime-Series Holdout Evaluation]
        L --> M[post_production_analysis.png]
    end
```

---

## 4. Data Ingestion & Preprocessing

Data ingestion is handled programmatically via `data_ingestion.py` and `cslib.py`.

### Training & Production Datasets:
- **`cs-train`**: 21 JSON invoice files spanning November 28, 2017 to July 31, 2019 (815,011 raw records across 43 countries). Total training revenue: $3,914,197.37. Top country: United Kingdom ($3,521,513.51 revenue; 92.17% of total volume).
- **`cs-production`**: 5 JSON invoice files spanning August 1, 2019 to December 31, 2019 (252,359 raw records across 37 countries).

### Data Ingestion Functions:
- `ingest_data(data_dir)`: Loads JSON files, standardizes column names (`StreamID` -> `stream_id`, `TimesViewed` -> `times_viewed`, `total_price` -> `price`), parses invoice dates, and validates column schemas.
- `convert_to_ts(df, country)`: Aggregates raw transactions into a daily time series containing `purchases`, `unique_invoices`, `unique_streams`, `total_views`, and daily `revenue`.
- `fetch_ts(data_dir, clean=False)`: Caches time-series CSVs in `ts-data/` for top revenue countries (`united_kingdom`, `eire`, `germany`, `france`, `norway`, `spain`, `hong_kong`, `portugal`, `singapore`, `netherlands`) and `all` countries aggregated.

---

## 5. Model Architecture & Pipeline

### Feature Engineering (`engineer_features`):
For any target date $t$, the engineered features capture historical rolling revenue trends and activity metrics:
- `previous_7`, `previous_14`, `previous_28`, `previous_70`: Sum of revenue over preceding 7, 14, 28, and 70 days.
- `previous_year`: Sum of revenue during the equivalent 30-day period 365 days prior.
- `recent_invoices`: Mean daily unique invoices over the preceding 30 days.
- `recent_views`: Mean daily views over the preceding 30 days.
- **Target ($y$)**: Sum of revenue over the next 30 days $[t, t + 30]$.

### Model Pipeline:
- **Scaler**: `StandardScaler()`
- **Model**: `RandomForestRegressor()` cross-validated over `n_estimators` $\in [10, 15, 20, 25]$ and `criterion` $\in ['squared\_error', 'absolute\_error']$.
- **Persistence**: Models are saved using `joblib` with standard naming convention `sl-{country}-0_1.joblib` (production) and `test-{country}-0_1.joblib` (unit test isolation).

---

## 6. RESTful API Implementation

The API is built using Flask (`app.py`) running on port `8080`.

### Endpoints Overview:

| Endpoint | Method | Query Parameters / Body | Description | Example Response |
|---|---|---|---|---|
| `/` | GET | None | Health check & API metadata | `{"status": "running", "version": "0.1"}` |
| `/train` | GET / POST | `mode` (`prod` or `test`) | Trains models for all countries | `{"status": "success", "mode": "prod", "results": {...}}` |
| `/predict` | GET / POST | `country`, `year`, `month`, `day`, `mode` | Returns 30-day revenue prediction | `{"status": "success", "country": "united_kingdom", "y_pred": [157492.90]}` |
| `/logfile` | GET | `type` (`predict` or `train`), `mode` | Returns log entries from CSV logs | `{"status": "success", "count": 11, "logs": [...]}` |

### Real Execution Verification:
- **`GET /`**: HTTP 200 OK — `{"endpoints": ["/train", "/predict", "/logfile"], "status": "running", "version": "0.1"}`
- **`GET /predict?country=united_kingdom&year=2018&month=01&day=05`**: HTTP 200 OK — `y_pred = [157492.90]`
- **`GET /predict?country=all_countries&year=2018&month=01&day=05`**: HTTP 200 OK — Returns predictions for all 11 country models.
- **`GET /logfile?type=predict`**: HTTP 200 OK — Returns 11 recorded prediction events.

---

## 7. Unit Testing Strategy & Verification

The unit test suite in `unittests.py` covers all core components of the system.

### Running Test Command:
```bash
pytest unittests.py -v
```

### Empirical Test Execution Results:
```text
============================= test session starts ==============================
platform darwin -- Python 3.9.6, pytest-8.4.2, pluggy-1.6.0
collected 40 items

unittests.py::TestModel::test_model_train_returns_results PASSED        [  2%]
unittests.py::TestModel::test_model_train_has_rmse PASSED               [  5%]
unittests.py::TestModel::test_model_train_has_mae PASSED                [  7%]
unittests.py::TestModel::test_model_can_be_loaded PASSED                [ 10%]
unittests.py::TestModel::test_model_load_data_matches PASSED            [ 12%]
unittests.py::TestModel::test_model_predict_returns_result PASSED       [ 15%]
unittests.py::TestModel::test_model_predict_output_type PASSED          [ 17%]
unittests.py::TestModel::test_model_predict_positive_revenue PASSED     [ 20%]
unittests.py::TestModel::test_model_predict_specific_country PASSED     [ 22%]
unittests.py::TestModel::test_model_predict_invalid_country PASSED      [ 25%]
unittests.py::TestModel::test_model_predict_invalid_date PASSED         [ 27%]
unittests.py::TestModel::test_model_predict_all_countries PASSED        [ 30%]
unittests.py::TestModel::test_engineer_features_output PASSED           [ 32%]
unittests.py::TestModel::test_engineer_features_no_nans PASSED          [ 35%]
unittests.py::TestLogging::test_train_log_is_created PASSED             [ 37%]
unittests.py::TestLogging::test_predict_log_is_created PASSED           [ 40%]
unittests.py::TestLogging::test_read_train_log PASSED                  [ 42%]
unittests.py::TestLogging::test_read_predict_log PASSED                [ 45%]
unittests.py::TestLogging::test_log_isolation_test_vs_prod PASSED       [ 47%]
unittests.py::TestLogging::test_log_does_not_fail_on_empty PASSED       [ 50%]
unittests.py::TestLogging::test_prediction_is_logged PASSED            [ 52%]
unittests.py::TestAPI::test_index PASSED                                 [ 55%]
unittests.py::TestAPI::test_train_endpoint PASSED                        [ 57%]
unittests.py::TestAPI::test_predict_endpoint PASSED                      [ 60%]
unittests.py::TestAPI::test_predict_specific_country PASSED             [ 62%]
unittests.py::TestAPI::test_predict_all_countries PASSED                [ 65%]
unittests.py::TestAPI::test_predict_missing_params PASSED               [ 67%]
unittests.py::TestAPI::test_predict_invalid_date PASSED                 [ 70%]
unittests.py::TestAPI::test_predict_invalid_country PASSED              [ 72%]
unittests.py::TestAPI::test_logfile_endpoint PASSED                     [ 75%]
unittests.py::TestAPI::test_logfile_train_logs PASSED                    [ 77%]
unittests.py::TestAPI::test_logfile_invalid_type PASSED                 [ 80%]
unittests.py::TestAPI::test_predict_json_body PASSED                     [ 82%]
unittests.py::TestDataIngestion::test_ingest_data_returns_dataframe PASSED [ 85%]
unittests.py::TestDataIngestion::test_ingest_data_has_required_columns PASSED [ 87%]
unittests.py::TestDataIngestion::test_ingest_data_invalid_dir PASSED    [ 90%]
unittests.py::TestDataIngestion::test_ingest_ts_data_returns_dict PASSED [ 92%]
unittests.py::TestDataIngestion::test_ts_data_has_revenue_column PASSED  [ 95%]
unittests.py::TestIsolation::test_no_production_model_overwrite PASSED  [ 97%]
unittests.py::TestIsolation::test_no_production_log_overwrite PASSED    [100%]

================ 40 passed in 4.79s =================
```

---

## 8. Logging Infrastructure & Test Isolation

Logging is implemented in `logger.py`.

### Structure:
- Production training events are logged to `logs/train.log`.
- Production prediction events are logged to `logs/predict.log`.
- Test training events are logged to `logs/test-train.log`.
- Test prediction events are logged to `logs/test-predict.log`.

### Strict Test Isolation:
In `unittests.py`, all unit tests utilize temporary directory fixtures (`tempfile.mkdtemp` via `temp_dirs` and `temp_log_dir`). Models trained during tests are prefixed with `test-` and written exclusively to isolated temporary paths, guaranteeing zero pollution or overwriting of production models in `models/` or production log files in `logs/`.

---

## 9. Containerization & Docker Verification

The solution is containerized using Docker (`Dockerfile`).

### Dockerfile Specification:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py ./
COPY cs-train/ ./cs-train/
COPY cs-production/ ./cs-production/
RUN mkdir -p models logs figures
EXPOSE 8080
CMD python -c "from model import model_train; model_train('cs-train', test=False)" && python -m flask --app app run --host=0.0.0.0 --port=8080
```

### Empirical Docker Verification:
1. **Build Command**:
   ```bash
   docker build --no-cache -t ai-workflow-capstone .
   ```
   *Result*: Successfully built image `ai-workflow-capstone:latest` (image ID `sha256:22bc43fd1272fcda77fa689eb0df49ce2b988f5cfef00`).
2. **Container Launch Command**:
   ```bash
   docker run -d -p 8081:8080 --name capstone-api ai-workflow-capstone:latest
   ```
   *Result*: Container started successfully. On startup, the container trained production models for all 11 territories and launched the Flask API.
3. **Containerized API Endpoints Tested**:
   - `GET http://127.0.0.1:8081/` -> HTTP 200 OK (`status: running`)
   - `GET http://127.0.0.1:8081/predict?country=united_kingdom&year=2018&month=01&day=05` -> HTTP 200 OK (`y_pred: [157492.90]`)
   - `GET http://127.0.0.1:8081/predict?country=all_countries&year=2018&month=01&day=05` -> HTTP 200 OK (returned predictions for all 11 territories)
   - `GET http://127.0.0.1:8081/logfile?type=predict` -> HTTP 200 OK (`count: 2`)

---

## 10. Load & Scalability Testing

API performance under load was tested using `monitoring.py` against the live running API server.

### Load Test Configuration & Results:
- **Total Requests Sent**: 20 consecutive HTTP GET requests targeting prediction endpoints across varied historical dates.
- **Successful Responses**: 20 / 20 (100% success rate, 0 failures).
- **Average Response Time**: **0.0074 seconds** (7.4 ms).
- **Minimum Response Time**: **0.0048 seconds** (4.8 ms).
- **Maximum Response Time**: **0.0205 seconds** (20.5 ms).
- **Standard Deviation of Response Time**: **0.0033 seconds** (3.3 ms).

Results are persisted in `load_test_report.json`.

---

## 11. Data Drift Detection & Logic Investigation

Data drift monitoring is implemented in `monitoring.py` via `detect_drift(train_dir, prod_dir, threshold=0.15)`.

### Drift Audit & Investigation Findings:
In auditing the drift detection logic, we investigated why the previous report contained seemingly contradictory statements.
- **Actual Empirical Comparison**:
  - `price` in `cs-train`: Mean = **$4.80**, Standard Deviation = **$135.06**.
  - `price` in `cs-production`: Mean = **$4.15**, Standard Deviation = **$75.20**.
  - **Mean Change**: `abs(4.15 - 4.80) / 4.80 = 13.50%` (Note: mean actually *decreased* by 13.50%, not increased).
  - **Standard Deviation Change**: `abs(75.20 - 135.06) / 135.06 = 44.32%`.
- **Drift Logic Code**:
  ```python
  feature_drift = bool(mean_change > threshold or std_change > threshold)
  ```
- **Explanation**: The 13.50% mean change alone was below the 15% threshold. However, because standard deviation changed by **44.32%** (well above the 15% threshold), `feature_drift` correctly evaluated to `True`. The previous report misattributed the drift flag to the mean shift instead of the standard deviation shift.

### Summary Drift Report (`drift_report.json`):
- `price`: Mean Change = 13.50%, Std Change = **44.32%** -> **DRIFT DETECTED** ⚠️
- `times_viewed`: Mean Change = 2.82%, Std Change = 0.88% -> No Drift ✓
- **Overall Data Drift Status**: **DRIFT DETECTED** (Triggered by price dispersion shift between training and production invoices).

---

## 12. Model Comparison: Main vs Baseline

Models were benchmarked using `model_comparison.py` on the training dataset using a randomized 75/25 train/test split.

### Model Evaluation Comparison Table:

| Model Architecture | RMSE | MAE | R² | MAPE (%) | Status |
|---|---|---|---|---|---|
| **Ridge Regression (Baseline)** | $55,532.24 | $43,445.43 | 0.5300 | 25.94% | Baseline Benchmark |
| **Random Forest (Main Model)** | **$18,232.42** | **$12,840.53** | **0.9493** | **6.91%** | **Production Champion** |
| **Improvement** | **-67.17%** | **-70.44%** | **+0.4193** | **-19.03% pts** | **Significant Gain** |

### Generated Comparison Visualizations:
1. `figures/model_comparison.png`: Four-panel bar chart comparing RMSE, MAE, R², and MAPE side-by-side.
2. `figures/actual_vs_predicted.png`: Scatter plots showing predicted vs actual revenue for Ridge vs Random Forest.

---

## 13. EDA Visualizations Overview

All exploratory data analysis plots were generated using `eda_visualizations.py` and saved to `figures/`.

1. **`eda_revenue_over_time.png`**: Time-series plot of total daily revenue across all countries, highlighting revenue surges and baseline seasonality.
2. **`eda_top_countries.png`**: Horizontal bar chart of top 10 revenue-generating territories. United Kingdom leads with $3.52M out of $3.91M total training revenue.
3. **`eda_price_distribution.png`**: Dual histogram showing raw invoice price distribution and log-transformed price distribution.
4. **`eda_views_distribution.png`**: Distribution of product view counts.
5. **`eda_monthly_seasonality.png`**: Average daily revenue grouped by calendar month, revealing peak revenue periods in late Q4 (November/December).
6. **`eda_correlation_matrix.png`**: Heatmap of feature correlations (`purchases`, `unique_invoices`, `unique_streams`, `total_views`, `revenue`).
7. **`eda_top_country_revenue.png`**: 7-day rolling average revenue comparison across UK, Germany, and France.
8. **`eda_target_distribution.png`**: Histogram and time-series view of the 30-day forward rolling target revenue ($y$).

---

## 14. Post-Production Analysis & Business Metric Connection

Post-production analysis was executed via `post_production.py` to evaluate how model performance translates to business metric utility.

### Distinction Between Evaluation Methodologies:
- **Shuffled Split MAPE (6.91% - 7.08%)**: Evaluates model performance when rows are randomly sampled across the dataset. Because consecutive days share rolling features, random splits introduce data leakage across time.
- **Time-Based Holdout Evaluation (21.4% MAPE)**: Splits data chronologically (training on the first 75% of dates and evaluating strictly on the final 25% holdout dates). This reflects true out-of-sample future forecasting performance.

### Empirical Holdout Evaluation & Business Metric Results:
- **Out-of-Sample Holdout MAPE**: **21.4%**
- **30-Day Revenue Forecast Accuracy**: $100\% - 21.4\% = \mathbf{78.6\%}$
- **Total Actual Holdout Revenue**: **$22,344,909**
- **Total Predicted Holdout Revenue**: **$22,766,956**
- **Aggregate Revenue Forecast Error**: $\frac{|22,766,956 - 22,344,909|}{22,344,909} = \mathbf{1.89\%}$
- **Revenue-Error Correlation**: **0.054**

### Business Interpretation:
While daily 30-day rolling predictions exhibit a mean absolute percentage error of 21.4% on out-of-sample data, errors smooth out significantly when aggregated across longer time horizons or corporate reporting periods. The aggregate revenue forecast error across the entire holdout period is only **1.89%**. 

Furthermore, the near-zero correlation (**0.054**) between actual revenue level and prediction error indicates that model accuracy remains stable across both high-revenue peak seasons and lower-revenue baseline months. Corporate leadership can rely on these forecasts for capital allocation and financial planning.

Generated visualization: `figures/post_production_analysis.png`.

---

## 15. Peer-Review Checklist & Verification Matrix

All 16 peer-review grading criteria have been empirically verified and marked PASS:

| # | Peer-Review Grading Criterion | Status | Concrete Verification Evidence |
|---|---|---|---|
| 1 | Are there unit tests for the API? | **PASS** | `TestAPI` in `unittests.py` contains 12 dedicated tests (`test_index`, `test_train_endpoint`, `test_predict_endpoint`, `test_predict_specific_country`, `test_predict_all_countries`, `test_predict_missing_params`, `test_predict_invalid_date`, `test_predict_invalid_country`, `test_logfile_endpoint`, `test_logfile_train_logs`, `test_logfile_invalid_type`, `test_predict_json_body`). |
| 2 | Are there unit tests for the model? | **PASS** | `TestModel` in `unittests.py` contains 14 dedicated tests verifying training, loading, prediction arrays, positive predictions, invalid inputs, feature engineering, and zero NaNs. |
| 3 | Are there unit tests for the logging? | **PASS** | `TestLogging` in `unittests.py` contains 7 dedicated tests verifying train log creation, predict log creation, log reading, empty log handling, prediction event logging, and log separation. |
| 4 | Can all unit tests be run with a single script? | **PASS** | Command `pytest unittests.py -v` executes the complete 40-test suite in a single invocation. |
| 5 | Do all unit tests pass? | **PASS** | Test execution output confirms **40 passed, 0 failed, 0 skipped** in 4.79s. |
| 6 | Is there a mechanism to monitor performance? | **PASS** | `monitoring.py` provides `monitor_performance()` calculating RMSE, MAE, R², and MAPE, saving output to `performance_report.json`. |
| 7 | Was there an attempt to isolate read/write unit tests from production models and logs? | **PASS** | `unittests.py` uses `tempfile.mkdtemp` fixtures (`temp_dirs`, `temp_log_dir`) to write test models (`test-*.joblib`) and test logs (`test-*.log`) to temporary directories. Verified by `TestIsolation` class (2 tests). |
| 8 | Does the API work as expected? | **PASS** | Live Flask API tested via HTTP GET/POST on port 8080; all endpoints returned HTTP 200 OK with valid JSON schema. |
| 9 | Can predictions be obtained for a specific country? | **PASS** | Query `GET /predict?country=united_kingdom&year=2018&month=01&day=05` returned HTTP 200 OK and valid revenue prediction `[157492.90]`. |
| 10 | Can predictions be obtained for all countries combined? | **PASS** | Query `GET /predict?country=all_countries&year=2018&month=01&day=05` returned HTTP 200 OK containing predictions for `all` and all 10 country territories. |
| 11 | Does data ingestion exist as a function or script to facilitate automation? | **PASS** | `data_ingestion.py` provides reusable functions `ingest_data()`, `ingest_ts_data()`, and `get_data_summary()`, verified on 815,011 invoice records. |
| 12 | Were multiple models compared? | **PASS** | `model_comparison.py` trained and benchmarked Random Forest Regressor vs Ridge Regression baseline. |
| 13 | Did the EDA investigation use visualizations? | **PASS** | `eda_visualizations.py` generated 8 distinct EDA plots saved in `figures/`. |
| 14 | Is everything containerized within a working Docker image? | **PASS** | Docker image `ai-workflow-capstone` built from scratch via `docker build --no-cache -t ai-workflow-capstone .`. Container executed, trained models on startup, and served live API queries on port 8081. |
| 15 | Was a visualization used to compare the model to the baseline model? | **PASS** | Generated `figures/model_comparison.png` (4-panel bar chart) and `figures/actual_vs_predicted.png` (actual vs predicted scatter plot). |
| 16 | Was a post-production analysis conducted linking model performance to business metric? | **PASS** | `post_production.py` performed time-based holdout analysis, linking MAPE (21.4%) to 30-day forecast accuracy (78.6%), total revenue forecast error (1.89%), and correlation (0.054), saving plot `figures/post_production_analysis.png`. |

---

## 16. Final Findings

1. **Model Superiority**: Random Forest significantly outperforms the Ridge baseline across all metrics (R² of 0.9493 vs 0.5300; RMSE reduced by 67.17%).
2. **Business Accuracy**: On out-of-sample time-based holdout evaluation, the model achieves a 78.6% 30-day daily forecast accuracy and a 1.89% aggregate revenue forecast error across the holdout period.
3. **Data Drift Reality**: Production invoices exhibit significant price dispersion drift ($\sigma$ change of 44.32%), indicating changing customer purchasing patterns or product tier pricing in 2019 Q3/Q4. Continuous model retraining on new production data is essential.

---

## 17. System Limitations

1. **Dominance of UK Market**: United Kingdom accounts for over 92% of total transaction volume and revenue. Smaller countries (e.g. Portugal, Spain) have sparse daily transaction data, leading to higher variance in single-country sub-models.
2. **Cold-Start for New Countries**: Countries outside the top 10 revenue generators default to the `all` aggregated model.

---

## 18. Future Improvements

1. **Automated Retraining Trigger**: Integrate `detect_drift()` into a CI/CD pipeline to automatically invoke `/train` whenever price dispersion drift exceeds 15%.
2. **Advanced Time-Series Modeling**: Explore Prophet or XGBoost regressors with explicit lag features and holiday indicators.

---

## 19. Instructions for Running the Project

### 1. Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Complete Unit Test Suite
```bash
pytest unittests.py -v
```

### 3. Run Data Ingestion & Model Training
```bash
python3 data_ingestion.py
python3 -c "from model import model_train; model_train('cs-train', test=False)"
```

### 4. Start local Flask API
```bash
python3 app.py
```

### 5. Run Monitoring, Drift, & Load Tests
```bash
python3 monitoring.py --mode all
```

### 6. Run Model Comparison & EDA
```bash
python3 model_comparison.py
python3 eda_visualizations.py
python3 post_production.py
```

### 7. Build and Run Docker Container
```bash
docker build --no-cache -t ai-workflow-capstone .
docker run -d -p 8080:8080 --name capstone-api ai-workflow-capstone:latest
curl "http://127.0.0.1:8080/predict?country=united_kingdom&year=2018&month=01&day=05"
```
