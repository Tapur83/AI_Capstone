# IBM AI Enterprise Workflow Capstone

Production deployment, RESTful API, MLOps automation, data drift monitoring, containerization, and post-production business metric analysis for AAVAIL's revenue forecasting engine.

---

## Quick Start & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run All Unit Tests
```bash
pytest unittests.py -v
```

### 3. Train Models Locally
```bash
python3 -c "from model import model_train; model_train('cs-train')"
```

### 4. Start the API Locally
```bash
python3 app.py
```
The API will run on `http://127.0.0.1:8080`.

---

## API Documentation

### `/train` — Train / Retrain Model
- **URL:** `http://127.0.0.1:8080/train?mode=prod`
- **Method:** `GET` or `POST`
- **Response:**
```json
{
  "status": "success",
  "mode": "prod",
  "results": {
    "all": { "rmse": 18753, "mae": 13135 },
    "united_kingdom": { "rmse": 14210, "mae": 9850 }
  }
}
```

### `/predict` — Generate Revenue Prediction
- **URL (Specific Country):** `http://127.0.0.1:8080/predict?country=united_kingdom&year=2018&month=01&day=05`
- **URL (All Countries):** `http://127.0.0.1:8080/predict?country=all_countries&year=2018&month=01&day=05`
- **Method:** `GET` or `POST` (Query parameters or JSON body)
- **Response:**
```json
{
  "status": "success",
  "country": "united_kingdom",
  "date": "2018-01-05",
  "y_pred": [178450.25],
  "y_proba": null
}
```

### `/logfile` — Access API Logs
- **URL:** `http://127.0.0.1:8080/logfile?type=predict&mode=prod`
- **Method:** `GET`
- **Response:**
```json
{
  "status": "success",
  "type": "predict",
  "mode": "prod",
  "count": 1,
  "logs": [...]
}
```

---

## Docker Instructions

### 1. Build Docker Image
```bash
docker build -t ai-workflow-capstone:latest .
```

### 2. Run Docker Container
```bash
docker run -d -p 8080:8080 --name capstone-api ai-workflow-capstone:latest
```

### 3. Test Containerized API
```bash
curl "http://127.0.0.1:8080/predict?country=all&year=2018&month=01&day=05"
```

---

## Monitoring, Drift & Post-Production Analysis

### Run Monitoring & Drift Analysis
```bash
python3 monitoring.py --mode all
```

### Run Model Comparison (Main vs Baseline)
```bash
python3 model_comparison.py
```

### Run EDA Visualizations
```bash
python3 eda_visualizations.py
```

### Run Post-Production Business Metric Analysis
```bash
python3 post_production.py
```

All figures are saved to `figures/`.

---

## Repository Structure

- `app.py`: Flask API with `/train`, `/predict`, `/logfile` endpoints.
- `model.py`: Random Forest revenue prediction pipeline and loading functions.
- `logger.py`: Production and test logging functions with strict isolation.
- `cslib.py`: Time-series conversion and feature engineering routines.
- `data_ingestion.py`: Programmatic data loading, validation, and preprocessing.
- `unittests.py`: Complete 40-test PyTest test suite.
- `monitoring.py`: Load testing, data drift detection, and performance tracking.
- `model_comparison.py`: Benchmark script comparing Random Forest vs Ridge baseline.
- `eda_visualizations.py`: EDA visualizer generating 8 key plots.
- `post_production.py`: Post-production analysis connecting MAPE to revenue accuracy.
- `Dockerfile`: Multi-stage Docker container configuration.
- `requirements.txt`: Python package dependencies.
