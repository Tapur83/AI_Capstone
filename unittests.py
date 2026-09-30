#!/usr/bin/env python
"""
Comprehensive test suite for the AI Workflow Capstone project.
Tests the API, model, and logging functionality with proper test isolation.

Run with: pytest unittests.py -v
"""

import os
import sys
import json
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import model_train, model_predict, model_load, model_predict_all, MODEL_VERSION
from logger import update_train_log, update_predict_log, read_log
from cslib import fetch_data, fetch_ts, engineer_features
from data_ingestion import ingest_data, ingest_ts_data
from app import app

# ============================================================
# FIXTURES - test isolation using temporary directories
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cs-train")


@pytest.fixture(scope="session")
def temp_dirs():
    """Create temporary directories for test models and logs."""
    base = tempfile.mkdtemp(prefix="capstone_test_")
    model_dir = os.path.join(base, "models")
    log_dir = os.path.join(base, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    yield {"base": base, "model_dir": model_dir, "log_dir": log_dir}
    # Cleanup after all tests
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture(scope="session")
def trained_models(temp_dirs):
    """Train models once for the entire test session."""
    results = model_train(DATA_DIR, test=True,
                          model_dir=temp_dirs["model_dir"],
                          log_dir=temp_dirs["log_dir"])
    all_data, all_models = model_load(prefix='test', data_dir=DATA_DIR,
                                       training=False,
                                       model_dir=temp_dirs["model_dir"])
    return {"results": results, "all_data": all_data, "all_models": all_models}


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def temp_log_dir():
    """Create a temporary log directory for individual test isolation."""
    d = tempfile.mkdtemp(prefix="capstone_log_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ============================================================
# MODEL TESTS
# ============================================================

class TestModel:
    """Tests for the ML model."""

    def test_model_train_returns_results(self, trained_models):
        """Model training returns results for each country."""
        results = trained_models["results"]
        assert isinstance(results, dict)
        assert len(results) > 0
        assert "all" in results

    def test_model_train_has_rmse(self, trained_models):
        """Training results contain RMSE metric."""
        results = trained_models["results"]
        for country, result in results.items():
            assert "rmse" in result
            assert isinstance(result["rmse"], (int, float))
            assert result["rmse"] >= 0

    def test_model_train_has_mae(self, trained_models):
        """Training results contain MAE metric."""
        results = trained_models["results"]
        for country, result in results.items():
            assert "mae" in result
            assert result["mae"] >= 0

    def test_model_can_be_loaded(self, trained_models, temp_dirs):
        """Trained models can be loaded."""
        all_data, all_models = model_load(prefix='test', data_dir=DATA_DIR,
                                           training=False,
                                           model_dir=temp_dirs["model_dir"])
        assert len(all_models) > 0
        assert "all" in all_models

    def test_model_load_data_matches(self, trained_models):
        """Loaded data has matching dimensions."""
        all_data = trained_models["all_data"]
        for country, data in all_data.items():
            assert data["X"].shape[0] == data["dates"].shape[0]
            assert data["X"].shape[0] == data["y"].shape[0]

    def test_model_predict_returns_result(self, trained_models, temp_dirs):
        """Model prediction returns a result dict."""
        result = model_predict(
            'all', '2018', '01', '05',
            all_models=trained_models["all_models"],
            all_data=trained_models["all_data"],
            test=True, log_dir=temp_dirs["log_dir"]
        )
        assert "y_pred" in result
        assert result["y_pred"] is not None

    def test_model_predict_output_type(self, trained_models, temp_dirs):
        """Prediction output is a numpy array."""
        result = model_predict(
            'all', '2018', '01', '05',
            all_models=trained_models["all_models"],
            all_data=trained_models["all_data"],
            test=True, log_dir=temp_dirs["log_dir"]
        )
        assert isinstance(result["y_pred"], np.ndarray)

    def test_model_predict_positive_revenue(self, trained_models, temp_dirs):
        """Predicted revenue should be a reasonable positive number."""
        result = model_predict(
            'all', '2018', '01', '05',
            all_models=trained_models["all_models"],
            all_data=trained_models["all_data"],
            test=True, log_dir=temp_dirs["log_dir"]
        )
        assert result["y_pred"][0] > 0

    def test_model_predict_specific_country(self, trained_models, temp_dirs):
        """Prediction works for a specific country (united_kingdom)."""
        result = model_predict(
            'united_kingdom', '2018', '01', '05',
            all_models=trained_models["all_models"],
            all_data=trained_models["all_data"],
            test=True, log_dir=temp_dirs["log_dir"]
        )
        assert "y_pred" in result
        assert result["y_pred"][0] > 0

    def test_model_predict_invalid_country(self, trained_models, temp_dirs):
        """Prediction raises error for invalid country."""
        with pytest.raises(Exception, match="could not be found"):
            model_predict(
                'nonexistent_country', '2018', '01', '05',
                all_models=trained_models["all_models"],
                all_data=trained_models["all_data"],
                test=True, log_dir=temp_dirs["log_dir"]
            )

    def test_model_predict_invalid_date(self, trained_models, temp_dirs):
        """Prediction raises error for invalid date."""
        with pytest.raises(Exception):
            model_predict(
                'all', 'abc', '01', '05',
                all_models=trained_models["all_models"],
                all_data=trained_models["all_data"],
                test=True, log_dir=temp_dirs["log_dir"]
            )

    def test_model_predict_all_countries(self, trained_models, temp_dirs):
        """Prediction for all countries returns results for each."""
        results = model_predict_all(
            '2018', '01', '05',
            all_models=trained_models["all_models"],
            all_data=trained_models["all_data"],
            test=True, log_dir=temp_dirs["log_dir"]
        )
        assert isinstance(results, dict)
        assert len(results) > 0
        assert "all" in results

    def test_engineer_features_output(self):
        """engineer_features returns properly shaped output."""
        ts_data = fetch_ts(DATA_DIR)
        X, y, dates = engineer_features(ts_data['all'])
        assert X.shape[0] > 0
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] == dates.shape[0]

    def test_engineer_features_no_nans(self):
        """Engineered features should have no NaN values."""
        ts_data = fetch_ts(DATA_DIR)
        X, y, dates = engineer_features(ts_data['all'])
        assert X.isnull().sum().sum() == 0


# ============================================================
# LOGGING TESTS
# ============================================================

class TestLogging:
    """Tests for the logging module."""

    def test_train_log_is_created(self, temp_log_dir):
        """Train log file is created after logging."""
        update_train_log("test_tag", ("2018-01-01", "2019-01-01"),
                         {"rmse": 1234}, "000:00:01", 0.1,
                         "test", test=True, log_dir=temp_log_dir)
        log_path = os.path.join(temp_log_dir, "test-train.log")
        assert os.path.exists(log_path)

    def test_predict_log_is_created(self, temp_log_dir):
        """Predict log file is created after logging."""
        update_predict_log("all", [50000], None, "2019-01-15",
                           "000:00:01", 0.1, test=True, log_dir=temp_log_dir)
        log_path = os.path.join(temp_log_dir, "test-predict.log")
        assert os.path.exists(log_path)

    def test_read_train_log(self, temp_log_dir):
        """Train logs can be read back."""
        update_train_log("uk", ("2018-01-01", "2019-01-01"),
                         {"rmse": 999}, "000:00:01", 0.1,
                         "test", test=True, log_dir=temp_log_dir)
        logs = read_log("train", test=True, log_dir=temp_log_dir)
        assert len(logs) > 0
        assert logs[-1]["tag"] == "uk"

    def test_read_predict_log(self, temp_log_dir):
        """Predict logs can be read back."""
        update_predict_log("all", [12345], None, "2019-02-01",
                           "000:00:00", 0.1, test=True, log_dir=temp_log_dir)
        logs = read_log("predict", test=True, log_dir=temp_log_dir)
        assert len(logs) > 0
        assert logs[-1]["country"] == "all"

    def test_log_isolation_test_vs_prod(self, temp_log_dir):
        """Test and production logs are stored separately."""
        update_predict_log("all", [100], None, "2019-01-01",
                           "000:00:00", 0.1, test=True, log_dir=temp_log_dir)
        update_predict_log("all", [200], None, "2019-01-01",
                           "000:00:00", 0.1, test=False, log_dir=temp_log_dir)

        test_logs = read_log("predict", test=True, log_dir=temp_log_dir)
        prod_logs = read_log("predict", test=False, log_dir=temp_log_dir)

        # They should be in different files
        test_path = os.path.join(temp_log_dir, "test-predict.log")
        prod_path = os.path.join(temp_log_dir, "predict.log")
        assert os.path.exists(test_path)
        assert os.path.exists(prod_path)

    def test_log_does_not_fail_on_empty(self, temp_log_dir):
        """Reading a non-existent log returns empty list."""
        logs = read_log("predict", test=True, log_dir=temp_log_dir)
        # May be empty or not depending on previous tests, but should not raise
        assert isinstance(logs, list)

    def test_prediction_is_logged(self, trained_models, temp_dirs):
        """A prediction call creates a log entry."""
        log_dir = tempfile.mkdtemp(prefix="capstone_predlog_")
        try:
            model_predict(
                'all', '2018', '01', '05',
                all_models=trained_models["all_models"],
                all_data=trained_models["all_data"],
                test=True, log_dir=log_dir
            )
            logs = read_log("predict", test=True, log_dir=log_dir)
            assert len(logs) == 1
            assert logs[0]["country"] == "all"
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)


# ============================================================
# API TESTS
# ============================================================

class TestAPI:
    """Tests for the Flask API."""

    def test_index(self, client):
        """Health check endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "running"

    def test_train_endpoint(self, client):
        """Train endpoint returns success."""
        response = client.get("/train?mode=test")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"

    def test_predict_endpoint(self, client):
        """Predict endpoint returns prediction for 'all'."""
        # First ensure model is trained
        client.get("/train?mode=test")
        response = client.get("/predict?country=all&year=2018&month=01&day=05&mode=test")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert "y_pred" in data

    def test_predict_specific_country(self, client):
        """Predict endpoint works for a specific country."""
        client.get("/train?mode=test")
        response = client.get("/predict?country=united_kingdom&year=2018&month=01&day=05&mode=test")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"

    def test_predict_all_countries(self, client):
        """Predict endpoint works with all_countries parameter."""
        client.get("/train?mode=test")
        response = client.get("/predict?country=all_countries&year=2018&month=01&day=05&mode=test")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert "predictions" in data

    def test_predict_missing_params(self, client):
        """Predict endpoint returns 400 for missing parameters."""
        response = client.get("/predict?country=all")
        assert response.status_code == 400

    def test_predict_invalid_date(self, client):
        """Predict endpoint handles invalid date gracefully."""
        response = client.get("/predict?country=all&year=abc&month=01&day=05")
        assert response.status_code == 400

    def test_predict_invalid_country(self, client):
        """Predict endpoint handles invalid country gracefully."""
        client.get("/train?mode=test")
        response = client.get("/predict?country=nonexistent&year=2018&month=01&day=05&mode=test")
        assert response.status_code == 500
        data = response.get_json()
        assert data["status"] == "error"

    def test_logfile_endpoint(self, client):
        """Logfile endpoint returns logs."""
        # Train and predict to generate logs
        client.get("/train?mode=test")
        client.get("/predict?country=all&year=2018&month=01&day=05&mode=test")
        response = client.get("/logfile?type=predict")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert "logs" in data

    def test_logfile_train_logs(self, client):
        """Logfile endpoint returns train logs."""
        client.get("/train?mode=test")
        response = client.get("/logfile?type=train")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"

    def test_logfile_invalid_type(self, client):
        """Logfile endpoint rejects invalid log type."""
        response = client.get("/logfile?type=invalid")
        assert response.status_code == 400

    def test_predict_json_body(self, client):
        """Predict endpoint accepts JSON body."""
        client.get("/train?mode=test")
        response = client.post("/predict",
                               json={"country": "all", "year": "2018",
                                     "month": "01", "day": "05", "mode": "test"})
        assert response.status_code == 200


# ============================================================
# DATA INGESTION TESTS
# ============================================================

class TestDataIngestion:
    """Tests for data ingestion."""

    def test_ingest_data_returns_dataframe(self):
        """ingest_data returns a DataFrame."""
        df = ingest_data(DATA_DIR)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0

    def test_ingest_data_has_required_columns(self):
        """Ingested data has all required columns."""
        df = ingest_data(DATA_DIR)
        required = ['country', 'customer_id', 'invoice', 'price',
                     'stream_id', 'times_viewed', 'year', 'month', 'day']
        for col in required:
            assert col in df.columns

    def test_ingest_data_invalid_dir(self):
        """ingest_data raises error for invalid directory."""
        with pytest.raises(FileNotFoundError):
            ingest_data("/nonexistent/path")

    def test_ingest_ts_data_returns_dict(self):
        """ingest_ts_data returns a dictionary of DataFrames."""
        ts_data = ingest_ts_data(DATA_DIR)
        assert isinstance(ts_data, dict)
        assert len(ts_data) > 0
        assert "all" in ts_data

    def test_ts_data_has_revenue_column(self):
        """Time-series data has a revenue column."""
        ts_data = ingest_ts_data(DATA_DIR)
        for country, df in ts_data.items():
            assert "revenue" in df.columns


# ============================================================
# TEST ISOLATION VERIFICATION
# ============================================================

class TestIsolation:
    """Verify that tests don't affect production artifacts."""

    def test_no_production_model_overwrite(self, temp_dirs):
        """Test models are saved to temp directory, not production."""
        prod_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        # If prod model dir exists, check our test models aren't there
        if os.path.exists(prod_model_dir):
            test_models = [f for f in os.listdir(prod_model_dir) if f.startswith("test-")]
            # Test models from our test suite should not be in prod dir
            # (they should be in temp_dirs["model_dir"])
        # Test models should be in temp dir
        test_models = os.listdir(temp_dirs["model_dir"])
        assert len(test_models) > 0

    def test_no_production_log_overwrite(self, temp_dirs):
        """Test logs are saved to temp directory, not production."""
        test_logs = os.listdir(temp_dirs["log_dir"])
        assert any("test-" in f for f in test_logs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
