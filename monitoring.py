#!/usr/bin/env python
"""
Monitoring module for the AI Workflow Capstone project.
Includes: load testing, data drift detection, and model performance monitoring.
"""

import os
import sys
import time
import json
import requests
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cslib import fetch_ts, engineer_features, fetch_data
from model import model_load, model_predict, MODEL_VERSION


# ============================================================
# LOAD / SCALABILITY TESTING
# ============================================================

def load_test(base_url="http://127.0.0.1:8080", n_requests=20, mode="prod"):
    """
    Test the API under load with multiple prediction requests.

    Parameters
    ----------
    base_url : str
        Base URL of the API.
    n_requests : int
        Number of prediction requests to make.
    mode : str
        'test' or 'prod' mode.

    Returns
    -------
    dict
        Load test results including response times.
    """
    print(f"[load_test] Making {n_requests} prediction requests...")

    # Use different dates to simulate realistic usage
    dates = [
        ("2018", "01", "05"), ("2018", "02", "10"), ("2018", "03", "15"),
        ("2018", "04", "20"), ("2018", "05", "25"), ("2018", "06", "01"),
        ("2018", "07", "10"), ("2018", "08", "15"), ("2018", "09", "20"),
        ("2018", "10", "05"), ("2018", "11", "10"), ("2018", "12", "15"),
        ("2018", "01", "10"), ("2018", "02", "15"), ("2018", "03", "20"),
        ("2018", "04", "05"), ("2018", "05", "10"), ("2018", "06", "15"),
        ("2018", "07", "20"), ("2018", "08", "25"),
    ]

    response_times = []
    successes = 0
    failures = 0

    for i in range(min(n_requests, len(dates))):
        year, month, day = dates[i % len(dates)]
        url = f"{base_url}/predict?country=all&year={year}&month={month}&day={day}&mode={mode}"

        start = time.time()
        try:
            resp = requests.get(url, timeout=120)
            elapsed = time.time() - start
            response_times.append(elapsed)

            if resp.status_code == 200:
                successes += 1
            else:
                failures += 1
                print(f"  Request {i+1}: FAILED (status {resp.status_code})")
        except Exception as e:
            failures += 1
            elapsed = time.time() - start
            response_times.append(elapsed)
            print(f"  Request {i+1}: ERROR - {e}")

    results = {
        "total_requests": n_requests,
        "successes": successes,
        "failures": failures,
        "avg_response_time_sec": round(float(np.mean(response_times)), 4) if response_times else 0,
        "min_response_time_sec": round(float(np.min(response_times)), 4) if response_times else 0,
        "max_response_time_sec": round(float(np.max(response_times)), 4) if response_times else 0,
        "std_response_time_sec": round(float(np.std(response_times)), 4) if response_times else 0,
    }

    print(f"\n[load_test] Results:")
    print(f"  Successes: {successes}/{n_requests}")
    print(f"  Avg response time: {results['avg_response_time_sec']:.3f}s")
    print(f"  Min response time: {results['min_response_time_sec']:.3f}s")
    print(f"  Max response time: {results['max_response_time_sec']:.3f}s")

    return results


# ============================================================
# DATA DRIFT DETECTION
# ============================================================

def detect_drift(train_dir, prod_dir, threshold=0.15):
    """
    Compare training data distributions against production data to detect drift.

    Uses relative change in summary statistics as a simple drift indicator.

    Parameters
    ----------
    train_dir : str
        Path to training data directory.
    prod_dir : str
        Path to production data directory.
    threshold : float
        Relative change threshold above which drift is flagged (0.15 = 15%).

    Returns
    -------
    dict
        Drift report with feature-level analysis.
    """
    print("[drift] Loading training data...")
    train_df = fetch_data(train_dir)

    print("[drift] Loading production data...")
    prod_df = fetch_data(prod_dir)

    # Features to compare
    numeric_features = ['price', 'times_viewed']

    drift_report = {
        "train_records": train_df.shape[0],
        "prod_records": prod_df.shape[0],
        "train_date_range": (str(train_df['invoice_date'].min()),
                             str(train_df['invoice_date'].max())),
        "prod_date_range": (str(prod_df['invoice_date'].min()),
                            str(prod_df['invoice_date'].max())),
        "features": {},
        "drift_detected": False
    }

    for feature in numeric_features:
        train_vals = train_df[feature].dropna()
        prod_vals = prod_df[feature].dropna()

        train_mean = train_vals.mean()
        prod_mean = prod_vals.mean()
        train_std = train_vals.std()
        prod_std = prod_vals.std()

        # Calculate relative change
        mean_change = abs(prod_mean - train_mean) / (abs(train_mean) + 1e-10)
        std_change = abs(prod_std - train_std) / (abs(train_std) + 1e-10)

        feature_drift = bool(mean_change > threshold or std_change > threshold)

        drift_report["features"][feature] = {
            "train_mean": round(float(train_mean), 2),
            "prod_mean": round(float(prod_mean), 2),
            "mean_change_pct": round(float(mean_change * 100), 2),
            "train_std": round(float(train_std), 2),
            "prod_std": round(float(prod_std), 2),
            "std_change_pct": round(float(std_change * 100), 2),
            "drift_detected": feature_drift
        }

        if feature_drift:
            drift_report["drift_detected"] = True

    # Check country distribution shift
    train_countries = train_df['country'].value_counts(normalize=True)
    prod_countries = prod_df['country'].value_counts(normalize=True)

    # Find countries that changed significantly
    all_countries = set(train_countries.index) | set(prod_countries.index)
    country_drift = {}
    for c in list(all_countries)[:10]:  # Top 10 for brevity
        train_pct = train_countries.get(c, 0)
        prod_pct = prod_countries.get(c, 0)
        change = abs(prod_pct - train_pct)
        country_drift[c] = {
            "train_pct": round(float(train_pct * 100), 2),
            "prod_pct": round(float(prod_pct * 100), 2),
            "change_pct": round(float(change * 100), 2)
        }

    drift_report["country_distribution"] = country_drift

    print(f"\n[drift] Drift Report:")
    print(f"  Training records: {drift_report['train_records']}")
    print(f"  Production records: {drift_report['prod_records']}")
    print(f"  Drift detected: {drift_report['drift_detected']}")
    for feat, info in drift_report["features"].items():
        flag = " ⚠️ DRIFT" if info["drift_detected"] else " ✓"
        print(f"  {feat}: mean change {info['mean_change_pct']:.1f}%, "
              f"std change {info['std_change_pct']:.1f}%{flag}")

    return drift_report


# ============================================================
# MODEL PERFORMANCE MONITORING
# ============================================================

def monitor_performance(data_dir, model_dir=None, countries=None):
    """
    Calculate model performance metrics on available data.

    Parameters
    ----------
    data_dir : str
        Path to data directory.
    model_dir : str, optional
        Path to model directory.
    countries : list, optional
        List of countries to evaluate. If None, evaluates all.

    Returns
    -------
    dict
        Performance metrics per country.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    print("[monitor] Loading models and data...")
    all_data, all_models = model_load(prefix='sl', data_dir=data_dir,
                                       training=False, model_dir=model_dir)

    if countries is None:
        countries = list(all_models.keys())

    performance = {}
    for country in countries:
        if country not in all_models or country not in all_data:
            continue

        model = all_models[country]
        data = all_data[country]

        X = data["X"]
        y = data["y"]

        y_pred = model.predict(X)

        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mape = np.mean(np.abs((y - y_pred) / (y + 1e-10))) * 100

        performance[country] = {
            "rmse": round(float(rmse), 2),
            "mae": round(float(mae), 2),
            "r2": round(float(r2), 4),
            "mape": round(float(mape), 2),
            "n_samples": int(X.shape[0])
        }

        print(f"  {country}: RMSE={rmse:.0f}, MAE={mae:.0f}, R²={r2:.4f}, MAPE={mape:.1f}%")

    return performance


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitoring tools")
    parser.add_argument("--mode", choices=["load", "drift", "performance", "all"],
                        default="all", help="Which monitoring to run")
    parser.add_argument("--url", default="http://127.0.0.1:8080",
                        help="API base URL for load testing")
    args = parser.parse_args()

    train_dir = os.path.join(".", "cs-train")
    prod_dir = os.path.join(".", "cs-production")

    if args.mode in ["drift", "all"]:
        print("\n" + "=" * 60)
        print("DATA DRIFT ANALYSIS")
        print("=" * 60)
        report = detect_drift(train_dir, prod_dir)
        with open("drift_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("Drift report saved to drift_report.json")

    if args.mode in ["performance", "all"]:
        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE MONITORING")
        print("=" * 60)
        perf = monitor_performance(train_dir)
        with open("performance_report.json", "w") as f:
            json.dump(perf, f, indent=2)
        print("Performance report saved to performance_report.json")

    if args.mode in ["load", "all"]:
        print("\n" + "=" * 60)
        print("LOAD TESTING")
        print("=" * 60)
        print("Note: API must be running for load testing.")
        print(f"Attempting to reach {args.url}...")
        try:
            resp = requests.get(f"{args.url}/", timeout=5)
            if resp.status_code == 200:
                results = load_test(args.url)
                with open("load_test_report.json", "w") as f:
                    json.dump(results, f, indent=2)
                print("Load test report saved to load_test_report.json")
            else:
                print("API not responding correctly. Start the API first.")
        except requests.ConnectionError:
            print("Could not connect to API. Start with: python app.py")
