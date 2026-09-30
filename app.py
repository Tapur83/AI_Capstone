#!/usr/bin/env python
"""
Flask API for the AI Workflow Capstone project.
Provides /train, /predict, and /logfile endpoints.
"""

import os
import json
import re
import traceback
from flask import Flask, request, jsonify
from model import model_train, model_predict, model_predict_all, model_load
from logger import read_log

app = Flask(__name__)

# Default data directory
DATA_DIR = os.path.join(".", "cs-train")
MODEL_DIR = os.path.join(".", "models")
LOG_DIR = os.path.join(".", "logs")


@app.route("/")
def index():
    """Health check endpoint."""
    return jsonify({
        "status": "running",
        "endpoints": ["/train", "/predict", "/logfile"],
        "version": "0.1"
    })


@app.route("/train", methods=["GET", "POST"])
def train():
    """
    Train or retrain the model.

    Query Parameters
    ----------------
    mode : str, optional
        'test' to train in test mode (subset of data/countries).

    Returns
    -------
    JSON response with training results.
    """
    try:
        mode = request.args.get("mode", "prod")
        test = mode == "test"

        print(f"[API /train] Training in {'test' if test else 'production'} mode")
        results = model_train(DATA_DIR, test=test, model_dir=MODEL_DIR, log_dir=LOG_DIR)

        return jsonify({
            "status": "success",
            "mode": mode,
            "results": {k: {"rmse": v["rmse"], "mae": v["mae"]} for k, v in results.items()}
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Make a revenue prediction.

    Query Parameters
    ----------------
    country : str
        Country identifier (e.g. 'all', 'united_kingdom'). Use 'all_countries' to predict for all.
    year : str
        Year (e.g. '2018').
    month : str
        Month (e.g. '01').
    day : str
        Day (e.g. '05').

    Returns
    -------
    JSON response with prediction.
    """
    try:
        # Get parameters from query string or JSON body
        if request.is_json:
            params = request.get_json()
        else:
            params = request.args.to_dict()

        country = params.get("country")
        year = params.get("year")
        month = params.get("month")
        day = params.get("day")

        # Input validation
        if not all([country, year, month, day]):
            return jsonify({
                "status": "error",
                "message": "Missing required parameters: country, year, month, day"
            }), 400

        # Validate date components are numeric
        for name, val in [("year", year), ("month", month), ("day", day)]:
            if re.search("\\D", str(val)):
                return jsonify({
                    "status": "error",
                    "message": f"Invalid {name}: must be numeric"
                }), 400

        mode = params.get("mode", "prod")
        test = mode == "test"

        print(f"[API /predict] country={country}, date={year}-{month}-{day}, mode={mode}")

        # Predict for all countries
        if country == "all_countries":
            results = model_predict_all(year, month, day, test=test,
                                        data_dir=DATA_DIR, model_dir=MODEL_DIR,
                                        log_dir=LOG_DIR)
            return jsonify({
                "status": "success",
                "predictions": results
            })

        # Predict for a specific country
        result = model_predict(country, year, month, day, test=test,
                               data_dir=DATA_DIR, model_dir=MODEL_DIR,
                               log_dir=LOG_DIR)

        return jsonify({
            "status": "success",
            "country": country,
            "date": f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}",
            "y_pred": result['y_pred'].tolist(),
            "y_proba": result['y_proba'].tolist() if result['y_proba'] is not None else None
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/logfile", methods=["GET"])
def logfile():
    """
    Access prediction and training logs.

    Query Parameters
    ----------------
    type : str
        'predict' or 'train'.
    mode : str, optional
        'test' to read test logs. Default is 'prod'.

    Returns
    -------
    JSON response with log entries.
    """
    try:
        log_type = request.args.get("type", "predict")
        mode = request.args.get("mode", "prod")
        test = mode == "test"

        if log_type not in ["predict", "train"]:
            return jsonify({
                "status": "error",
                "message": "Invalid log type. Use 'predict' or 'train'."
            }), 400

        logs = read_log(log_type, test=test, log_dir=LOG_DIR)

        return jsonify({
            "status": "success",
            "type": log_type,
            "mode": mode,
            "count": len(logs),
            "logs": logs
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
