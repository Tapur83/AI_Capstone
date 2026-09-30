#!/usr/bin/env python
"""
Logging module for the AI Workflow Capstone project.
Provides separate logging for train and predict events,
with isolation between test and production logs.
"""

import os
import csv
import time
from datetime import datetime

LOG_DIR = os.path.join(".", "logs")


def _ensure_log_dir(log_dir=None):
    """Create the log directory if it doesn't exist."""
    d = log_dir if log_dir else LOG_DIR
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def _get_log_path(log_type, test=False, log_dir=None):
    """Get the log file path based on type and test flag."""
    d = log_dir if log_dir else LOG_DIR
    prefix = "test-" if test else ""
    return os.path.join(d, f"{prefix}{log_type}.log")


def update_train_log(tag, date_range, metrics, runtime,
                     model_version, model_version_note, test=False, log_dir=None):
    """
    Log a training event.

    Parameters
    ----------
    tag : str
        Country or model identifier.
    date_range : tuple
        (start_date, end_date) of training data.
    metrics : dict
        Training metrics, e.g. {'rmse': 1234}.
    runtime : str
        Training runtime string.
    model_version : float
        Model version number.
    model_version_note : str
        Note about this model version.
    test : bool
        If True, log to test log file.
    log_dir : str, optional
        Override default log directory.
    """
    _ensure_log_dir(log_dir)
    log_path = _get_log_path("train", test=test, log_dir=log_dir)

    header = ['timestamp', 'tag', 'date_range', 'metrics', 'runtime',
              'model_version', 'model_version_note']
    row = [datetime.now().isoformat(), tag, str(date_range), str(metrics),
           runtime, model_version, model_version_note]

    write_header = not os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def update_predict_log(country, y_pred, y_proba, target_date,
                       runtime, model_version, test=False, log_dir=None):
    """
    Log a prediction event.

    Parameters
    ----------
    country : str
        Country for which prediction was made.
    y_pred : array-like
        Predicted values.
    y_proba : array-like or None
        Predicted probabilities (if applicable).
    target_date : str
        Date for which prediction was made.
    runtime : str
        Prediction runtime string.
    model_version : float
        Model version number.
    test : bool
        If True, log to test log file.
    log_dir : str, optional
        Override default log directory.
    """
    _ensure_log_dir(log_dir)
    log_path = _get_log_path("predict", test=test, log_dir=log_dir)

    header = ['timestamp', 'country', 'y_pred', 'y_proba', 'target_date',
              'runtime', 'model_version']
    row = [datetime.now().isoformat(), country, str(y_pred), str(y_proba),
           target_date, runtime, model_version]

    write_header = not os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def read_log(log_type, test=False, log_dir=None):
    """
    Read a log file and return its contents as a list of dicts.

    Parameters
    ----------
    log_type : str
        'train' or 'predict'.
    test : bool
        If True, read test log file.
    log_dir : str, optional
        Override default log directory.

    Returns
    -------
    list of dict
        Log entries.
    """
    _ensure_log_dir(log_dir)
    log_path = _get_log_path(log_type, test=test, log_dir=log_dir)

    if not os.path.exists(log_path):
        return []

    rows = []
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


if __name__ == "__main__":
    # Quick test
    _ensure_log_dir()
    update_train_log("test_tag", ("2018-01-01", "2019-07-01"),
                     {"rmse": 12345}, "000:00:05", 0.1,
                     "test note", test=True)
    update_predict_log("all", [50000], None, "2019-01-15",
                       "000:00:01", 0.1, test=True)
    logs = read_log("predict", test=True)
    print("Predict logs:", logs)
    logs = read_log("train", test=True)
    print("Train logs:", logs)
