#!/usr/bin/env python
"""
Model module for the AI Workflow Capstone project.
Provides train, predict, and load functions for revenue forecasting.
Based on solution-guidance/model.py with fixes for modern sklearn.
"""

import time
import os
import re
import csv
import sys
import uuid
import joblib
from datetime import date
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from logger import update_predict_log, update_train_log
from cslib import fetch_ts, engineer_features

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

## model specific variables
MODEL_DIR = os.path.join(".", "models")
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "supervised learning model for time-series"


def _model_train(df, tag, test=False, model_dir=None, log_dir=None):
    """
    Train a single model for a given country/tag.

    Parameters
    ----------
    df : DataFrame
        Time-series data for one country.
    tag : str
        Country identifier.
    test : bool
        If True, subset data and save as test model.
    model_dir : str, optional
        Override default model directory.
    log_dir : str, optional
        Override default log directory.

    Returns
    -------
    dict
        Training results including rmse.
    """
    m_dir = model_dir if model_dir else MODEL_DIR

    ## start timer for runtime
    time_start = time.time()

    X, y, dates = engineer_features(df)

    if test:
        n_samples = int(np.round(0.3 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]), n_samples,
                                          replace=False).astype(int)
        mask = np.in1d(np.arange(y.size), subset_indices)
        y = y[mask]
        X = X[mask]
        dates = dates[mask]

    ## Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, random_state=42)

    ## train a random forest model
    param_grid_rf = {
        'rf__criterion': ['squared_error', 'absolute_error'],
        'rf__n_estimators': [10, 15, 20, 25]
    }

    pipe_rf = Pipeline(steps=[('scaler', StandardScaler()),
                              ('rf', RandomForestRegressor())])

    grid = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    eval_rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)))
    eval_mae = round(mean_absolute_error(y_test, y_pred))

    ## retrain using all data
    grid.fit(X, y)
    model_name = re.sub("\\.", "_", str(MODEL_VERSION))
    if test:
        saved_model = os.path.join(m_dir,
                                   "test-{}-{}.joblib".format(tag, model_name))
        print("... saving test version of model: {}".format(saved_model))
    else:
        saved_model = os.path.join(m_dir,
                                   "sl-{}-{}.joblib".format(tag, model_name))
        print("... saving model: {}".format(saved_model))

    joblib.dump(grid, saved_model)

    m, s = divmod(time.time() - time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d" % (h, m, s)

    ## update log
    update_train_log(tag, (str(dates[0]), str(dates[-1])),
                     {'rmse': eval_rmse}, runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE, test=test, log_dir=log_dir)

    return {'rmse': eval_rmse, 'mae': eval_mae, 'runtime': runtime}


def model_train(data_dir, test=False, model_dir=None, log_dir=None):
    """
    Train models for all countries in the data directory.

    Parameters
    ----------
    data_dir : str
        Path to directory containing invoice JSON files.
    test : bool
        If True, only train on subset of countries and data.
    model_dir : str, optional
        Override default model directory.
    log_dir : str, optional
        Override default log directory.

    Returns
    -------
    dict
        Training results for each country.
    """
    m_dir = model_dir if model_dir else MODEL_DIR

    if not os.path.isdir(m_dir):
        os.makedirs(m_dir, exist_ok=True)

    if test:
        print("... test flag on")
        print("...... subsetting data")
        print("...... subsetting countries")

    ## fetch time-series formatted data
    ts_data = fetch_ts(data_dir)

    ## train a different model for each data set
    results = {}
    for country, df in ts_data.items():
        if test and country not in ['all', 'united_kingdom']:
            continue
        print(f"... training model for {country}")
        results[country] = _model_train(df, country, test=test,
                                        model_dir=m_dir, log_dir=log_dir)

    return results


def model_load(prefix='sl', data_dir=None, training=True, model_dir=None):
    """
    Load trained models and associated data.

    Parameters
    ----------
    prefix : str
        Model file prefix ('sl' for production, 'test' for test).
    data_dir : str, optional
        Path to data directory.
    training : bool
        If True, trim last 30 days from features.
    model_dir : str, optional
        Override default model directory.

    Returns
    -------
    tuple
        (all_data, all_models) dictionaries keyed by country.
    """
    m_dir = model_dir if model_dir else MODEL_DIR

    if not data_dir:
        data_dir = os.path.join(".", "cs-train")

    models = [f for f in os.listdir(m_dir) if re.search(prefix, f)]

    if len(models) == 0:
        raise Exception("Models with prefix '{}' cannot be found — did you train?".format(prefix))

    all_models = {}
    for model in models:
        all_models[re.split("-", model)[1]] = joblib.load(os.path.join(m_dir, model))

    ## load data
    ts_data = fetch_ts(data_dir)
    all_data = {}
    for country, df in ts_data.items():
        X, y, dates = engineer_features(df, training=training)
        dates = np.array([str(d) for d in dates])
        all_data[country] = {"X": X, "y": y, "dates": dates}

    return (all_data, all_models)


def model_predict(country, year, month, day, all_models=None, all_data=None,
                  test=False, data_dir=None, model_dir=None, log_dir=None):
    """
    Make a revenue prediction for a specific country and date.

    Parameters
    ----------
    country : str
        Country identifier (e.g. 'all', 'united_kingdom').
    year : str
        Year string.
    month : str
        Month string.
    day : str
        Day string.
    all_models : dict, optional
        Pre-loaded models.
    all_data : dict, optional
        Pre-loaded data.
    test : bool
        If True, log to test log file.
    data_dir : str, optional
        Override data directory.
    model_dir : str, optional
        Override model directory.
    log_dir : str, optional
        Override log directory.

    Returns
    -------
    dict
        {'y_pred': array, 'y_proba': array or None}
    """
    ## start timer for runtime
    time_start = time.time()

    ## load model if needed
    if not all_models or not all_data:
        prefix = 'test' if test else 'sl'
        d_dir = data_dir if data_dir else os.path.join(".", "cs-train")
        all_data, all_models = model_load(prefix=prefix, data_dir=d_dir,
                                          training=False, model_dir=model_dir)

    ## input checks
    if country not in all_models.keys():
        raise Exception("ERROR (model_predict) - model for country '{}' could not be found".format(country))

    for d in [str(year), str(month), str(day)]:
        if re.search("\\D", str(d)):
            raise Exception("ERROR (model_predict) - invalid year, month or day")

    ## load data
    model = all_models[country]
    data = all_data[country]

    ## check date
    target_date = "{}-{}-{}".format(year, str(month).zfill(2), str(day).zfill(2))

    if target_date not in data['dates']:
        raise Exception("ERROR (model_predict) - date {} not in range {}-{}".format(
            target_date, data['dates'][0], data['dates'][-1]))

    date_indx = np.where(data['dates'] == target_date)[0][0]
    query = data['X'].iloc[[date_indx]]

    ## sanity check
    if data['dates'].shape[0] != data['X'].shape[0]:
        raise Exception("ERROR (model_predict) - dimensions mismatch")

    ## make prediction and gather data for log entry
    y_pred = model.predict(query)
    y_proba = None
    if 'predict_proba' in dir(model) and 'probability' in dir(model):
        if model.probability:
            y_proba = model.predict_proba(query)

    m, s = divmod(time.time() - time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d" % (h, m, s)

    ## update predict log
    update_predict_log(country, y_pred, y_proba, target_date,
                       runtime, MODEL_VERSION, test=test, log_dir=log_dir)

    return {'y_pred': y_pred, 'y_proba': y_proba}


def model_predict_all(year, month, day, all_models=None, all_data=None,
                      test=False, data_dir=None, model_dir=None, log_dir=None):
    """
    Make predictions for all available countries.

    Returns
    -------
    dict
        {country: {'y_pred': ..., 'y_proba': ...}, ...}
    """
    if not all_models or not all_data:
        prefix = 'test' if test else 'sl'
        d_dir = data_dir if data_dir else os.path.join(".", "cs-train")
        all_data, all_models = model_load(prefix=prefix, data_dir=d_dir,
                                          training=False, model_dir=model_dir)

    results = {}
    for country in all_models.keys():
        try:
            result = model_predict(country, year, month, day,
                                   all_models=all_models, all_data=all_data,
                                   test=test, log_dir=log_dir)
            results[country] = {'y_pred': result['y_pred'].tolist()}
        except Exception as e:
            results[country] = {'error': str(e)}

    return results


if __name__ == "__main__":
    """
    Basic test procedure for model.py
    """
    ## train the model
    print("TRAINING MODELS")
    data_dir = os.path.join(".", "cs-train")
    model_train(data_dir, test=True)

    ## load the model
    print("LOADING MODELS")
    all_data, all_models = model_load(prefix='test')
    print("... models loaded: ", ",".join(all_models.keys()))

    ## test predict
    country = 'all'
    year = '2018'
    month = '01'
    day = '05'
    result = model_predict(country, year, month, day,
                           all_models=all_models, all_data=all_data, test=True)
    print(result)
