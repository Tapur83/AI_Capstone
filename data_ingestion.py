#!/usr/bin/env python
"""
Data ingestion module for the AI Workflow Capstone project.
Provides automated data loading, validation, and preparation.
"""

import os
import re
import numpy as np
import pandas as pd
from cslib import fetch_data, fetch_ts, convert_to_ts


def ingest_data(data_dir):
    """
    Load, validate, and return cleaned invoice data from JSON files.

    Parameters
    ----------
    data_dir : str
        Path to directory containing invoice JSON files.

    Returns
    -------
    pd.DataFrame
        Cleaned and validated DataFrame with all invoice data.
    """
    print(f"[data_ingestion] Loading data from: {data_dir}")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    if not json_files:
        raise ValueError(f"No JSON files found in: {data_dir}")

    print(f"[data_ingestion] Found {len(json_files)} JSON files")

    # Use existing fetch_data function
    df = fetch_data(data_dir)

    # Validation
    required_cols = ['country', 'customer_id', 'day', 'invoice', 'month',
                     'price', 'stream_id', 'times_viewed', 'year']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n_rows = df.shape[0]
    n_countries = df['country'].nunique()
    date_range = (df['invoice_date'].min(), df['invoice_date'].max())

    print(f"[data_ingestion] Loaded {n_rows} records")
    print(f"[data_ingestion] {n_countries} unique countries")
    print(f"[data_ingestion] Date range: {date_range[0]} to {date_range[1]}")

    # Check for nulls
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"[data_ingestion] WARNING: Found null values:")
        print(null_counts[null_counts > 0])

    # Check for negative prices
    neg_price = (df['price'] < 0).sum()
    if neg_price > 0:
        print(f"[data_ingestion] Note: {neg_price} records with negative price (returns/refunds)")

    return df


def ingest_ts_data(data_dir):
    """
    Load and return time-series formatted data ready for modeling.

    Parameters
    ----------
    data_dir : str
        Path to directory containing invoice JSON files.

    Returns
    -------
    dict
        Dictionary mapping country names to time-series DataFrames.
    """
    print(f"[data_ingestion] Preparing time-series data from: {data_dir}")
    ts_data = fetch_ts(data_dir)
    print(f"[data_ingestion] Created time-series for {len(ts_data)} countries/aggregates")
    for key, df in ts_data.items():
        print(f"  - {key}: {df.shape[0]} days, revenue range [{df['revenue'].min():.0f}, {df['revenue'].max():.0f}]")
    return ts_data


def get_data_summary(data_dir):
    """
    Generate a summary report of the data.

    Returns
    -------
    dict
        Summary statistics.
    """
    df = ingest_data(data_dir)
    summary = {
        'n_records': df.shape[0],
        'n_countries': df['country'].nunique(),
        'n_customers': df['customer_id'].nunique(),
        'date_range': (str(df['invoice_date'].min()), str(df['invoice_date'].max())),
        'total_revenue': df['price'].sum(),
        'top_countries': df.groupby('country')['price'].sum().sort_values(ascending=False).head(10).to_dict()
    }
    return summary


if __name__ == "__main__":
    import json

    data_dir = os.path.join(".", "cs-train")
    print("=" * 60)
    print("DATA INGESTION TEST")
    print("=" * 60)

    # Test raw data ingestion
    df = ingest_data(data_dir)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Test time-series ingestion
    ts_data = ingest_ts_data(data_dir)

    # Print summary
    summary = get_data_summary(data_dir)
    print(f"\nData Summary:")
    print(json.dumps({k: str(v) for k, v in summary.items()}, indent=2))
