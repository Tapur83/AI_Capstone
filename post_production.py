#!/usr/bin/env python
"""
Post-production analysis script for the AI Workflow Capstone project.

Investigates the relationship between model performance and the business metric
(revenue). The business metric for AAVAIL is next-30-day revenue prediction
accuracy, which directly impacts leadership's ability to make financial projections.

This script:
1. Calculates model performance metrics (RMSE, MAE, R²)
2. Calculates business metrics (revenue forecast accuracy, % error)
3. Investigates their relationship over time
4. Produces visualizations
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cslib import fetch_ts, engineer_features

FIGURES_DIR = os.path.join(".", "figures")


def post_production_analysis(train_dir, prod_dir=None, country='all'):
    """
    Perform post-production analysis linking model performance to business metrics.

    The business metric is 30-day revenue forecast accuracy.
    When predictions are accurate, leadership can make better financial projections.

    Parameters
    ----------
    train_dir : str
        Training data directory.
    prod_dir : str, optional
        Production data directory for out-of-sample evaluation.
    country : str
        Country to analyze.

    Returns
    -------
    dict
        Analysis results.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("POST-PRODUCTION ANALYSIS")
    print("=" * 60)

    # ================================================================
    # 1. Train model and get predictions on all available data
    # ================================================================
    print("\n[analysis] Loading and processing data...")
    ts_data = fetch_ts(train_dir)
    df = ts_data[country]
    X, y, dates = engineer_features(df, training=True)

    # Time-series aware split: use last 25% as "production" holdout
    split_idx = int(0.75 * len(X))
    X_train, X_hold = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_hold = y[:split_idx], y[split_idx:]
    dates_train, dates_hold = dates[:split_idx], dates[split_idx:]

    # Train model
    print("[analysis] Training model for analysis...")
    model = Pipeline([('scaler', StandardScaler()),
                      ('rf', RandomForestRegressor(n_estimators=20, random_state=42))])
    model.fit(X_train, y_train)

    y_pred_hold = model.predict(X_hold)

    # ================================================================
    # 2. Model Performance Metrics
    # ================================================================
    overall_rmse = np.sqrt(mean_squared_error(y_hold, y_pred_hold))
    overall_mae = mean_absolute_error(y_hold, y_pred_hold)
    overall_r2 = r2_score(y_hold, y_pred_hold)
    overall_mape = np.mean(np.abs((y_hold - y_pred_hold) / (y_hold + 1e-10))) * 100

    print(f"\n[analysis] Model Performance (holdout set):")
    print(f"  RMSE:  {overall_rmse:,.0f}")
    print(f"  MAE:   {overall_mae:,.0f}")
    print(f"  R²:    {overall_r2:.4f}")
    print(f"  MAPE:  {overall_mape:.1f}%")

    # ================================================================
    # 3. Business Metric: Revenue Forecast Accuracy
    # ================================================================
    # The business metric is how accurately we can predict the next 30 days of revenue.
    # A lower MAPE means leadership can make more reliable financial decisions.

    # Calculate rolling prediction error over time
    abs_errors = np.abs(y_hold - y_pred_hold)
    pct_errors = abs_errors / (y_hold + 1e-10) * 100

    # Business value: forecast reliability
    forecast_accuracy = 100 - overall_mape
    total_actual_revenue = y_hold.sum()
    total_predicted_revenue = y_pred_hold.sum()
    revenue_forecast_error_pct = abs(total_predicted_revenue - total_actual_revenue) / total_actual_revenue * 100

    print(f"\n[analysis] Business Metrics:")
    print(f"  Forecast Accuracy:     {forecast_accuracy:.1f}%")
    print(f"  Total Actual Revenue:  {total_actual_revenue:,.0f}")
    print(f"  Total Predicted Rev:   {total_predicted_revenue:,.0f}")
    print(f"  Revenue Forecast Error: {revenue_forecast_error_pct:.2f}%")

    # ================================================================
    # 4. Relationship Analysis
    # ================================================================
    # Split holdout into monthly chunks to analyze how performance varies
    dates_hold_dt = pd.to_datetime(dates_hold)
    monthly_df = pd.DataFrame({
        'date': dates_hold_dt,
        'actual': y_hold,
        'predicted': y_pred_hold,
        'abs_error': abs_errors,
        'pct_error': pct_errors
    })
    monthly_df['year_month'] = monthly_df['date'].dt.to_period('M')

    monthly_stats = monthly_df.groupby('year_month').agg({
        'actual': 'mean',
        'predicted': 'mean',
        'abs_error': 'mean',
        'pct_error': 'mean'
    }).reset_index()

    print(f"\n[analysis] Monthly performance breakdown:")
    for _, row in monthly_stats.iterrows():
        print(f"  {row['year_month']}: Actual={row['actual']:,.0f}, "
              f"Predicted={row['predicted']:,.0f}, "
              f"Error={row['pct_error']:.1f}%")

    # ================================================================
    # 5. Visualization: Predicted vs Actual Revenue Over Time
    # ================================================================
    print("\n[analysis] Generating visualizations...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    fig.suptitle(f'Post-Production Analysis: Model Performance vs Business Metric ({country})',
                 fontsize=14, fontweight='bold')

    # Plot 1: Actual vs Predicted over time
    ax = axes[0]
    ax.plot(dates_hold_dt, y_hold, label='Actual 30-day Revenue',
            color='royalblue', linewidth=1.5)
    ax.plot(dates_hold_dt, y_pred_hold, label='Predicted 30-day Revenue',
            color='darkorange', linewidth=1.5, linestyle='--')
    ax.fill_between(dates_hold_dt, y_hold, y_pred_hold,
                    alpha=0.15, color='red', label='Prediction Error')
    ax.set_ylabel('30-day Revenue')
    ax.set_title('Revenue Forecast: Actual vs Predicted')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Plot 2: Prediction error over time
    ax = axes[1]
    ax.bar(dates_hold_dt, pct_errors, color='crimson', alpha=0.6, width=1.0)
    ax.axhline(y=overall_mape, color='black', linestyle='--',
               label=f'Avg MAPE: {overall_mape:.1f}%')
    ax.set_ylabel('Percentage Error (%)')
    ax.set_title('Prediction Error Over Time (Business Impact)')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Plot 3: Scatter — actual revenue vs prediction error
    ax = axes[2]
    scatter = ax.scatter(y_hold, pct_errors, c=np.arange(len(y_hold)),
                         cmap='viridis', alpha=0.6, edgecolors='k', linewidths=0.3)
    ax.set_xlabel('Actual 30-day Revenue')
    ax.set_ylabel('Prediction Error (%)')
    ax.set_title('Relationship: Revenue Level vs Prediction Accuracy')
    plt.colorbar(scatter, ax=ax, label='Time Index')

    # Calculate correlation
    corr = np.corrcoef(y_hold, pct_errors)[0, 1]
    ax.text(0.02, 0.95, f'Correlation: {corr:.3f}',
            transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "post_production_analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[analysis] Saved analysis plot to {fig_path}")

    # ================================================================
    # 6. Summary findings
    # ================================================================
    findings = {
        "model_performance": {
            "rmse": round(float(overall_rmse), 2),
            "mae": round(float(overall_mae), 2),
            "r2": round(float(overall_r2), 4),
            "mape": round(float(overall_mape), 2)
        },
        "business_metrics": {
            "forecast_accuracy_pct": round(float(forecast_accuracy), 2),
            "total_actual_revenue": round(float(total_actual_revenue), 2),
            "total_predicted_revenue": round(float(total_predicted_revenue), 2),
            "revenue_forecast_error_pct": round(float(revenue_forecast_error_pct), 2)
        },
        "relationship": {
            "revenue_error_correlation": round(float(corr), 4),
            "interpretation": (
                "The correlation between revenue level and prediction error "
                f"is {corr:.3f}. " +
                ("A negative correlation suggests the model performs better (lower % error) "
                 "for higher-revenue periods, which is favorable for business planning."
                 if corr < -0.1 else
                 "A positive correlation suggests the model has higher % error "
                 "for higher-revenue periods, which warrants attention."
                 if corr > 0.1 else
                 "The near-zero correlation suggests prediction accuracy is "
                 "relatively consistent across different revenue levels.")
            )
        },
        "conclusion": (
            f"The model achieves a forecast accuracy of {forecast_accuracy:.1f}%, "
            f"with a total revenue forecast error of {revenue_forecast_error_pct:.2f}%. "
            f"This means leadership can rely on the model's 30-day revenue projections "
            f"with reasonable confidence for financial planning purposes."
        )
    }

    print(f"\n[analysis] Key Findings:")
    print(f"  • Model MAPE: {overall_mape:.1f}% → Forecast Accuracy: {forecast_accuracy:.1f}%")
    print(f"  • Revenue-Error Correlation: {corr:.3f}")
    print(f"  • {findings['relationship']['interpretation']}")
    print(f"  • {findings['conclusion']}")

    return findings


if __name__ == "__main__":
    train_dir = os.path.join(".", "cs-train")
    prod_dir = os.path.join(".", "cs-production")

    findings = post_production_analysis(train_dir, prod_dir, country='all')

    with open("post_production_findings.json", "w") as f:
        json.dump(findings, f, indent=2)
    print(f"\nFindings saved to post_production_findings.json")
