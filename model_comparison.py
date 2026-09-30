#!/usr/bin/env python
"""
Model comparison: Random Forest (main) vs Ridge Regression (baseline).
Generates actual metrics and visualization.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cslib import fetch_ts, engineer_features

warnings.filterwarnings("ignore")

FIGURES_DIR = os.path.join(".", "figures")


def compare_models(data_dir, country='all'):
    """
    Compare Random Forest (main model) vs Ridge Regression (baseline)
    on actual data.

    Parameters
    ----------
    data_dir : str
        Path to data directory.
    country : str
        Country to evaluate on.

    Returns
    -------
    dict
        Comparison metrics for both models.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"[compare] Loading data for '{country}'...")
    ts_data = fetch_ts(data_dir)
    df = ts_data[country]
    X, y, dates = engineer_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, random_state=42)

    # ---- Baseline: Ridge Regression ----
    print("[compare] Training baseline (Ridge Regression)...")
    baseline = Pipeline([('scaler', StandardScaler()),
                         ('ridge', Ridge(alpha=1.0))])
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)

    baseline_metrics = {
        'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_pred_baseline))), 2),
        'mae': round(float(mean_absolute_error(y_test, y_pred_baseline)), 2),
        'r2': round(float(r2_score(y_test, y_pred_baseline)), 4),
        'mape': round(float(np.mean(np.abs((y_test - y_pred_baseline) / (y_test + 1e-10))) * 100), 2)
    }

    # ---- Main Model: Random Forest ----
    print("[compare] Training main model (Random Forest)...")
    main_model = Pipeline([('scaler', StandardScaler()),
                           ('rf', RandomForestRegressor(n_estimators=20,
                                                        criterion='squared_error',
                                                        random_state=42))])
    main_model.fit(X_train, y_train)
    y_pred_main = main_model.predict(X_test)

    main_metrics = {
        'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_pred_main))), 2),
        'mae': round(float(mean_absolute_error(y_test, y_pred_main)), 2),
        'r2': round(float(r2_score(y_test, y_pred_main)), 4),
        'mape': round(float(np.mean(np.abs((y_test - y_pred_main) / (y_test + 1e-10))) * 100), 2)
    }

    print(f"\n  Baseline (Ridge): RMSE={baseline_metrics['rmse']}, "
          f"MAE={baseline_metrics['mae']}, R²={baseline_metrics['r2']}")
    print(f"  Main (RF):        RMSE={main_metrics['rmse']}, "
          f"MAE={main_metrics['mae']}, R²={main_metrics['r2']}")

    # ---- Visualization: Bar chart ----
    metrics_names = ['RMSE', 'MAE', 'R²', 'MAPE (%)']
    baseline_vals = [baseline_metrics['rmse'], baseline_metrics['mae'],
                     baseline_metrics['r2'], baseline_metrics['mape']]
    main_vals = [main_metrics['rmse'], main_metrics['mae'],
                 main_metrics['r2'], main_metrics['mape']]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'Model Comparison: Random Forest vs Ridge Regression ({country})',
                 fontsize=14, fontweight='bold')

    colors = ['#2196F3', '#FF9800']
    for i, (ax, name) in enumerate(zip(axes, metrics_names)):
        bars = ax.bar(['Baseline\n(Ridge)', 'Main\n(RF)'],
                      [baseline_vals[i], main_vals[i]], color=colors)
        ax.set_title(name)
        ax.set_ylabel(name)
        for bar, val in zip(bars, [baseline_vals[i], main_vals[i]]):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "model_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[compare] Saved comparison plot to {fig_path}")

    # ---- Visualization: Actual vs Predicted scatter ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Actual vs Predicted Revenue ({country})', fontsize=14, fontweight='bold')

    ax1.scatter(y_test, y_pred_baseline, alpha=0.6, c='#FF9800', edgecolors='k', linewidths=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Revenue')
    ax1.set_ylabel('Predicted Revenue')
    ax1.set_title(f'Baseline (Ridge) — R²={baseline_metrics["r2"]:.4f}')

    ax2.scatter(y_test, y_pred_main, alpha=0.6, c='#2196F3', edgecolors='k', linewidths=0.5)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual Revenue')
    ax2.set_ylabel('Predicted Revenue')
    ax2.set_title(f'Main (RF) — R²={main_metrics["r2"]:.4f}')

    plt.tight_layout()
    fig_path2 = os.path.join(FIGURES_DIR, "actual_vs_predicted.png")
    plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[compare] Saved actual vs predicted plot to {fig_path2}")

    return {
        'baseline': baseline_metrics,
        'main_model': main_metrics,
        'improvement': {
            'rmse_pct': round((1 - main_metrics['rmse'] / (baseline_metrics['rmse'] + 1e-10)) * 100, 2),
            'mae_pct': round((1 - main_metrics['mae'] / (baseline_metrics['mae'] + 1e-10)) * 100, 2),
            'r2_improvement': round(main_metrics['r2'] - baseline_metrics['r2'], 4)
        }
    }


if __name__ == "__main__":
    import json
    data_dir = os.path.join(".", "cs-train")
    results = compare_models(data_dir, country='all')
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))

    with open("model_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to model_comparison_results.json")
