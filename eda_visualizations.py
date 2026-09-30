#!/usr/bin/env python
"""
EDA Visualizations for the AI Workflow Capstone project.
Generates exploratory data analysis plots and saves them to figures/.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cslib import fetch_data, fetch_ts

FIGURES_DIR = os.path.join(".", "figures")


def run_eda(data_dir):
    """
    Run full EDA and generate visualizations.

    Parameters
    ----------
    data_dir : str
        Path to data directory.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("[EDA] Loading data...")
    df = fetch_data(data_dir)
    ts_data = fetch_ts(data_dir)

    # ================================================================
    # 1. Revenue over time (all countries)
    # ================================================================
    print("[EDA] Plotting revenue over time...")
    ts_all = ts_data['all']
    fig, ax = plt.subplots(figsize=(14, 5))
    dates = pd.to_datetime(ts_all['date'])
    ax.plot(dates, ts_all['revenue'], color='royalblue', alpha=0.8, linewidth=1)
    ax.fill_between(dates, ts_all['revenue'], alpha=0.2, color='royalblue')
    ax.set_title('Daily Revenue Over Time (All Countries)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Revenue')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eda_revenue_over_time.png"), dpi=150)
    plt.close()

    # ================================================================
    # 2. Top 10 countries by total revenue
    # ================================================================
    print("[EDA] Plotting top countries by revenue...")
    country_revenue = df.groupby('country')['price'].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(range(len(country_revenue)), country_revenue.values, color='darkorange')
    ax.set_yticks(range(len(country_revenue)))
    ax.set_yticklabels(country_revenue.index)
    ax.set_xlabel('Total Revenue')
    ax.set_title('Top 10 Countries by Total Revenue', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    for i, (val, name) in enumerate(zip(country_revenue.values, country_revenue.index)):
        ax.text(val, i, f'  {val:,.0f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eda_top_countries.png"), dpi=150)
    plt.close()

    # ================================================================
    # 3. Price distribution
    # ================================================================
    print("[EDA] Plotting price distribution...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # All prices
    prices = df['price'].values
    prices_clipped = prices[(prices > 0) & (prices < np.percentile(prices[prices > 0], 99))]
    ax1.hist(prices_clipped, bins=50, color='slategrey', edgecolor='white', alpha=0.8)
    ax1.set_title('Price Distribution (Positive, <99th percentile)', fontsize=12)
    ax1.set_xlabel('Price')
    ax1.set_ylabel('Count')

    # Log-scale
    ax2.hist(np.log1p(prices_clipped), bins=50, color='royalblue', edgecolor='white', alpha=0.8)
    ax2.set_title('Log(Price+1) Distribution', fontsize=12)
    ax2.set_xlabel('Log(Price+1)')
    ax2.set_ylabel('Count')

    plt.suptitle('Feature Distribution: Price', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eda_price_distribution.png"), dpi=150)
    plt.close()

    # ================================================================
    # 4. Views distribution
    # ================================================================
    print("[EDA] Plotting views distribution...")
    fig, ax = plt.subplots(figsize=(10, 5))
    views = df['times_viewed'].values
    views_clipped = views[views < np.percentile(views, 99)]
    ax.hist(views_clipped, bins=50, color='darkorange', edgecolor='white', alpha=0.8)
    ax.set_title('Times Viewed Distribution (<99th percentile)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Times Viewed')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eda_views_distribution.png"), dpi=150)
    plt.close()

    # ================================================================
    # 5. Revenue by month (seasonality)
    # ================================================================
    print("[EDA] Plotting monthly revenue trends...")
    ts_all = ts_data['all'].copy()
    ts_all['date'] = pd.to_datetime(ts_all['date'])
    ts_all['month'] = ts_all['date'].dt.month
    monthly = ts_all.groupby('month')['revenue'].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(monthly.index, monthly.values, color='royalblue', edgecolor='white')
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Daily Revenue')
    ax.set_title('Average Daily Revenue by Month (Seasonality)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eda_monthly_seasonality.png"), dpi=150)
    plt.close()

    # ================================================================
    # 6. Correlation of time-series features
    # ================================================================
    print("[EDA] Plotting feature correlations...")
    ts_all_corr = ts_data['all'][['purchases', 'unique_invoices',
                                   'unique_streams', 'total_views', 'revenue']]
    corr = ts_all_corr.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black',
                    fontsize=10)
    plt.colorbar(im)
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eda_correlation_matrix.png"), dpi=150)
    plt.close()

    # ================================================================
    # 7. Top country revenue time series comparison
    # ================================================================
    print("[EDA] Plotting top country revenue time series...")
    fig, ax = plt.subplots(figsize=(14, 6))
    top_countries = ['united_kingdom', 'germany', 'france']
    colors = ['royalblue', 'darkorange', 'green']
    for country, color in zip(top_countries, colors):
        if country in ts_data:
            ts_c = ts_data[country]
            dates = pd.to_datetime(ts_c['date'])
            # Rolling average for smoother view
            revenue_smooth = pd.Series(ts_c['revenue'].values).rolling(7, min_periods=1).mean()
            ax.plot(dates, revenue_smooth, label=country.replace('_', ' ').title(),
                    color=color, linewidth=1.5)

    ax.set_title('Revenue Over Time — Top Countries (7-day Rolling Avg)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Revenue (7-day avg)')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eda_top_country_revenue.png"), dpi=150)
    plt.close()

    # ================================================================
    # 8. 30-day rolling revenue (target variable distribution)
    # ================================================================
    print("[EDA] Plotting target distribution (30-day revenue)...")
    from cslib import engineer_features
    X, y, edates = engineer_features(ts_data['all'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(y, bins=40, color='slategrey', edgecolor='white', alpha=0.8)
    ax1.set_title('Target Distribution: 30-day Revenue', fontsize=12)
    ax1.set_xlabel('30-day Revenue')
    ax1.set_ylabel('Count')

    edates_dt = pd.to_datetime(edates)
    ax2.plot(edates_dt, y, color='royalblue', linewidth=1)
    ax2.set_title('Target Over Time: 30-day Revenue', fontsize=12)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('30-day Revenue')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.suptitle('Target Variable Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eda_target_distribution.png"), dpi=150)
    plt.close()

    print(f"\n[EDA] All visualizations saved to {FIGURES_DIR}/")
    return list(os.listdir(FIGURES_DIR))


if __name__ == "__main__":
    data_dir = os.path.join(".", "cs-train")
    figures = run_eda(data_dir)
    print(f"Generated {len(figures)} figures:")
    for f in sorted(figures):
        print(f"  - {f}")
