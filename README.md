# Forecasting Model Comparison

This project compares four time series forecasting models on the public NOAA Mauna Loa CO2 dataset:

- Seasonal Naive
- Holt-Winters Exponential Smoothing
- SARIMA
- Prophet

The goal is to evaluate how different forecasting approaches perform on the same time series using a fair train/test split and a common set of evaluation metrics.

## Dataset

The analysis uses the NOAA Mauna Loa daily CO2 dataset, which is publicly available and well suited for forecasting because it includes:

- a long historical record
- a clear upward trend
- strong yearly seasonality
- some irregular or invalid values that require preprocessing

Source:
https://gml.noaa.gov/aftp/data/trace_gases/co2/in-situ/surface/txt/co2_mlo_surface-insitu_1_ccgg_DailyData.txt

## What This Project Does

- downloads or reuses the public NOAA dataset
- cleans invalid daily readings
- converts the series into monthly average CO2 values
- splits the data chronologically into training and testing sets
- trains four forecasting models
- evaluates each model using:
  - MAE
  - RMSE
  - MAPE
  - R-squared
- generates comparison plots
- writes a summary report with findings and recommendations

## Project Files

- `model_comparison.py` — main script for loading data, training models, evaluating results, and saving plots
- `model_comparison_summary.md` — written summary of findings
- `plots/forecast_grid.png` — forecast vs actual plot for each model
- `plots/forecast_overlay.png` — all forecasts plotted against actual values
- `plots/error_metrics.png` — metric comparison chart
- `plots/residuals_boxplot.png` — residual comparison plot

## Models Compared

### 1. Seasonal Naive
A simple baseline that repeats the value from the same season in the previous year.

### 2. Holt-Winters
A smoothing-based model that handles level, trend, and seasonality directly.

### 3. SARIMA
A classical statistical forecasting model that captures autoregressive, moving average, and seasonal behavior.

### 4. Prophet
A trend-and-seasonality forecasting model designed to work well on business and historical time series.

## Evaluation Approach

The data is split in time order so the last 36 months are used as the test set. This keeps the evaluation realistic and avoids data leakage.

Each model is trained only on the training data and then evaluated on the same test window.

## Results Summary

In this run:

- Holt-Winters achieved the best RMSE
- Prophet was very close and had the lowest MAE and MAPE
- SARIMA performed reasonably well but did not beat Holt-Winters
- Seasonal Naive worked as a useful baseline but was the weakest overall

For the full write-up, see `model_comparison_summary.md`.

## How To Run

Install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels prophet
