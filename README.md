# Forecasting Model Comparison

This project compares four time series forecasting models on the public NOAA Mauna Loa CO2 dataset:

- Seasonal Naive
- Holt-Winters Exponential Smoothing
- SARIMA
- Prophet

The goal is to see how different forecasting approaches behave on the same time series using the same train/test split, the same evaluation metrics, and the same visual comparison framework.

## Dataset

The project uses the NOAA Mauna Loa daily CO2 dataset, a strong choice for forecasting because it includes:

- a long historical record
- a clear upward trend
- strong yearly seasonality
- some irregular or invalid values that need preprocessing

Source:  
https://gml.noaa.gov/aftp/data/trace_gases/co2/in-situ/surface/txt/co2_mlo_surface-insitu_1_ccgg_DailyData.txt

## What the Script Does

The script:

- downloads or reuses the public NOAA dataset
- removes invalid daily readings
- converts the daily values into monthly average CO2 values
- splits the series chronologically into training and test sets
- trains four forecasting models
- evaluates each model using:
  - MAE
  - RMSE
  - MAPE
  - R-squared
- generates comparison plots
- writes a short markdown summary with findings and recommendations

## Project Files

- `model_comparison.py` — main Python script
- `model_comparison_summary.md` — summary of results and conclusions
- `plots/forecast_grid.png` — forecast vs actual view for each model
- `plots/forecast_overlay.png` — all model forecasts overlaid on the actual series
- `plots/error_metrics.png` — comparison chart for key error metrics
- `plots/residuals_boxplot.png` — residual spread comparison across models

## Models Compared

### Seasonal Naive
A simple baseline that repeats the value from the same month in the previous year.

### Holt-Winters Exponential Smoothing
A smoothing model that captures level, trend, and seasonality directly.

### SARIMA
A classical time series model that handles autoregressive, differencing, moving average, and seasonal structure.

### Prophet
A forecasting model designed to model trend and seasonality in an interpretable way.

## Evaluation Strategy

The time series is split in chronological order, with the last 36 months reserved for testing.

This means:

- all models are trained only on earlier data
- all forecasts are made on the same unseen test period
- the comparison stays fair and realistic

The evaluation metrics used in this project are:

- MAE
- RMSE
- MAPE
- R-squared

## Results Summary

From the current run:

- Holt-Winters produced the best RMSE
- Prophet was very close and had the best MAE and MAPE
- SARIMA performed reasonably well but did not outperform Holt-Winters
- Seasonal Naive worked as a baseline but was clearly the weakest overall

For the full discussion, see `model_comparison_summary.md`.

## How to Run

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels prophet
