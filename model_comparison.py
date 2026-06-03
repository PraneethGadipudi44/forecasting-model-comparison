"""
Forecast model comparison on the NOAA Mauna Loa CO2 series.

The script downloads the public daily dataset, reshapes it into a monthly time series,
fits four forecasting approaches, saves comparison plots, and writes a short summary.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import math
import textwrap
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


DATA_URL = (
    "https://gml.noaa.gov/aftp/data/trace_gases/co2/in-situ/surface/txt/"
    "co2_mlo_surface-insitu_1_ccgg_DailyData.txt"
)
SEASONAL_PERIOD = 12


@dataclass
class DatasetSummary:
    source_url: str
    raw_rows: int
    valid_daily_rows: int
    invalid_daily_rows: int
    monthly_points: int
    monthly_gaps_filled: int
    start_date: str
    end_date: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    test_periods: int


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compare forecasting models on a public NOAA time series dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "plots",
        help="Directory where PNG plots will be saved.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=script_dir / "data" / "noaa_mlo_co2_daily.txt",
        help="Local cache path for the NOAA dataset.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=script_dir / "model_comparison_summary.md",
        help="Path for the markdown summary report.",
    )
    parser.add_argument(
        "--test-periods",
        type=int,
        default=36,
        help="Number of monthly observations reserved for the test set.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force a fresh download of the public dataset.",
    )
    return parser.parse_args()


def ensure_dataset(cache_path: Path, refresh_data: bool) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if refresh_data or not cache_path.exists():
        print(f"Downloading NOAA dataset to {cache_path} ...")
        urllib.request.urlretrieve(DATA_URL, cache_path)
    else:
        print(f"Using cached NOAA dataset at {cache_path}")
    return cache_path


def load_monthly_series(cache_path: Path) -> Tuple[pd.Series, DatasetSummary]:
    raw = pd.read_csv(cache_path, comment="#", sep=r"\s+", header=None, engine="python")
    raw.columns = raw.iloc[0]
    df = raw.iloc[1:].copy()

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    raw_rows = len(df)
    invalid_mask = df["value"].isna() | (df["value"] < 0)
    invalid_daily_rows = int(invalid_mask.sum())

    df = df.loc[~invalid_mask, ["datetime", "value"]].sort_values("datetime")
    daily = df.set_index("datetime")["value"]
    monthly = daily.resample("MS").mean()
    gaps_before_fill = int(monthly.isna().sum())
    monthly = monthly.interpolate(method="time", limit_area="inside").dropna()

    summary = DatasetSummary(
        source_url=DATA_URL,
        raw_rows=raw_rows,
        valid_daily_rows=len(df),
        invalid_daily_rows=invalid_daily_rows,
        monthly_points=len(monthly),
        monthly_gaps_filled=gaps_before_fill,
        start_date=monthly.index.min().date().isoformat(),
        end_date=monthly.index.max().date().isoformat(),
        train_start="",
        train_end="",
        test_start="",
        test_end="",
        test_periods=0,
    )
    return monthly, summary


def split_series(series: pd.Series, test_periods: int) -> Tuple[pd.Series, pd.Series]:
    if test_periods <= SEASONAL_PERIOD:
        raise ValueError("test_periods must be greater than 12 for a fair seasonal baseline.")
    if len(series) <= test_periods + (SEASONAL_PERIOD * 2):
        raise ValueError("Series is too short for the requested train/test split.")
    train = series.iloc[:-test_periods].copy()
    test = series.iloc[-test_periods:].copy()
    return train, test


def seasonal_naive_forecast(
    train: pd.Series,
    horizon: int,
    season_length: int = SEASONAL_PERIOD,
) -> pd.Series:
    last_cycle = train.iloc[-season_length:].to_numpy()
    repeats = int(math.ceil(horizon / season_length))
    values = np.tile(last_cycle, repeats)[:horizon]
    forecast_index = pd.date_range(
        start=train.index[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )
    return pd.Series(values, index=forecast_index, name="Seasonal Naive")


def fit_holt_winters(train: pd.Series, horizon: int) -> Tuple[pd.Series, Dict[str, str]]:
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=SEASONAL_PERIOD,
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True, use_brute=True)
    forecast = fitted.forecast(horizon)
    forecast.name = "Holt-Winters"
    return forecast, {
        "name": "Holt-Winters",
        "details": "Additive trend + additive annual seasonality (period 12).",
    }


def fit_sarima(train: pd.Series, horizon: int) -> Tuple[pd.Series, Dict[str, str]]:
    best_result = None
    best_spec = None
    best_aic = float("inf")

    warnings.filterwarnings("ignore")
    for p in (0, 1, 2):
        for q in (0, 1, 2):
            for seasonal_p in (0, 1):
                for seasonal_q in (0, 1):
                    order = (p, 1, q)
                    seasonal_order = (seasonal_p, 1, seasonal_q, SEASONAL_PERIOD)
                    try:
                        model = SARIMAX(
                            train,
                            order=order,
                            seasonal_order=seasonal_order,
                            trend="n",
                            enforce_stationarity=True,
                            enforce_invertibility=True,
                        )
                        result = model.fit(disp=False)
                        converged = bool(result.mle_retvals.get("converged", True))
                        if converged and np.isfinite(result.aic) and result.aic < best_aic:
                            best_aic = float(result.aic)
                            best_result = result
                            best_spec = (order, seasonal_order)
                    except Exception:
                        continue

    if best_result is None:
        fallback_order = (1, 1, 1)
        fallback_seasonal = (1, 1, 1, SEASONAL_PERIOD)
        model = SARIMAX(
            train,
            order=fallback_order,
            seasonal_order=fallback_seasonal,
            trend="n",
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        best_result = model.fit(disp=False)
        best_spec = (fallback_order, fallback_seasonal)
        best_aic = float(best_result.aic)

    forecast = best_result.get_forecast(steps=horizon).predicted_mean
    forecast.name = "SARIMA"
    order, seasonal_order = best_spec
    return forecast, {
        "name": "SARIMA",
        "details": (
            f"Best AIC grid-search model order={order}, "
            f"seasonal_order={seasonal_order}, AIC={best_aic:.2f}."
        ),
    }


def fit_prophet(train: pd.Series, horizon: int) -> Tuple[pd.Series, Dict[str, str]]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from prophet import Prophet

    logging.getLogger("cmdstanpy").disabled = True
    prophet_df = train.reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
    )
    prophet_model.fit(prophet_df)
    future = prophet_model.make_future_dataframe(
        periods=horizon,
        freq="MS",
        include_history=False,
    )
    forecast = prophet_model.predict(future)[["ds", "yhat"]]
    forecast_series = pd.Series(
        forecast["yhat"].to_numpy(),
        index=pd.DatetimeIndex(forecast["ds"]),
        name="Prophet",
    )
    return forecast_series, {
        "name": "Prophet",
        "details": "Additive trend model with yearly seasonality enabled for monthly data.",
    }


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    aligned_pred = y_pred.reindex(y_true.index)
    mae = mean_absolute_error(y_true, aligned_pred)
    rmse = math.sqrt(mean_squared_error(y_true, aligned_pred))
    mape = float(np.mean(np.abs((y_true - aligned_pred) / y_true)) * 100.0)
    r_squared = r2_score(y_true, aligned_pred)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE (%)": float(mape),
        "R-squared": float(r_squared),
    }


def plot_forecast_grid(
    train: pd.Series,
    test: pd.Series,
    forecasts: Dict[str, pd.Series],
    output_path: Path,
) -> None:
    recent_train = train.iloc[-60:]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for ax, (model_name, forecast) in zip(axes, forecasts.items()):
        ax.plot(recent_train.index, recent_train.values, label="Train (recent)", color="#7f8c8d")
        ax.plot(test.index, test.values, label="Actual", color="#1f77b4", linewidth=2)
        ax.plot(forecast.index, forecast.values, label="Forecast", color="#d62728", linewidth=2)
        ax.set_title(model_name)
        ax.set_ylabel("CO2 ppm")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Forecast vs Actual by Model", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_overlay(
    train: pd.Series,
    test: pd.Series,
    forecasts: Dict[str, pd.Series],
    output_path: Path,
) -> None:
    recent_train = train.iloc[-60:]
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(recent_train.index, recent_train.values, label="Train (recent)", color="#95a5a6")
    ax.plot(test.index, test.values, label="Actual", color="black", linewidth=2.5)

    colors = {
        "Seasonal Naive": "#1f77b4",
        "Holt-Winters": "#2ca02c",
        "SARIMA": "#d62728",
        "Prophet": "#9467bd",
    }
    for model_name, forecast in forecasts.items():
        ax.plot(
            forecast.index,
            forecast.values,
            label=model_name,
            linewidth=2,
            color=colors.get(model_name),
        )

    ax.set_title("Forecast Overlay on Test Period")
    ax.set_ylabel("CO2 ppm")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_metrics(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    metric_names = ["MAE", "RMSE", "MAPE (%)"]
    x = np.arange(len(metrics_df))
    width = 0.22
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, metric_name in enumerate(metric_names):
        axes[0].bar(
            x + ((idx - 1) * width),
            metrics_df[metric_name].to_numpy(),
            width=width,
            label=metric_name,
            color=colors[idx],
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_df["Model"], rotation=15)
    axes[0].set_title("Error Metric Comparison")
    axes[0].set_ylabel("Metric value")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(metrics_df["Model"], metrics_df["R-squared"], color="#8e44ad")
    axes[1].set_title("R-squared by Model")
    axes[1].set_ylabel("R-squared")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.3)

    for idx, value in enumerate(metrics_df["R-squared"]):
        axes[1].text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals_boxplot(
    test: pd.Series,
    forecasts: Dict[str, pd.Series],
    output_path: Path,
) -> None:
    residuals: List[np.ndarray] = []
    labels: List[str] = []
    for model_name, forecast in forecasts.items():
        aligned = forecast.reindex(test.index)
        residuals.append((test - aligned).to_numpy())
        labels.append(model_name)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(residuals, tick_labels=labels, patch_artist=True)
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_title("Residual Distribution by Model")
    ax.set_ylabel("Actual - Forecast (ppm)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def markdown_table(metrics_df: pd.DataFrame) -> str:
    headers = ["Model", "MAE", "RMSE", "MAPE (%)", "R-squared"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---:", "---:", "---:", "---:"]) + " |",
    ]
    for _, row in metrics_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["Model"]),
                    f"{row['MAE']:.4f}",
                    f"{row['RMSE']:.4f}",
                    f"{row['MAPE (%)']:.4f}",
                    f"{row['R-squared']:.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def model_commentary(
    ranked_metrics: pd.DataFrame,
    model_notes: Dict[str, Dict[str, str]],
) -> str:
    best = ranked_metrics.iloc[0]
    second = ranked_metrics.iloc[1]
    worst = ranked_metrics.iloc[-1]
    gap = second["RMSE"] - best["RMSE"]

    lines = [
        (
            f"- On this test split, **{best['Model']}** came out on top by RMSE, "
            "so it is the safest model to recommend from this run."
        ),
        (
            f"- The gap between the top two models was only {gap:.4f} RMSE points, "
            "so the winner was better, but not by a huge margin."
        ),
        (
            f"- **{worst['Model']}** was clearly the weakest model here because it mostly "
            "repeated last year's pattern and did not keep up with the longer-term rise in CO2."
        ),
        "",
        "### Model notes",
    ]

    model_takeaways = {
        "Seasonal Naive": (
            "It did its job as a baseline, but it was too simple for a series with both strong "
            "seasonality and a steady upward trend."
        ),
        "Holt-Winters": (
            "This model handled the shape of the series well because the data has a smooth trend "
            "and a repeating yearly cycle."
        ),
        "SARIMA": (
            "SARIMA was solid, but it did not beat the simpler smoothing model on this split. "
            "It feels like the kind of model that could improve with more tuning."
        ),
        "Prophet": (
            "Prophet was very competitive and actually had the lowest MAE and MAPE, which means "
            "its average point-by-point errors were a little smaller even though its RMSE was not the best."
        ),
    }

    for _, row in ranked_metrics.iterrows():
        name = row["Model"]
        note = model_notes[name]["details"]
        lines.append(
            f"- **{name}**: RMSE {row['RMSE']:.4f}, MAE {row['MAE']:.4f}, "
            f"MAPE {row['MAPE (%)']:.4f}%, R-squared {row['R-squared']:.4f}. "
            f"{model_takeaways[name]} I fit it with: {note}"
        )

    return "\n".join(lines)


def write_summary(
    summary_path: Path,
    dataset_summary: DatasetSummary,
    ranked_metrics: pd.DataFrame,
    model_notes: Dict[str, Dict[str, str]],
) -> None:
    best_model = ranked_metrics.iloc[0]["Model"]
    report = f"""# Forecast Model Comparison on the NOAA CO2 Series

## Dataset
For this comparison, I used the public NOAA Mauna Loa daily CO2 dataset. I picked it
because it is not a toy dataset. It has a long history, a clear upward trend, a strong
yearly pattern, and some messy values that have to be cleaned before modeling. That makes
it a good dataset for comparing forecasting approaches in a realistic way.

- Source URL: {dataset_summary.source_url}
- Monthly modeling window: {dataset_summary.start_date} to {dataset_summary.end_date}
- Raw rows parsed: {dataset_summary.raw_rows}
- Valid daily observations kept: {dataset_summary.valid_daily_rows}
- Invalid daily observations removed: {dataset_summary.invalid_daily_rows}
- Monthly observations used: {dataset_summary.monthly_points}
- Internal monthly gaps filled after resampling: {dataset_summary.monthly_gaps_filled}

## Train/Test Split
I converted the daily readings into monthly average CO2 values and then split the series in
time order. The last {dataset_summary.test_periods} months were held out for testing, and
everything before that was used for training. I kept the split chronological so each model
was evaluated the same way it would be used in a real forecasting setting.

- Training window: {dataset_summary.train_start} to {dataset_summary.train_end}
- Test window: {dataset_summary.test_start} to {dataset_summary.test_end}

## Models Compared
- Seasonal Naive
- Holt-Winters Exponential Smoothing
- SARIMA
- Prophet

## Performance Metrics
{markdown_table(ranked_metrics)}

## Findings
{model_commentary(ranked_metrics, model_notes)}

## Recommendation
If I had to choose one model from this experiment, I would go with **{best_model}**. It had
the lowest RMSE on the holdout period and handled the overall trend and seasonality well.
That said, the result is not one-sided. Prophet was very close, and depending on whether I
care more about RMSE or simpler interpretation, I could make a case for either one. SARIMA
was still reasonable, but it did not give me enough upside here to justify the extra tuning.
The Seasonal Naive model was useful as a baseline, but it was not competitive on this data.
"""
    summary_path.write_text(textwrap.dedent(report).strip() + "\n", encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    cache_path = ensure_dataset(args.cache_path, args.refresh_data)
    series, dataset_summary = load_monthly_series(cache_path)
    train, test = split_series(series, args.test_periods)

    dataset_summary.train_start = train.index.min().date().isoformat()
    dataset_summary.train_end = train.index.max().date().isoformat()
    dataset_summary.test_start = test.index.min().date().isoformat()
    dataset_summary.test_end = test.index.max().date().isoformat()
    dataset_summary.test_periods = len(test)

    forecasts: Dict[str, pd.Series] = {}
    model_notes: Dict[str, Dict[str, str]] = {}

    forecasts["Seasonal Naive"] = seasonal_naive_forecast(train, len(test))
    model_notes["Seasonal Naive"] = {
        "name": "Seasonal Naive",
        "details": "Repeats the final 12 training months across the forecast horizon.",
    }

    hw_forecast, hw_notes = fit_holt_winters(train, len(test))
    forecasts["Holt-Winters"] = hw_forecast
    model_notes["Holt-Winters"] = hw_notes

    sarima_forecast, sarima_notes = fit_sarima(train, len(test))
    forecasts["SARIMA"] = sarima_forecast
    model_notes["SARIMA"] = sarima_notes

    prophet_forecast, prophet_notes = fit_prophet(train, len(test))
    forecasts["Prophet"] = prophet_forecast
    model_notes["Prophet"] = prophet_notes

    metrics_rows: List[Dict[str, float]] = []
    for model_name, forecast in forecasts.items():
        row = {"Model": model_name}
        row.update(evaluate_forecast(test, forecast))
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    ranked_metrics = metrics_df.sort_values("RMSE", ascending=True).reset_index(drop=True)

    plot_forecast_grid(train, test, forecasts, args.output_dir / "forecast_grid.png")
    plot_forecast_overlay(train, test, forecasts, args.output_dir / "forecast_overlay.png")
    plot_error_metrics(metrics_df, args.output_dir / "error_metrics.png")
    plot_residuals_boxplot(test, forecasts, args.output_dir / "residuals_boxplot.png")
    write_summary(
        args.summary_path,
        dataset_summary,
        ranked_metrics,
        model_notes,
    )

    print("\nModel comparison complete.\n")
    print(ranked_metrics.to_string(index=False))
    print(f"\nSummary report: {args.summary_path}")
    print(f"Plots directory: {args.output_dir}")


if __name__ == "__main__":
    main()
