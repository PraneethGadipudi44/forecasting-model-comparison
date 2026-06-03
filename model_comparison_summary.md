# Forecast Model Comparison on the NOAA CO2 Series

## Dataset
For this comparison, I used the public NOAA Mauna Loa daily CO2 dataset. I picked it
because it is not a toy dataset. It has a long history, a clear upward trend, a strong
yearly pattern, and some messy values that have to be cleaned before modeling. That makes
it a good dataset for comparing forecasting approaches in a realistic way.

- Source URL: https://gml.noaa.gov/aftp/data/trace_gases/co2/in-situ/surface/txt/co2_mlo_surface-insitu_1_ccgg_DailyData.txt
- Monthly modeling window: 1974-05-01 to 2024-12-01
- Raw rows parsed: 18628
- Valid daily observations kept: 15324
- Invalid daily observations removed: 3304
- Monthly observations used: 608
- Internal monthly gaps filled after resampling: 8

## Train/Test Split
I converted the daily readings into monthly average CO2 values and then split the series in
time order. The last 36 months were held out for testing, and
everything before that was used for training. I kept the split chronological so each model
was evaluated the same way it would be used in a real forecasting setting.

- Training window: 1974-05-01 to 2021-12-01
- Test window: 2022-01-01 to 2024-12-01

## Models Compared
- Seasonal Naive
- Holt-Winters Exponential Smoothing
- SARIMA
- Prophet

## Performance Metrics
| Model | MAE | RMSE | MAPE (%) | R-squared |
| --- | ---: | ---: | ---: | ---: |
| Holt-Winters | 1.0313 | 1.3291 | 0.2445 | 0.8081 |
| Prophet | 0.9494 | 1.3473 | 0.2257 | 0.8028 |
| SARIMA | 1.0679 | 1.3713 | 0.2532 | 0.7958 |
| Seasonal Naive | 4.5946 | 5.3928 | 1.0876 | -2.1588 |

## Findings
- On this test split, **Holt-Winters** came out on top by RMSE, so it is the safest model to recommend from this run.
- The gap between the top two models was only 0.0182 RMSE points, so the winner was better, but not by a huge margin.
- **Seasonal Naive** was clearly the weakest model here because it mostly repeated last year's pattern and did not keep up with the longer-term rise in CO2.

### Model notes
- **Holt-Winters**: RMSE 1.3291, MAE 1.0313, MAPE 0.2445%, R-squared 0.8081. This model handled the shape of the series well because the data has a smooth trend and a repeating yearly cycle. I fit it with: Additive trend + additive annual seasonality (period 12).
- **Prophet**: RMSE 1.3473, MAE 0.9494, MAPE 0.2257%, R-squared 0.8028. Prophet was very competitive and actually had the lowest MAE and MAPE, which means its average point-by-point errors were a little smaller even though its RMSE was not the best. I fit it with: Additive trend model with yearly seasonality enabled for monthly data.
- **SARIMA**: RMSE 1.3713, MAE 1.0679, MAPE 0.2532%, R-squared 0.7958. SARIMA was solid, but it did not beat the simpler smoothing model on this split. It feels like the kind of model that could improve with more tuning. I fit it with: Best AIC grid-search model order=(0, 1, 1), seasonal_order=(0, 1, 1, 12), AIC=305.19.
- **Seasonal Naive**: RMSE 5.3928, MAE 4.5946, MAPE 1.0876%, R-squared -2.1588. It did its job as a baseline, but it was too simple for a series with both strong seasonality and a steady upward trend. I fit it with: Repeats the final 12 training months across the forecast horizon.

## Recommendation
If I had to choose one model from this experiment, I would go with **Holt-Winters**. It had
the lowest RMSE on the holdout period and handled the overall trend and seasonality well.
That said, the result is not one-sided. Prophet was very close, and depending on whether I
care more about RMSE or simpler interpretation, I could make a case for either one. SARIMA
was still reasonable, but it did not give me enough upside here to justify the extra tuning.
The Seasonal Naive model was useful as a baseline, but it was not competitive on this data.
