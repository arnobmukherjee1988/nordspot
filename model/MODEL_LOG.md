# SE3 Electricity Price Forecast — Model Log

All training runs are appended automatically by `ml/train.py`.
Use `--note` to document what changed.  KPIs evaluated on a 90-day holdout.

Metrics key:
- **MAE** — Mean Absolute Error (EUR/MWh). Primary point-forecast metric.
- **RMSE** — Root Mean Squared Error (EUR/MWh). Penalises large errors more.
- **Coverage** — Fraction of actuals inside [q05, q95]. Target: ~90%.
- **Spike MAE** — MAE conditioned on actual price > 100 EUR/MWh.
- **Night MAE** — MAE for hours 00–05 and 23.
- **Peak MAE** — MAE for hours 08–09 and 17–20.

---

## Baseline — 2026-05-13 (before v2 improvements)

**Train:** 2020-01-01 → 2026-02-12  (23,943 labelled rows — data truncated by
`retention="medium"` TTL; only 3 years readable)
**Holdout:** 2026-02-12 → 2026-05-13

**State:** Baseline v2 — LightGBM (3 independent quantile models) + LEAR (fixed alpha=0.001, full in-sample residuals). Features: 3 price lags (24h/48h/168h), 2 rolling stats, standard calendar, 3 weather vars, 1 weather interaction.

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM | 22.13 | 28.98 | 63.5% | — | — | — |
| LEAR | 21.33 | 27.85 | 82.6% | — | — | — |

**Observations:**
- Coverage far below 90% target for LightGBM (overconfident intervals).
- LEAR better calibrated but MAE slightly higher.
- Only 45% of feature rows were labelled due to retention TTL bug.

---

## Fix — 2026-05-13: retention="forever" for all historical data

**Change:** Re-fetched all prices and weather with `retention="forever"`.  
ClickHouse TTL for `medium` tier is 1095 days (3 years from `valid_time`),  
causing deletion of all data before 2023-05-15.

**Result after re-fetch + retrain:**

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM | 21.50 | 28.11 | 71.3% | — | — | — |
| LEAR | 22.45 | 29.50 | 91.7% | — | — | — |

**Observations:**
- 100% labelled rows now (53,463 samples vs 23,943 before).
- LightGBM coverage improved 63.5% → 71.3% but still well below target.
- LEAR coverage 91.7% — nearly perfectly calibrated.
- MAE barely changed despite doubling training data: Feb–May 2026 is a
  structurally volatile period (high renewables, frequent near-zero prices).

---

## v2 Model Improvements — Applied 2026-05-13

### Changes to LightGBM (`ml/models/lgbm.py`)

| Parameter | Before | After | Rationale |
|---|---|---|---|
| `num_leaves` | 63 | 127 | More expressive trees; offset by stronger regularisation below |
| `min_child_samples` | 20 | 50 | Primary anti-overfitting guard — each leaf needs ≥50 samples |
| `colsample_bytree` | 0.8 | 0.6 | Stronger feature-level bagging → more diverse ensemble |
| `subsample` | 0.8 | 0.7 | Stronger row-level bagging |
| `reg_alpha` | 0.1 | 0.3 | Increased L1 penalty → sparse feature weights |
| `reg_lambda` | 0.1 | 0.3 | Increased L2 penalty → smaller weights overall |
| `learning_rate` | 0.05 | 0.03 | Slower convergence → finer optimisation |
| `n_estimators` | 1000 (fixed) | 3000 + early stopping (50 rounds) | Actual count determined by held-out 15% validation window |
| Early stopping | None | 15% of training data held out | Prevents overfitting by stopping when val loss plateaus |
| Sample weights | Uniform | Exponential decay, half-life = 365 days | Recent hours weighted more heavily to adapt to regime changes |

### Changes to LEAR (`ml/models/lear.py`)

| Change | Before | After | Rationale |
|---|---|---|---|
| Regularisation (alpha) | Fixed 0.001 (Lago et al. default) | LassoCV, TimeSeriesSplit(3), alphas=[1e-4…5e-1] | Data-driven selection; optimal alpha varies across hours and regimes |
| Cross-hour lags | None | `price_lag23h`, `price_lag25h` added | Breaks hour-silo assumption — adjacent hours share information |
| Residual quantile window | All in-sample residuals | Rolling last 365 days | Intervals adapt to current volatility; stale history dilutes the estimate |

### New features (`pipeline/features.py`)

| Feature | Type | Rationale |
|---|---|---|
| `price_lag23h` | AR lag | Adjacent-hour price from yesterday (cross-hour context) |
| `price_lag25h` | AR lag | Adjacent-hour price from yesterday (cross-hour context) |
| `price_lag72h` | AR lag | 3-day lag — captures Mon/Fri vs midweek patterns |
| `price_lag336h` | AR lag | 2-week lag — nuclear outage scheduling cycles in SE3 |
| `hour_x_month` | Interaction | Seasonal time-of-day pattern (winter peak longer than summer) |
| `weekend_x_hour` | Interaction | Weekend vs weekday daily profile differs by hour |
| `temp_x_hour` | Interaction | Temperature effect on demand varies by hour (morning ramp-up) |

### Anti-overfitting measures applied
- `min_child_samples=50` (LightGBM): prevents leaf splits on sparse spike observations.
- `colsample_bytree=0.6` + `subsample=0.7` (LightGBM): ensemble diversity via stochastic regularisation.
- `reg_alpha=0.3, reg_lambda=0.3` (LightGBM): explicit penalty on large weights.
- Early stopping on held-out 15% window (LightGBM): training stops when generalisation degrades.
- LassoCV cross-validation (LEAR): alpha selected by test error, not assumed.
- Rolling residual window (LEAR): avoids memorising historical volatility patterns that no longer apply.

---

---

## Run — 2026-05-13 19:38 UTC

**Train:** 2020-01-01 → 2026-02-12  (53,635 labelled rows)  
**Holdout:** 2026-02-12 → 2026-05-13

**Note:** v2 improvements: LassoCV alpha, cross-hour lags 23h/25h, 72h/336h lags, hour_x_month/weekend_x_hour/temp_x_hour interactions, LightGBM early stopping + recency weights + stronger regularisation

### Before

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 21.50 | 28.11 | 71.3% | 39.51 | 21.11 | 25.62 |
| LEAR       | 22.45 | 29.50 | 91.7% | 38.53 | 21.00 | 27.47 |

### After

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 21.73 | 28.32 | 80.0% | 42.61 | 21.78 | 26.22 |
| LEAR       | 21.11 | 27.54 | 91.7% | 37.42 | 20.80 | 25.27 |

### Delta vs previous run

| | MAE Δ | Coverage Δ |
|---|---|---|
| LightGBM | ▲0.23 ⚠️ | ▲8.68 ✅ |
| LEAR | ▼1.34 ✅ | ▲0.04 ✅ |

**LightGBM early stopping best iterations:** 357, 391, 110


---

## Run — 2026-05-13 20:07 UTC

**Train:** 2020-01-01 → 2026-02-12  (53,636 labelled rows)  
**Holdout:** 2026-02-12 → 2026-05-13

**Note:** Store forecasts to TimeDB, add training cache

### Before

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 21.73 | 28.32 | 80.0% | 42.61 | 21.78 | 26.22 |
| LEAR       | 21.11 | 27.54 | 91.7% | 37.42 | 20.80 | 25.27 |

### After

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 21.72 | 28.31 | 80.0% | 42.61 | 21.78 | 26.18 |
| LEAR       | 21.10 | 27.55 | 91.7% | 37.42 | 20.80 | 25.27 |

### Delta vs previous run

| | MAE Δ | Coverage Δ |
|---|---|---|
| LightGBM | ▼0.01 ✅ | ▼0.01 ⚠️ |
| LEAR | ▼0.00 ✅ | ▼0.00 ⚠️ |

**LightGBM early stopping best iterations:** 357, 391, 110


---

## Run — 2026-05-14 07:09 UTC

**Train:** 2020-01-01 → 2026-02-13  (53,647 labelled rows)  
**Holdout:** 2026-02-13 → 2026-05-14

**Note:** Routine training run

### Before

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 21.72 | 28.31 | 80.0% | 42.61 | 21.78 | 26.18 |
| LEAR       | 21.10 | 27.55 | 91.7% | 37.42 | 20.80 | 25.27 |

### After

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 21.59 | 28.28 | 79.7% | 42.94 | 21.29 | 26.31 |
| LEAR       | 21.18 | 27.62 | 91.7% | 37.42 | 20.98 | 25.29 |

### Delta vs previous run

| | MAE Δ | Coverage Δ |
|---|---|---|
| LightGBM | ▼0.13 ✅ | ▼0.34 ⚠️ |
| LEAR | ▲0.07 ⚠️ | ▼0.05 ⚠️ |

**LightGBM early stopping best iterations:** 307, 336, 114


---

## Evaluation Run — 2026-05-14 18:39 UTC

**Mode:** quick (no retraining)  **Period:** 2026-02-13 → 2026-05-14

**Note:** Existing trained models; conformal correction applied to LGBM.

| Model | CRPS | MAE | Coverage | Spike MAE | Interval Width |
|---|---|---|---|---|---|
| LGBM | 15.481 | 21.60 | 79.7% | 42.94 | 74.40 |
| LEAR | 15.595 | 21.13 | 91.7% | 37.42 | 106.70 |


---

## Run — 2026-05-14 18:40 UTC

**Train:** 2020-01-01 → 2026-02-13  (53,658 labelled rows)  
**Holdout:** 2026-02-13 → 2026-05-14

**Note:** Conformal calibration + run_eval runner added

### Before

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 21.59 | 28.28 | 79.7% | 42.94 | 21.29 | 26.31 |
| LEAR       | 21.18 | 27.62 | 91.7% | 37.42 | 20.98 | 25.29 |

### After

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 21.74 | 28.43 | 79.5%→90.1% ✅ | 43.65 | 21.60 | 26.39 |
| LEAR       | 21.26 | 27.69 | 91.6% | 37.42 | 20.98 | 25.41 |

### Delta vs previous run

| | MAE Δ | Coverage Δ |
|---|---|---|
| LightGBM | ▲0.15 ⚠️ | ▲10.44 ✅ |
| LEAR | ▲0.08 ⚠️ | ▼0.05 ⚠️ |

**LightGBM early stopping best iterations:** 310, 189, 112


---

## Run — 2026-05-14 18:45 UTC

**Train:** 2020-01-01 → 2026-02-13  (53,658 labelled rows)  
**Holdout:** 2026-02-13 → 2026-05-14

**Note:** Conformal calibration + run_eval runner added

### Before

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 21.74 | 28.43 | 79.5%→90.1% ✅ | 43.65 | 21.60 | 26.39 |
| LEAR       | 21.26 | 27.69 | 91.6% | 37.42 | 20.98 | 25.41 |

### After

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 21.74 | 28.43 | 90.1%→79.5% ✅ | 43.65 | 21.60 | 26.39 |
| LEAR       | 21.26 | 27.69 | 91.6% | 37.42 | 20.98 | 25.41 |

### Delta vs previous run

| | MAE Δ | Coverage Δ |
|---|---|---|
| LightGBM | ▼0.00 ⚠️ | ▼10.62 ⚠️ |
| LEAR | ▼0.00 ⚠️ | ▼0.00 ⚠️ |

**LightGBM early stopping best iterations:** 310, 189, 112


---

## Run — 2026-05-15 05:33 UTC

**Train:** 2020-01-01 → 2026-02-14  (53,669 labelled rows)  
**Holdout:** 2026-02-14 → 2026-05-15

**Note:** Fix conformal re-calibration + table formatting

### Before

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 21.74 | 28.43 | 90.1%→79.5% ✅ | 43.65 | 21.60 | 26.39 |
| LEAR       | 21.26 | 27.69 | 91.6% | 37.42 | 20.98 | 25.41 |

### After

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 21.61 | 28.26 | 79.5%→90.1% ✅ | 42.30 | 21.53 | 26.26 |
| LEAR       | 21.27 | 27.73 | 91.6% | 37.42 | 21.05 | 25.41 |

### Delta vs previous run

| | MAE Δ | Coverage Δ |
|---|---|---|
| LightGBM | ▼0.13 ✅ | ▲10.62 ✅ |
| LEAR | ▲0.01 ⚠️ | ▼0.05 ⚠️ |

**LightGBM early stopping best iterations:** 334, 318, 112


---

## Run — 2026-05-15 05:44 UTC

**Train:** 2020-01-01 -> 2026-02-14  (53,669 labelled rows)  
**Holdout:** 2026-02-14 -> 2026-05-15

**Note:** Fix conformal re-calibration + table formatting

### Before

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 21.61 | 28.26 | 79.5%->90.1% [OK] | 42.30 | 21.53 | 26.26 |
| LEAR       | 21.27 | 27.73 | 91.6% | 37.42 | 21.05 | 25.41 |

### After

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 21.61 | 28.26 | 79.5%->90.1% [OK] | 42.30 | 21.53 | 26.26 |
| LEAR       | 21.13 | 27.56 | 92.3% | 36.64 | 20.97 | 25.17 |

### Delta vs previous run

| | MAE delta | Coverage delta |
|---|---|---|
| LightGBM | -0.00 [!] | -0.00 [!] |
| LEAR | -0.14 [OK] | +0.78 [OK] |

**LightGBM early stopping best iterations:** 334, 318, 112


---

## Run — 2026-06-03 14:50 UTC

**Train:** 2020-01-01 -> 2026-03-05  (54,134 labelled rows)  
**Holdout:** 2026-03-05 -> 2026-06-03

**Note:** Routine training run

### Before

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 21.61 | 28.26 | 79.5%->90.1% [OK] | 42.30 | 21.53 | 26.26 |
| LEAR       | 21.13 | 27.56 | 92.3% | 36.64 | 20.97 | 25.17 |

### After

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 23.15 | 30.10 | 81.6%->90.1% [OK] | 41.35 | 23.84 | 27.76 |
| LEAR       | 22.69 | 29.14 | 90.6% | 35.06 | 22.66 | 26.70 |

### Delta vs previous run

| | MAE delta | Coverage delta |
|---|---|---|
| LightGBM | +1.54 [!] | -0.02 [!] |
| LEAR | +1.56 [!] | -1.76 [!] |

**LightGBM early stopping best iterations:** 314, 189, 113


---

## Run — 2026-06-06 08:17 UTC

**Train:** 2020-01-01 -> 2026-03-08  (54,200 labelled rows)  
**Holdout:** 2026-03-08 -> 2026-06-06

**Note:** Routine training run

### Before

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 23.15 | 30.10 | 81.6%->90.1% [OK] | 41.35 | 23.84 | 27.76 |
| LEAR       | 22.69 | 29.14 | 90.6% | 35.06 | 22.66 | 26.70 |

### After

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 23.30 | 30.15 | 81.3%->90.1% [OK] | 40.91 | 24.11 | 27.79 |
| LEAR       | 22.59 | 28.98 | 90.6% | 34.41 | 22.47 | 26.98 |

### Delta vs previous run

| | MAE delta | Coverage delta |
|---|---|---|
| LightGBM | +0.14 [!] | -0.00 [!] |
| LEAR | -0.10 [OK] | +0.05 [OK] |

**LightGBM early stopping best iterations:** 319, 189, 115


---

## Run — 2026-06-08 11:20 UTC

**Train:** 2020-01-01 -> 2026-03-10  (54,251 labelled rows)  
**Holdout:** 2026-03-10 -> 2026-06-08

**Note:** Routine training run

### Before

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 23.30 | 30.15 | 81.3%->90.1% [OK] | 40.91 | 24.11 | 27.79 |
| LEAR       | 22.59 | 28.98 | 90.6% | 34.41 | 22.47 | 26.98 |

### After

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 23.63 | 30.59 | 81.4%->90.1% [OK] | 40.68 | 24.07 | 28.30 |
| LEAR       | 22.89 | 29.27 | 90.5% | 34.12 | 22.81 | 27.10 |

### Delta vs previous run

| | MAE delta | Coverage delta |
|---|---|---|
| LightGBM | +0.33 [!] | -0.00 [!] |
| LEAR | +0.30 [!] | -0.16 [!] |

**LightGBM early stopping best iterations:** 356, 235, 115


---

## Run — 2026-06-09 18:57 UTC

**Train:** 2020-01-01 -> 2026-03-11  (54,282 labelled rows)  
**Holdout:** 2026-03-11 -> 2026-06-09

**Note:** Routine training run

### After

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 28.31 | 34.83 | 85.4%->92.2% [OK] | 38.43 | 26.99 | 32.89 |
| LEAR       | 27.89 | 35.37 | 86.7% | 38.65 | 25.76 | 32.99 |

**LightGBM early stopping best iterations:** 344, 249, 109


---

## Run — 2026-06-09 19:35 UTC

**Train:** 2020-01-01 -> 2026-03-11  (54,283 labelled rows)  
**Holdout:** 2026-03-11 -> 2026-06-09

**Note:** Routine training run

### Before

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 28.31 | 34.83 | 85.4%->92.2% [OK] | 38.43 | 26.99 | 32.89 |
| LEAR       | 27.89 | 35.37 | 86.7% | 38.65 | 25.76 | 32.99 |

### After

| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |
|---|---|---|---|---|---|---|
| LGBM       | 28.28 | 34.81 | 85.4%->92.2% [OK] | 38.28 | 26.99 | 32.81 |
| LEAR       | 27.91 | 35.39 | 86.7% | 38.61 | 25.76 | 33.07 |

### Delta vs previous run

| | MAE delta | Coverage delta |
|---|---|---|
| LightGBM | -0.03 [OK] | -0.00 [!] |
| LEAR | +0.02 [!] | -0.00 [!] |

**LightGBM early stopping best iterations:** 344, 249, 109

