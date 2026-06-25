# NordSpot

Production-grade electricity spot price forecasting platform for the Nordic market.

NordSpot ingests real-time data from ENTSO-E and Open-Meteo, trains a suite of
tree-based ML models, and serves calibrated 24-hour ahead probabilistic forecasts
via an authenticated REST API — for all four Swedish bidding zones (SE1–SE4).

Built by [ATO Energy](https://ato.energy) as the forecasting engine for EV fleet
charge scheduling. Architecturally ready to extend to any European bidding zone.

---

## Quickstart (local Docker)

```bash
cp .env.example .env        # 1. fill in ENTSOE_API_KEY at minimum
make up                     # 2. start ClickHouse, Redis, MLflow, dashboard
make train                  # 3. fetch data and train models
```

Dashboard: http://localhost:8501
MLflow: http://localhost:5000

---

## Architecture

Data flows through three quality layers (Medallion Architecture):

| Layer | Storage | Contents |
|-------|---------|----------|
| Bronze | GCS (Parquet) | Raw ENTSO-E + weather data, immutable |
| Silver | ClickHouse + BigQuery | Cleaned, structured, queryable |
| Gold | Feast (BigQuery + Redis) | Point-in-time correct ML features |

Models: LightGBM · XGBoost · CatBoost · LEAR · Stacking Ensemble
Serving: FastAPI · MLflow Model Registry · Evidently AI drift detection

## Project Structure

```
nordspot/
├── pipeline/     # Data ingestion (ENTSO-E, Open-Meteo)
├── dbt/          # SQL transforms Bronze → Silver
├── feast/        # Feature store definitions
├── ml/           # Model training and evaluation
├── api/          # FastAPI prediction service
├── monitoring/   # Prometheus + Grafana
├── dashboard/    # Streamlit explorer
├── tests/        # pytest unit + integration tests
└── infra/        # Terraform (GCP)
```

## Development

```bash
make dev     # start lightweight stack (ClickHouse + Redis only)
make test    # run pytest
make lint    # ruff check
```

## Current model performance (SE3 holdout — 2026-02-14 to 2026-05-15)

| Model | MAE EUR/MWh | RMSE | q05–q95 Coverage |
|-------|-------------|------|-----------------|
| LightGBM (conformal) | 21.61 | 28.26 | 90.1% |
| LEAR | 21.13 | 27.56 | 92.3% |

## Zones supported

SE1 (Luleå) · SE2 (Umeå) · SE3 (Stockholm) · SE4 (Malmö)

New zones are added by editing `FORECAST_ZONES` in `.env` — no code changes required.

## License

MIT
