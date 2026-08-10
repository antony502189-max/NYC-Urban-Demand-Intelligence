# NYC Urban Demand Intelligence

Production-oriented data science project for forecasting hourly Yellow Taxi pickup demand across New York City taxi zones.

The project is designed as an end-to-end forecasting system rather than a single notebook: reproducible data ingestion, validation, feature generation, leakage-safe backtesting, model training, experiment tracking, API serving, tests, and monitoring-ready outputs.

## Problem

Given historical NYC TLC Yellow Taxi trips, predict the number of pickups for each taxi zone for future hourly horizons. The initial targets are **1h, 6h, and 24h** ahead.

Business use cases include fleet positioning, driver supply planning, service-level monitoring, and identification of unusual demand patterns.

## Data

Primary source: NYC Taxi & Limousine Commission (TLC) Trip Record Data. Yellow Taxi trip records are published monthly in Parquet format and include pickup/drop-off timestamps, location IDs, trip attributes, fares, and payment information.

Large raw datasets and generated artifacts are intentionally excluded from Git. The pipeline downloads source files reproducibly.

## Architecture

```text
NYC TLC Parquet
      |
      v
Data ingestion -> validation -> hourly zone aggregation
                                  |
                                  v
                         leakage-safe features
                                  |
                 +----------------+----------------+
                 |                                 |
                 v                                 v
          seasonal baselines                boosting models
                 |                                 |
                 +----------------+----------------+
                                  |
                                  v
                         rolling backtesting
                                  |
                                  v
                     experiment/model registry
                                  |
                       +----------+----------+
                       |                     |
                       v                     v
                  FastAPI serving       monitoring
```

## Planned modelling stack

- Seasonal naive and moving-average baselines
- LightGBM / CatBoost / XGBoost
- Lag, rolling-window, calendar and zone-level features
- Expanding-window time-series validation
- MAE, RMSE, WAPE and sMAPE
- Quantile forecasts / prediction intervals
- SHAP-based diagnostics
- MLflow experiment tracking and model registry

## Repository layout

```text
configs/                       Runtime and training configuration
src/nyc_demand/
  data/                        Download, validation, aggregation
  features/                    Leakage-safe feature engineering
  models/                      Baselines and train/evaluate code
  api/                         FastAPI inference service
  monitoring/                  Drift and performance checks
scripts/                       CLI entry points
notebooks/                     EDA/research only
reports/                       Generated figures and model cards
tests/                         Unit and integration tests
.github/workflows/             CI
```

## Quick start

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

Download one month of Yellow Taxi data:

```bash
python -m nyc_demand.data.download --year 2024 --month 1
```

Aggregate trips to hourly pickup demand:

```bash
python -m nyc_demand.data.aggregate \
  --input data/raw/yellow_tripdata_2024-01.parquet \
  --output data/processed/demand_2024-01.parquet
```

Run tests:

```bash
pytest
```

## Modelling contract

The forecasting target for zone `z` and hour `t` is:

```text
y[z, t] = number of valid Yellow Taxi pickups in zone z during hour t
```

All lag and rolling features must be computable using information available strictly before the prediction timestamp. Random train/test splitting is prohibited for model evaluation because it would leak future information into training.

## Status

**Stage 1 — foundation and reproducible data pipeline: in progress.**

Next milestones:

1. deterministic ingestion and schema checks;
2. hourly zone demand mart;
3. temporal/geospatial EDA;
4. leakage-safe feature pipeline;
5. baseline backtesting;
6. boosting models and tuning;
7. uncertainty estimation;
8. MLflow tracking;
9. FastAPI inference;
10. drift/performance monitoring.

## License

MIT. NYC TLC source data remains subject to the terms and notices of its publisher.
