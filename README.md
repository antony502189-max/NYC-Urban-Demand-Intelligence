# NYC Urban Demand Intelligence

[![CI](https://github.com/antony502189-max/NYC-Urban-Demand-Intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/antony502189-max/NYC-Urban-Demand-Intelligence/actions/workflows/ci.yml)

Production-oriented data science system for forecasting hourly Yellow Taxi pickup demand across New York City taxi zones.

This repository is intentionally structured as an end-to-end forecasting product rather than a single notebook: reproducible TLC ingestion, explicit data-quality contracts, complete hourly demand marts, leakage-safe feature engineering, direct multi-horizon modelling, temporal backtesting, uncertainty calibration, experiment tracking, model persistence, an inference API, containerization, tests, and drift monitoring.

## Problem

Given historical NYC TLC Yellow Taxi trips, forecast the number of pickups in each taxi zone at **1h, 6h, and 24h** horizons.

Potential operational applications include fleet positioning, driver-supply planning, service-level monitoring, and detection of unusual demand patterns.

## Data

The primary source is NYC Taxi & Limousine Commission Trip Record Data. The pipeline downloads monthly Yellow Taxi Parquet files and keeps large raw/generated datasets outside Git.

Trip-level records are validated before aggregation. Valid rows are transformed into a complete `timestamp × zone_id` hourly grid so that a missing trip record is not confused with an absent observation: true zero-demand hours are represented explicitly.

## System architecture

```text
NYC TLC Parquet
      |
      v
atomic download -> quality validation -> hourly zone demand mart
                                           |
                                           v
                                  historical-only features
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
             weekly seasonal baseline                 LightGBM direct models
                                                            1h / 6h / 24h
                    |                                             |
                    +----------------------+----------------------+
                                           |
                                           v
                               expanding-window backtesting
                                           |
                         +-----------------+------------------+
                         |                 |                  |
                         v                 v                  v
                    Optuna tuning     conformal bands     MLflow runs
                         |                 |                  |
                         +-----------------+------------------+
                                           |
                                           v
                                   versioned model bundle
                                           |
                              +------------+------------+
                              |                         |
                              v                         v
                         FastAPI / Docker          PSI monitoring
```

## Implemented capabilities

### Data engineering

- reproducible monthly TLC Parquet downloader;
- explicit schema/domain quality checks with persisted quality reports;
- hourly pickup-demand aggregation by taxi zone;
- zero-filled complete hourly zone grid;
- Parquet-first storage using Polars/PyArrow.

### Feature engineering

- calendar features;
- zone-specific demand lags;
- shifted rolling means and standard deviations;
- strict historical-only feature construction;
- direct future targets for 1h/6h/24h models;
- target columns explicitly excluded from feature inference.

### Modelling and evaluation

- weekly seasonal-naive benchmark;
- LightGBM Poisson demand regression;
- expanding-window temporal cross-validation;
- label-boundary safeguards that prevent future validation labels entering training;
- MAE, RMSE, WAPE and sMAPE;
- Optuna hyperparameter search using temporal CV;
- split-conformal prediction intervals;
- portable LightGBM model bundles with ordered feature metadata.

### Production layer

- MLflow backtest logging;
- FastAPI `/health` and `/v1/forecast` endpoints;
- strict online feature contract validation;
- Docker image for inference;
- PSI-based numerical drift monitoring;
- model-card template;
- GitHub Actions CI with Ruff and pytest.

## Repository layout

```text
configs/                       Data/model contracts
scripts/                       Runnable training workflows
src/nyc_demand/
  api/                         FastAPI inference service
  data/                        Download, validation, aggregation
  features/                    Leakage-safe feature engineering
  models/                      Baselines, horizons, training, CV, tuning, intervals
  monitoring/                  Drift checks
  tracking/                    MLflow integration
reports/                       Model card and generated evaluation outputs
tests/                         Unit/integration tests
.github/workflows/             CI and manual maintenance workflows
Dockerfile                     Inference container
```

## Quick start

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[ml,api,dev]"
```

Download one month of Yellow Taxi data:

```bash
python -m nyc_demand.data.download --year 2024 --month 1
```

Aggregate trips into the hourly zone-demand mart:

```bash
python -m nyc_demand.data.aggregate \
  --input data/raw/yellow_tripdata_2024-01.parquet \
  --output data/processed/demand_2024-01.parquet
```

Run the test suite:

```bash
pytest
ruff check src tests
```

## Train a model

A training run builds features, executes expanding-window backtests, trains the final direct-horizon model, saves a portable model bundle, and writes a JSON evaluation report.

Example for the six-hour horizon:

```bash
python scripts/train_model.py \
  --input data/processed/demand_2024-01.parquet \
  --horizon 6 \
  --model-dir artifacts/models/h6 \
  --report reports/metrics/h6.json
```

Enable MLflow logging with `--mlflow` after installing the tracking extra:

```bash
pip install -e ".[ml,tracking]"
mlflow ui
```

## Serve a trained model

Point the service at a saved model bundle:

```bash
export NYC_DEMAND_MODEL_DIR=artifacts/models/h6
uvicorn nyc_demand.api.app:app --host 0.0.0.0 --port 8000
```

Windows PowerShell:

```powershell
$env:NYC_DEMAND_MODEL_DIR="artifacts/models/h6"
uvicorn nyc_demand.api.app:app --host 0.0.0.0 --port 8000
```

The low-level inference endpoint consumes the exact numeric feature contract stored with the model bundle. This prevents silent feature-order mismatches between training and serving.

## Docker

```bash
docker build -t nyc-demand-api .
docker run --rm -p 8000:8000 \
  -e NYC_DEMAND_MODEL_DIR=/models/h6 \
  -v "$(pwd)/artifacts/models:/models:ro" \
  nyc-demand-api
```

## Leakage contract

For zone `z`, target hour `t`, and forecast origin `o`, every input feature must be computable from information available at or before the origin. Random train/test splitting is prohibited.

For direct multi-horizon validation, training rows are additionally removed when their **future label timestamp crosses the validation boundary**. This avoids a less obvious form of label leakage that can survive otherwise chronological splitting.

## Evaluation contract

Every candidate model must beat or justify itself against a weekly seasonal-naive benchmark. Metrics are recorded per fold and horizon, then aggregated across expanding-window validation folds.

Primary metrics:

- **MAE** — absolute error in trips;
- **RMSE** — emphasizes large misses;
- **WAPE** — volume-normalized operational error;
- **sMAPE** — scale-normalized symmetric percentage error.

## Uncertainty and monitoring

Point forecasts can be wrapped with split-conformal intervals calibrated from held-out residuals. Coverage is a property to verify on future samples, not merely a configuration value.

Production monitoring currently includes Population Stability Index (PSI) utilities for numerical feature drift. The model card specifies additional monitoring requirements for missingness, prediction drift, realized error, API reliability, and feature-contract compatibility.

## Current roadmap

Completed foundation:

- [x] ingestion and quality contracts
- [x] complete hourly demand mart
- [x] leakage-safe lag/rolling features
- [x] seasonal baseline
- [x] temporal CV
- [x] LightGBM model layer
- [x] direct 1h/6h/24h targets
- [x] Optuna tuning
- [x] conformal intervals
- [x] MLflow logging
- [x] model persistence
- [x] FastAPI service
- [x] Docker packaging
- [x] PSI drift monitoring
- [x] CI and automated tests

Next research layer:

- [ ] multi-month EDA and demand-regime analysis
- [ ] weather and holiday/event covariates
- [ ] taxi-zone adjacency/geospatial features
- [ ] SHAP diagnostics generated from trained production candidates
- [ ] benchmark CatBoost/XGBoost alternatives
- [ ] automated model comparison/model-card generation
- [ ] production-grade feature retrieval rather than client-supplied feature vectors

## License

MIT. NYC TLC source data remains subject to the terms and notices of its publisher.
