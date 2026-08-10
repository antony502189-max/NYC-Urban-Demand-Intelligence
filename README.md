# NYC Urban Demand Intelligence

[![CI](https://github.com/antony502189-max/NYC-Urban-Demand-Intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/antony502189-max/NYC-Urban-Demand-Intelligence/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Production-oriented data science and ML engineering system for forecasting hourly NYC Yellow Taxi pickup demand by taxi zone at **1h, 6h, and 24h** horizons.

This is intentionally not a notebook-only project. The repository covers reproducible data ingestion, explicit quality contracts, multi-month dataset builds, temporal/geographic EDA, leakage-safe feature engineering, baseline benchmarking, gradient-boosted forecasting, uncertainty calibration, experiment tracking, SHAP explainability, model persistence, FastAPI serving, Docker packaging, and production monitoring.

## Business problem

Given historical NYC TLC Yellow Taxi trips, estimate future pickup volume for every taxi zone so that an operator can make better decisions about:

- fleet and driver positioning;
- supply planning for peak demand;
- service-level monitoring;
- zone-level demand anomalies;
- forecast-risk and uncertainty management.

The core target is:

```text
y[z, t] = number of valid Yellow Taxi pickups in zone z during hour t
```

## System architecture

```text
Official NYC TLC Parquet + taxi-zone lookup
                 |
                 v
      atomic download / range builder
                 |
                 v
      schema + domain validation
                 |
                 v
      hourly timestamp x zone mart
                 |
        +--------+---------+
        |                  |
        v                  v
  temporal EDA       zone metadata
        |                  |
        +--------+---------+
                 |
                 v
       historical-only features
                 |
    +------------+-------------+
    |                          |
    v                          v
weekly seasonal naive      LightGBM
    |                     1h / 6h / 24h
    +------------+-------------+
                 |
                 v
      paired expanding-window CV
                 |
     +-----------+------------+-------------+
     |                        |             |
     v                        v             v
 Optuna tuning        conformal bands   SHAP diagnostics
     |                        |             |
     +-----------+------------+-------------+
                 |
                 v
          versioned model bundle
                 |
        +--------+---------+
        |                  |
        v                  v
 FastAPI / Docker     monitoring reports
                    PSI + WAPE + bias
```

## Implemented capabilities

### Data engineering

- atomic downloader for official monthly NYC TLC Yellow Taxi Parquet files;
- inclusive multi-month range builds with deterministic manifests;
- explicit schema and domain quality checks;
- persisted quality reports for every monthly aggregation;
- hourly pickup-demand aggregation by taxi zone;
- complete zero-filled hourly zone grids;
- deterministic merge of multi-month demand marts;
- validated taxi-zone reference lookup with borough, zone name, and service-zone metadata;
- Parquet-first processing with Polars/PyArrow.

### EDA and geographic context

- dataset-level coverage and zero-demand summaries;
- hourly demand profiles;
- weekday demand profiles;
- top-zone ranking with human-readable zone labels;
- deterministic JSON EDA reports;
- non-ordinal borough/service-zone one-hot features;
- safe `Unknown` handling for missing zone metadata.

### Leakage-safe feature engineering

- calendar features;
- per-zone lag features at multiple horizons;
- shifted rolling means and rolling standard deviations;
- direct 1h/6h/24h target construction;
- explicit target/forecast timestamp exclusion from feature inference;
- validation-label boundary checks so future labels cannot cross into training;
- no random train/test splitting for forecasting evaluation.

### Modelling and evaluation

- direct weekly seasonal-naive benchmark for each forecast horizon;
- LightGBM Poisson demand regression;
- expanding-window temporal cross-validation;
- paired model-vs-baseline folds using identical validation rows;
- MAE, RMSE, WAPE, and sMAPE;
- fold-level relative improvement over baseline;
- residual bias and tail-error diagnostics;
- error tables by taxi zone and demand band;
- Optuna hyperparameter optimization over temporal backtests;
- split-conformal prediction intervals;
- portable LightGBM model bundles with ordered feature metadata.

### Explainability and reporting

- deterministic global SHAP feature importance;
- local SHAP contribution explanations;
- stored-model SHAP report CLI;
- benchmark-driven model-card generation;
- model-card sections for intended use, limitations, reproducibility, and explainability.

### Production and MLOps

- MLflow backtest logging;
- FastAPI `/health` and `/v1/forecast` endpoints;
- strict online feature-contract validation;
- Dockerized inference service;
- PSI-based numerical feature drift monitoring;
- realized forecast-performance snapshots;
- daily WAPE, MAE, and bias monitoring;
- performance-regression severity bands;
- combined drift + performance monitoring reports;
- GitHub Actions quality gates: install, compile, Ruff, pytest.

## Repository layout

```text
configs/                       Data and model contracts
docs/                          Pipeline and CI documentation
scripts/                       Runnable data/ML/monitoring workflows
src/nyc_demand/
  analysis/                    Deterministic EDA summaries
  api/                         FastAPI inference service
  data/                        Download, validation, aggregation, range builds
  features/                    Temporal and zone feature engineering
  models/                      Baselines, training, CV, tuning, diagnostics, SHAP
  monitoring/                  PSI and realized-performance monitoring
  reporting/                   Model-card generation
  tracking/                    MLflow integration
reports/                       Generated model/evaluation documentation
tests/                         Unit and integration tests
.github/workflows/             CI and maintenance workflows
Dockerfile                     Inference container
```

## Installation

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[ml,api,dev]"
```

Install MLflow support when needed:

```bash
pip install -e ".[tracking]"
```

## Build the dataset

### One month

```bash
python -m nyc_demand.data.download --year 2024 --month 1
python -m nyc_demand.data.aggregate \
  --input data/raw/yellow_tripdata_2024-01.parquet \
  --output data/processed/demand_2024-01.parquet
```

### Multi-month research dataset

```bash
python scripts/build_dataset.py --start 2024-01 --end 2024-06
```

The range builder creates a merged Parquet mart plus a `*.build.json` manifest containing the month range, monthly outputs, and per-month quality reports.

## Generate EDA

```bash
python scripts/generate_eda_report.py \
  --input data/processed/demand_2024-01_to_2024-06.parquet \
  --output reports/eda/2024-h1.json \
  --zone-lookup data/reference/taxi_zone_lookup.csv
```

## Benchmark against the seasonal baseline

```bash
python scripts/benchmark_model.py \
  --input data/processed/demand_2024-01_to_2024-06.parquet \
  --horizon 6 \
  --output reports/metrics/h6-benchmark.json
```

The benchmark evaluates LightGBM and weekly seasonal naive on identical temporal validation rows. A production candidate should demonstrate a meaningful improvement rather than merely reporting an isolated model metric.

## Train and persist a model

```bash
python scripts/train_model.py \
  --input data/processed/demand_2024-01_to_2024-06.parquet \
  --horizon 6 \
  --model-dir artifacts/models/h6 \
  --report reports/metrics/h6.json
```

With MLflow:

```bash
mlflow ui
python scripts/train_model.py \
  --input data/processed/demand_2024-01_to_2024-06.parquet \
  --horizon 6 \
  --model-dir artifacts/models/h6 \
  --report reports/metrics/h6.json \
  --mlflow
```

## Explain a trained model

```bash
python scripts/explain_model.py \
  --model-dir artifacts/models/h6 \
  --features data/features/h6.parquet \
  --output reports/explainability/h6-shap.json
```

Then render a model card from real run artifacts:

```bash
python scripts/render_model_card.py \
  --model-dir artifacts/models/h6 \
  --benchmark reports/metrics/h6-benchmark.json \
  --shap reports/explainability/h6-shap.json \
  --output reports/model-card-h6.md
```

## Serve a trained model

```bash
export NYC_DEMAND_MODEL_DIR=artifacts/models/h6
uvicorn nyc_demand.api.app:app --host 0.0.0.0 --port 8000
```

PowerShell:

```powershell
$env:NYC_DEMAND_MODEL_DIR="artifacts/models/h6"
uvicorn nyc_demand.api.app:app --host 0.0.0.0 --port 8000
```

The inference endpoint accepts the exact numeric feature contract stored with the model bundle. This intentionally prevents silent feature-order mismatches between training and serving.

## Docker

```bash
docker build -t nyc-demand-api .
docker run --rm -p 8000:8000 \
  -e NYC_DEMAND_MODEL_DIR=/models/h6 \
  -v "$(pwd)/artifacts/models:/models:ro" \
  nyc-demand-api
```

## Production monitoring

Given scored reference and current windows containing actuals, predictions, and monitored numeric features:

```bash
python scripts/monitor_model.py \
  --reference data/monitoring/reference.parquet \
  --current data/monitoring/current.parquet \
  --features demand_lag_1h,demand_lag_24h,demand_roll_mean_24h \
  --output reports/monitoring/current.json
```

The report combines:

- PSI feature-distribution drift;
- realized WAPE/MAE/RMSE/sMAPE;
- forecast bias;
- p90 absolute error;
- reference-vs-current performance degradation;
- an overall `stable`, `watch`, or `critical` status.

## Leakage contract

For forecast origin `o` and future target timestamp `t`, every feature must be computable using information available at or before `o`.

Two safeguards are enforced:

1. lag/rolling features are shifted so the current target observation cannot enter its own predictors;
2. training rows are removed when their future label timestamp crosses the validation boundary.

This second condition matters for direct multi-horizon models and is easy to miss even with otherwise chronological splitting.

## Evaluation contract

A candidate model is evaluated against the weekly seasonal baseline using expanding-window validation and identical validation rows.

Primary metrics:

- **MAE** — error measured directly in trips;
- **RMSE** — increases the penalty for large misses;
- **WAPE** — operational error normalized by observed volume;
- **sMAPE** — symmetric percentage-style scale normalization.

Model selection should consider aggregate metrics, zone-level failure modes, demand-band errors, uncertainty coverage, and stability across folds.

## Run quality gates locally

```bash
python -m compileall -q src tests scripts
ruff check src tests
pytest --cov=nyc_demand --cov-report=term-missing
```

The same sequence runs in GitHub Actions before feature branches are merged.

## Current status

Implemented:

- [x] official TLC ingestion and quality contracts
- [x] single- and multi-month demand marts
- [x] temporal and zone EDA
- [x] taxi-zone categorical context
- [x] leakage-safe temporal features
- [x] direct 1h/6h/24h targets
- [x] weekly seasonal baseline
- [x] paired temporal benchmarking
- [x] LightGBM model layer
- [x] Optuna tuning
- [x] conformal intervals
- [x] residual and segment diagnostics
- [x] MLflow logging
- [x] SHAP explainability
- [x] model-card generation
- [x] model persistence
- [x] FastAPI serving
- [x] Docker packaging
- [x] PSI drift monitoring
- [x] realized-performance monitoring
- [x] CI quality gates and automated tests

Next research extensions:

- [ ] weather covariates;
- [ ] public-holiday and major-event features;
- [ ] true taxi-zone adjacency/geometry features from TLC shapefiles;
- [ ] CatBoost/XGBoost challenger models;
- [ ] production feature retrieval instead of client-supplied feature vectors;
- [ ] published real-data benchmark artifacts for all three horizons.

## License

MIT. NYC TLC source data remains subject to the terms and notices of its publisher.
