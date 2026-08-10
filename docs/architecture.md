# Architecture and design decisions

## Objective

The system forecasts hourly Yellow Taxi pickup demand for each NYC taxi zone at 1h, 6h, and 24h horizons. The design prioritizes leakage safety, reproducibility, operational diagnostics, and a clear path from offline research to online inference.

## Data boundary

Raw NYC TLC Parquet files are treated as immutable source inputs. They are not committed to Git. Every processed dataset is reproducible from a configured month range and explicit validation rules.

Trip rows are rejected when required timestamps are missing, pickup zones are outside the configured domain, durations are invalid, or trip distances are outside allowed bounds. Each monthly aggregation emits a quality report so filtering decisions remain auditable.

## Demand mart

Valid trips are aggregated to `(timestamp, zone_id) -> demand`. Missing combinations inside the observed time range are represented as explicit zero-demand rows. This distinction is important: an absent database row and an observed hour with zero pickups are not equivalent states.

Multi-month builds aggregate each source month independently and then merge timestamp-zone keys deterministically. A build manifest records the input range and monthly quality outputs.

## Feature boundary

Forecast features are defined at forecast origin time. Calendar fields are known deterministically. Demand lag and rolling features are shifted so they only use historical observations for the same taxi zone.

Taxi-zone borough and service-zone categories are one-hot encoded rather than integer-label encoded. This avoids introducing a false ordinal relationship between categories.

## Target construction

Direct models are trained separately for 1h, 6h, and 24h horizons. A row at origin time `o` receives target demand from `o + horizon`.

Direct modelling was chosen because error characteristics and useful feature relationships can differ materially by horizon. It also makes horizon-specific validation and model replacement straightforward.

## Leakage prevention

Chronological splitting alone is not sufficient for direct forecasting. A training origin can occur before the validation boundary while its future label occurs inside validation.

The backtester therefore enforces both conditions:

1. features contain no information after forecast origin;
2. a training row is removed when its `forecast_timestamp` crosses the validation start.

Random train/test splitting is prohibited for model evaluation.

## Baseline contract

The primary benchmark is weekly seasonal naive. For a target at `t`, the baseline uses demand at `t - 168h`. For direct horizon `h`, the origin-time history lag is therefore `168 - h` hours.

Model and baseline scores are computed on paired validation rows. This prevents apparent improvements caused by evaluating candidates on easier or different samples.

## Model layer

LightGBM with a Poisson objective is the primary production candidate because the target is a non-negative count, the feature set is heterogeneous, and tree boosting handles non-linear lag/calendar interactions efficiently.

Optuna is used for hyperparameter search over temporal backtests. The repository keeps challenger models as future work rather than adding algorithms without demonstrating value.

## Uncertainty

Point forecasts can be wrapped with split-conformal intervals calibrated on held-out absolute residuals. The interval layer is separate from the point estimator so uncertainty calibration can be replaced without retraining the forecasting model.

## Explainability

Global SHAP summaries expose average feature contribution magnitude. Local SHAP explanations decompose individual tree-model outputs into a base value and per-feature contributions. Explainability artifacts are generated from stored model bundles rather than from ad hoc notebook state.

## Model persistence

A saved model bundle contains the native LightGBM model plus metadata including forecast horizon and ordered feature names. The feature contract is reused by the inference service, reducing the risk of silent training-serving column-order mismatch.

## Serving

FastAPI exposes health and forecast endpoints. The forecast endpoint validates the exact stored feature contract before prediction. Docker packages the same service for reproducible execution.

The current API intentionally accepts engineered numeric features. Production-grade automated feature retrieval is a future extension and is kept separate from the model-serving contract.

## Monitoring

Monitoring has two independent dimensions:

- feature-distribution drift via Population Stability Index;
- realized forecast quality via MAE, RMSE, WAPE, sMAPE, bias, and p90 absolute error.

Reference and current scoring windows are compared using severity bands. An overall status escalates when either meaningful feature drift or forecast-performance regression is detected.

## Quality gates

Every feature branch is expected to pass the same CI sequence before merge:

```text
package installation
    -> Python compilation
    -> Ruff static checks
    -> pytest suite
```

This keeps research utilities, production code, tests, and CLI workflows under the same integration contract.
