# Real-data research benchmark

The `Research benchmark` GitHub Actions workflow turns an official TLC month range into reproducible evaluation artifacts without committing large source datasets to the repository.

## Default experiment

- source window: January–June 2024;
- forecast horizon: 6 hours;
- model: LightGBM using the versioned model configuration;
- baseline: weekly seasonal naive;
- validation: expanding-window temporal folds;
- explainability sample: at most 5,000 deterministic rows.

The six-month default is long enough to support a 90-day initial training window plus multiple validation folds under the current configuration.

## Workflow stages

```text
TLC monthly Parquet downloads
        -> monthly quality validation and aggregation
        -> merged demand mart + build manifest
        -> taxi-zone reference validation
        -> EDA JSON
        -> leakage-safe feature materialization
        -> paired baseline benchmark
        -> final horizon model
        -> SHAP importance
        -> model card
        -> downloadable GitHub Actions artifact
```

## Outputs

The workflow uploads a compact artifact containing:

- EDA report;
- paired model-vs-baseline benchmark report;
- training/backtest report;
- SHAP feature-importance report;
- generated model card;
- native LightGBM model bundle and feature metadata;
- dataset build manifest.

Raw TLC Parquet files, merged research data, and the full feature matrix are deliberately not uploaded as workflow artifacts because they are reproducible and potentially large.

## Interpretation

A model should not be promoted because one metric is numerically lower in one split. Review:

1. average improvement over weekly seasonal naive;
2. consistency across temporal folds;
3. WAPE and MAE together;
4. zone and demand-band failure modes;
5. SHAP features for plausible signal usage;
6. conformal interval coverage on a held-out calibration/evaluation window;
7. drift and realized-error monitoring after deployment.

The resulting JSON/Markdown artifacts are intended to replace hand-written or invented portfolio metrics with results produced from official public data and versioned code.
