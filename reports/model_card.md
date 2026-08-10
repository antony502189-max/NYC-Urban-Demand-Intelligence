# Model Card — NYC Taxi Zone Demand Forecasting

## Intended use

Forecast hourly Yellow Taxi pickup demand by NYC taxi zone for operational planning. Primary horizons are 1, 6, and 24 hours ahead.

The model is intended for decision support such as fleet positioning, staffing, and demand monitoring. It is not intended for safety-critical or individual-level decisions.

## Data

Training data is derived from NYC TLC Yellow Taxi trip records. Raw trip rows are validated, filtered using explicit domain rules, aggregated to a complete hourly zone grid, and converted into strictly historical lag, rolling, calendar, and zone features.

## Evaluation protocol

- Chronological expanding-window validation only
- No random train/test splitting
- Training labels must occur before each validation boundary
- Metrics: MAE, RMSE, WAPE, sMAPE
- Results are reported by forecast horizon and validation fold
- Seasonal-naive weekly demand is retained as a mandatory benchmark

## Uncertainty

Point forecasts can be augmented with split-conformal intervals calibrated on held-out residuals. Coverage must be measured on future data rather than assumed from calibration data.

## Monitoring

Production monitoring should include:

- input schema and missingness checks;
- feature-distribution drift using PSI;
- prediction-distribution drift;
- realized MAE/WAPE once labels arrive;
- API latency/error rate;
- model version and feature-contract compatibility.

## Known limitations

- TLC trip records describe completed taxi trips and do not directly measure unmet passenger demand.
- Sudden events, severe weather, road closures, strikes, policy changes, and structural mobility shifts can degrade forecasts.
- Direct multi-horizon models require separate validation and calibration for each horizon.
- Zone identifiers are treated as model inputs; future versions should evaluate richer geospatial encodings and adjacency features.

## Reproducibility checklist

- Source month/version recorded
- Data-quality report persisted
- Feature configuration versioned
- Training parameters logged
- Git commit recorded
- Cross-validation folds persisted
- Model artifact and feature order stored together
- Evaluation metrics and uncertainty calibration retained
