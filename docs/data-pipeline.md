# Data pipeline

The project treats NYC TLC trip records as immutable source inputs and builds reproducible hourly pickup-demand marts from them.

## Single month

```bash
python -m nyc_demand.data.download --year 2024 --month 1
python -m nyc_demand.data.aggregate \
  --input data/raw/yellow_tripdata_2024-01.parquet \
  --output data/processed/demand_2024-01.parquet
```

Each aggregation writes a sibling `*.quality.json` audit report containing input, retained, and rejected row counts by validation rule.

## Multi-month build

```bash
python scripts/build_dataset.py --start 2024-01 --end 2024-06
```

The range builder downloads missing monthly Parquet files, aggregates every month independently, merges duplicate `(timestamp, zone_id)` keys deterministically, writes one compressed Parquet mart, and produces a `*.build.json` manifest containing the source months and quality reports.

## Reproducibility contract

- raw source files are never committed to Git;
- download URLs are derived from versioned configuration;
- invalid zones, durations, distances, and missing timestamps are filtered by explicit rules;
- demand is aggregated hourly by pickup taxi zone;
- merged datasets are sorted by `(timestamp, zone_id)`;
- generated build manifests record the exact month range and monthly outputs used.
