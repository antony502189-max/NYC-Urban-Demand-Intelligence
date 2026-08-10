from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from nyc_demand.analysis.eda import demand_overview, hourly_profile, top_zones, weekday_profile
from nyc_demand.data.zones import load_zone_lookup


def generate_report(
    demand_path: str | Path,
    output_path: str | Path,
    *,
    zone_lookup_path: str | Path | None = None,
    top_n: int = 20,
) -> Path:
    demand = pl.read_parquet(demand_path)
    zones = load_zone_lookup(zone_lookup_path) if zone_lookup_path is not None else None

    payload = {
        "overview": demand_overview(demand),
        "hourly_profile": hourly_profile(demand).to_dicts(),
        "weekday_profile": weekday_profile(demand).to_dicts(),
        "top_zones": top_zones(demand, n=top_n, zone_lookup=zones).to_dicts(),
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic NYC demand EDA summaries")
    parser.add_argument("--input", required=True, help="Hourly demand Parquet file")
    parser.add_argument("--output", required=True, help="Destination JSON report")
    parser.add_argument("--zone-lookup", help="Optional taxi_zone_lookup.csv path")
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_report(
        args.input,
        args.output,
        zone_lookup_path=args.zone_lookup,
        top_n=args.top_n,
    )
    print(result)


if __name__ == "__main__":
    main()
