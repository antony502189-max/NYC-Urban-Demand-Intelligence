from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from nyc_demand.config import ensure_directory, load_yaml
from nyc_demand.data.aggregate import aggregate_file
from nyc_demand.data.download import download_month
from nyc_demand.data.range_pipeline import YearMonth, merge_hourly_demand, month_range


def _parse_year_month(value: str) -> YearMonth:
    try:
        year_text, month_text = value.split("-", maxsplit=1)
        return YearMonth(int(year_text), int(month_text))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM") from exc


def build_dataset_range(
    start: YearMonth,
    end: YearMonth,
    *,
    output_path: str | Path | None = None,
    overwrite_downloads: bool = False,
) -> dict[str, object]:
    config = load_yaml("configs/data.yaml")
    processed_dir = ensure_directory(config["paths"]["processed_dir"])
    months = month_range(start, end)

    demand_frames: list[pl.DataFrame] = []
    quality_reports: dict[str, object] = {}
    monthly_outputs: list[str] = []

    for item in months:
        raw_path = download_month(
            item.year,
            item.month,
            overwrite=overwrite_downloads,
        )
        monthly_output = processed_dir / f"demand_{item.label}.parquet"
        aggregate_file(raw_path, monthly_output)
        demand_frames.append(pl.read_parquet(monthly_output))
        monthly_outputs.append(str(monthly_output))

        quality_path = monthly_output.with_suffix(".quality.json")
        quality_reports[item.label] = json.loads(quality_path.read_text(encoding="utf-8"))

    merged = merge_hourly_demand(demand_frames)
    destination = (
        Path(output_path)
        if output_path is not None
        else processed_dir / f"demand_{start.label}_to_{end.label}.parquet"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(destination, compression="zstd", statistics=True)

    report = {
        "start_month": start.label,
        "end_month": end.label,
        "months": [item.label for item in months],
        "monthly_outputs": monthly_outputs,
        "merged_output": str(destination),
        "rows": merged.height,
        "zones": merged["zone_id"].n_unique(),
        "quality": quality_reports,
    }
    report_path = destination.with_suffix(".build.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a multi-month NYC taxi demand dataset")
    parser.add_argument("--start", type=_parse_year_month, required=True, help="YYYY-MM")
    parser.add_argument("--end", type=_parse_year_month, required=True, help="YYYY-MM")
    parser.add_argument("--output")
    parser.add_argument("--overwrite-downloads", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_dataset_range(
        args.start,
        args.end,
        output_path=args.output,
        overwrite_downloads=args.overwrite_downloads,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
