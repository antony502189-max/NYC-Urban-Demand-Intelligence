from __future__ import annotations

import argparse
import json
from pathlib import Path

from nyc_demand.reporting.model_card import render_model_card


def render_from_artifacts(
    model_dir: str | Path,
    benchmark_path: str | Path,
    output_path: str | Path,
    *,
    shap_path: str | Path | None = None,
) -> Path:
    model_root = Path(model_dir)
    metadata = json.loads((model_root / "metadata.json").read_text(encoding="utf-8"))
    benchmark_payload = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    shap_payload = (
        json.loads(Path(shap_path).read_text(encoding="utf-8")) if shap_path is not None else {}
    )

    markdown = render_model_card(
        horizon_hours=int(metadata["horizon_hours"]),
        benchmark=benchmark_payload["summary"],
        feature_names=tuple(metadata["feature_names"]),
        top_features=tuple(shap_payload.get("features", [])),
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markdown, encoding="utf-8")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a model card from benchmark artifacts")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--shap")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        render_from_artifacts(
            args.model_dir,
            args.benchmark,
            args.output,
            shap_path=args.shap,
        )
    )


if __name__ == "__main__":
    main()
