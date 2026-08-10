from __future__ import annotations

from collections.abc import Mapping, Sequence


def _format_metric(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_model_card(
    *,
    horizon_hours: int,
    benchmark: Mapping[str, object],
    feature_names: Sequence[str],
    top_features: Sequence[Mapping[str, object]] = (),
    model_name: str = "LightGBM Poisson demand forecaster",
) -> str:
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")
    if not feature_names:
        raise ValueError("feature_names must not be empty")

    lines = [
        "# Model Card — NYC Urban Demand Intelligence",
        "",
        "## Model overview",
        "",
        f"- Model: **{model_name}**",
        f"- Forecast horizon: **{horizon_hours} hours**",
        "- Target: hourly Yellow Taxi pickup demand per NYC taxi zone",
        "- Validation: expanding-window temporal backtesting",
        "- Primary baseline: weekly seasonal naive",
        "",
        "## Benchmark",
        "",
        "| Metric | Model | Baseline | Relative improvement |",
        "|---|---:|---:|---:|",
    ]

    for metric in ("mae", "wape"):
        model_value = benchmark.get(f"model_{metric}", "n/a")
        baseline_value = benchmark.get(f"baseline_{metric}", "n/a")
        improvement = benchmark.get(f"{metric}_improvement", "n/a")
        if isinstance(improvement, float):
            improvement_display = f"{improvement:.2%}"
        else:
            improvement_display = str(improvement)
        lines.append(
            f"| {metric.upper()} | {_format_metric(model_value)} | "
            f"{_format_metric(baseline_value)} | {improvement_display} |"
        )

    lines.extend(
        [
            "",
            "## Feature contract",
            "",
            f"The serving model expects **{len(feature_names)} features** in a fixed order.",
            "Feature engineering is restricted to information available at forecast origin time.",
            "",
            "## Explainability",
            "",
        ]
    )

    if top_features:
        lines.extend(
            [
                "| Feature | Mean |SHAP| | Importance share |",
                "|---|---:|---:|",
            ]
        )
        for item in top_features:
            feature = str(item.get("feature", "unknown"))
            magnitude = _format_metric(item.get("mean_abs_shap", "n/a"))
            share = item.get("importance_share", "n/a")
            share_display = f"{share:.2%}" if isinstance(share, float) else str(share)
            lines.append(f"| {feature} | {magnitude} | {share_display} |")
    else:
        lines.append("SHAP importance has not been attached to this model-card build.")

    lines.extend(
        [
            "",
            "## Intended use",
            "",
            "Use forecasts for fleet positioning, supply planning, operational monitoring, "
            "and demand-pattern analysis. The model is not intended for safety-critical or "
            "individual-level decision making.",
            "",
            "## Limitations",
            "",
            "- Performance can degrade under regime shifts, major events, weather shocks, or data outages.",
            "- Taxi demand is a proxy for one mobility mode and does not represent total city travel demand.",
            "- Prediction quality should be monitored by zone, demand band, and forecast horizon.",
            "",
            "## Reproducibility",
            "",
            "The repository records data-quality rules, feature definitions, temporal validation, "
            "benchmark methodology, model configuration, and CI-tested inference contracts.",
            "",
        ]
    )
    return "\n".join(lines)
