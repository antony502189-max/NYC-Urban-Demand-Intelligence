from nyc_demand.reporting.model_card import render_model_card


def test_model_card_contains_benchmark_and_explainability_sections() -> None:
    markdown = render_model_card(
        horizon_hours=6,
        benchmark={
            "model_mae": 8.0,
            "baseline_mae": 10.0,
            "mae_improvement": 0.2,
            "model_wape": 0.12,
            "baseline_wape": 0.15,
            "wape_improvement": 0.2,
        },
        feature_names=("hour", "demand_lag_1h"),
        top_features=(
            {
                "feature": "demand_lag_1h",
                "mean_abs_shap": 4.2,
                "importance_share": 0.7,
            },
        ),
    )

    assert "Forecast horizon: **6 hours**" in markdown
    assert "| MAE | 8.0000 | 10.0000 | 20.00% |" in markdown
    assert "demand_lag_1h" in markdown
    assert "## Limitations" in markdown
