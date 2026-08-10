import math

import pytest

from nyc_demand.models.metrics import evaluate, mae, rmse, smape, wape


def test_forecasting_metrics_have_expected_values() -> None:
    actual = [10.0, 20.0, 30.0]
    predicted = [12.0, 18.0, 33.0]

    assert mae(actual, predicted) == pytest.approx(7 / 3)
    assert rmse(actual, predicted) == pytest.approx(math.sqrt(17 / 3))
    assert wape(actual, predicted) == pytest.approx(7 / 60)
    assert 0.0 < smape(actual, predicted) < 1.0


def test_zero_demand_wape_is_well_defined() -> None:
    assert wape([0, 0], [0, 0]) == 0.0
    assert math.isinf(wape([0, 0], [1, 0]))


def test_evaluate_returns_metric_contract() -> None:
    result = evaluate([1, 2], [1, 3])
    assert set(result) == {"mae", "rmse", "wape", "smape"}


def test_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        mae([1, 2], [1])
