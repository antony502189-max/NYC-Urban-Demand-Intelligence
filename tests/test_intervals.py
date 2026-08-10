import numpy as np
import pytest

from nyc_demand.models.intervals import fit_conformal_interval


def test_conformal_interval_produces_non_negative_lower_bound() -> None:
    interval = fit_conformal_interval(
        y_true=[10, 12, 14, 16, 18],
        y_pred=[9, 11, 16, 15, 20],
        coverage=0.8,
    )

    lower, upper = interval.bounds([1.0, 15.0])

    assert interval.radius >= 0
    assert np.all(lower >= 0)
    assert np.all(upper >= lower)


def test_conformal_interval_rejects_invalid_coverage() -> None:
    with pytest.raises(ValueError, match="coverage"):
        fit_conformal_interval([1.0], [1.0], coverage=1.0)
