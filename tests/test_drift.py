import numpy as np

from nyc_demand.monitoring.drift import classify_psi, population_stability_index


def test_psi_is_small_for_similar_distributions() -> None:
    reference = np.arange(1, 101, dtype=float)
    current = reference + 0.05

    psi = population_stability_index(reference, current)

    assert psi < 0.1
    assert classify_psi(psi) == "stable"


def test_psi_detects_large_distribution_shift() -> None:
    reference = np.arange(1, 101, dtype=float)
    current = np.arange(201, 301, dtype=float)

    psi = population_stability_index(reference, current)

    assert psi >= 0.25
    assert classify_psi(psi) == "drift"
