from __future__ import annotations

from collections.abc import Sequence

import numpy as np


ArrayLike = Sequence[float] | np.ndarray


def population_stability_index(
    reference: ArrayLike,
    current: ArrayLike,
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Estimate univariate distribution drift using quantile-binned PSI."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    expected = np.asarray(reference, dtype=float)
    observed = np.asarray(current, dtype=float)
    if expected.size == 0 or observed.size == 0:
        raise ValueError("reference and current samples must be non-empty")
    if not (np.isfinite(expected).all() and np.isfinite(observed).all()):
        raise ValueError("drift samples must contain only finite values")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if edges.size < 2:
        return 0.0 if np.allclose(expected, observed) else float("inf")

    edges[0] = -np.inf
    edges[-1] = np.inf
    expected_counts, _ = np.histogram(expected, bins=edges)
    observed_counts, _ = np.histogram(observed, bins=edges)

    expected_share = expected_counts / expected_counts.sum()
    observed_share = observed_counts / observed_counts.sum()
    expected_share = np.clip(expected_share, epsilon, None)
    observed_share = np.clip(observed_share, epsilon, None)

    return float(
        np.sum((observed_share - expected_share) * np.log(observed_share / expected_share))
    )


def classify_psi(psi: float) -> str:
    """Map PSI to conventional operational severity bands."""
    if psi < 0:
        raise ValueError("PSI cannot be negative")
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "watch"
    return "drift"
