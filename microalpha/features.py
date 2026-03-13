from __future__ import annotations

import numpy as np

from microalpha import _cpp


FEATURE_NAMES = [
    "ofi_best",
    "queue_imbalance_best",
    "depth_imbalance",
    "spread",
    "microprice_deviation",
]


def compute_features(
    bid_prices: np.ndarray,
    bid_sizes: np.ndarray,
    ask_prices: np.ndarray,
    ask_sizes: np.ndarray,
) -> np.ndarray:
    """
    Compute event-level microstructure features via C++ backend.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (N, 5).
    """
    X = _cpp.compute_features_series(
        np.asarray(bid_prices, dtype=np.float64, order="C"),
        np.asarray(bid_sizes, dtype=np.float64, order="C"),
        np.asarray(ask_prices, dtype=np.float64, order="C"),
        np.asarray(ask_sizes, dtype=np.float64, order="C"),
    )
    return np.asarray(X, dtype=np.float64)