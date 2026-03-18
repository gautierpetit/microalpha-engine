from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from microalpha import _cpp


CORE_FEATURE_NAMES = [
    "ofi_best",
    "ofi_best_norm",
    "queue_imbalance_best",
    "depth_imbalance_3",
    "depth_imbalance_5",
    "depth_imbalance_10",
    "spread",
    "microprice_deviation",
]

TEMPORAL_FEATURE_NAMES = [
    "ofi_roll_sum_50",
    "ofi_best_norm_roll_sum_10",
    "ofi_best_norm_roll_sum_50",
    "ofi_best_norm_roll_sum_100",
    "event_intensity_1s",
    "midprice_vol_50",
]

FEATURE_NAMES = CORE_FEATURE_NAMES + TEMPORAL_FEATURE_NAMES


@dataclass(frozen=True)
class TemporalFeatureConfig:
    ofi_window: int = 50
    vol_window: int = 50
    intensity_window: str = "1s"


def compute_core_features(
    bid_prices: np.ndarray,
    bid_sizes: np.ndarray,
    ask_prices: np.ndarray,
    ask_sizes: np.ndarray,
) -> np.ndarray:
    """
    Compute event-level microstructure core features via the C++ backend.

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


def augment_temporal_features(
    X_core: np.ndarray,
    midprice: np.ndarray,
    timestamps: np.ndarray,
    cfg: TemporalFeatureConfig | None = None,
) -> np.ndarray:
    """
    Add temporal / rolling features on top of the core C++ feature matrix.

    Added features:
    - rolling OFI sum over last `ofi_window` events
    - event intensity over last `intensity_window` clock-time window
    - rolling midprice volatility over last `vol_window` events
      computed on midprice deltas, not levels

    Parameters
    ----------
    X_core : np.ndarray
        Core feature matrix of shape (N, 5).
    midprice : np.ndarray
        Midprice series of shape (N,).
    timestamps : np.ndarray
        Event timestamps in seconds from midnight, shape (N,).
    cfg : TemporalFeatureConfig | None
        Temporal feature configuration.

    Returns
    -------
    np.ndarray
        Augmented feature matrix of shape (N, 8).
    """
    cfg = cfg or TemporalFeatureConfig()

    _validate_temporal_inputs(X_core, midprice, timestamps)

    # Raw OFI rolling sum
    ofi_raw_series = pd.Series(X_core[:, 0], copy=False)
    ofi_roll_sum_50 = (
        ofi_raw_series.rolling(window=50, min_periods=50)
        .sum()
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
   
    # Normalized OFI multi-scale rolling sums
    ofi_norm_series = pd.Series(X_core[:, 1], copy=False)
    ofi_best_norm_roll_sum_10 = (
        ofi_norm_series.rolling(window=10, min_periods=10)
        .sum()
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    ofi_best_norm_roll_sum_50 = (
        ofi_norm_series.rolling(window=50, min_periods=50)
        .sum()
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    ofi_best_norm_roll_sum_100 = (
        ofi_norm_series.rolling(window=100, min_periods=100)
        .sum()
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )    

    # Event intensity over rolling clock-time window
    event_df = pd.DataFrame({"timestamp": timestamps})
    event_df.index = pd.to_datetime(event_df["timestamp"], unit="s")
    event_intensity = (
        event_df["timestamp"]
        .rolling(window=cfg.intensity_window)
        .count()
        .to_numpy(dtype=np.float64)
    )

    # Rolling midprice volatility over event window
    # Use midprice deltas, not levels
    mid_delta = np.diff(midprice, prepend=midprice[0])
    mid_delta_series = pd.Series(mid_delta, copy=False)
    midprice_vol = (
        mid_delta_series.rolling(window=cfg.vol_window, min_periods=cfg.vol_window)
        .std()
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    X_aug = np.column_stack(
        [
            X_core,
            ofi_roll_sum_50,
            ofi_best_norm_roll_sum_10,
            ofi_best_norm_roll_sum_50,
            ofi_best_norm_roll_sum_100,
            event_intensity,
            midprice_vol,
        ]
    )

    return X_aug


def compute_features(
    bid_prices: np.ndarray,
    bid_sizes: np.ndarray,
    ask_prices: np.ndarray,
    ask_sizes: np.ndarray,
    midprice: np.ndarray,
    timestamps: np.ndarray,
    cfg: TemporalFeatureConfig | None = None,
) -> np.ndarray:
    """
    Full feature pipeline:
    1. compute core event-level features in C++
    2. augment with temporal features in Python

    Returns
    -------
    np.ndarray
        Feature matrix of shape (N, 8).
    """
    X_core = compute_core_features(
        bid_prices=bid_prices,
        bid_sizes=bid_sizes,
        ask_prices=ask_prices,
        ask_sizes=ask_sizes,
    )

    X = augment_temporal_features(
        X_core=X_core,
        midprice=midprice,
        timestamps=timestamps,
        cfg=cfg,
    )
    return X


def _validate_temporal_inputs(
    X_core: np.ndarray,
    midprice: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    if X_core.ndim != 2:
        raise ValueError(f"X_core must be 2D, got shape {X_core.shape}")
    if midprice.ndim != 1:
        raise ValueError(f"midprice must be 1D, got shape {midprice.shape}")
    if timestamps.ndim != 1:
        raise ValueError(f"timestamps must be 1D, got shape {timestamps.shape}")

    n = X_core.shape[0]
    if midprice.shape[0] != n:
        raise ValueError(
            f"midprice length mismatch: expected {n}, got {midprice.shape[0]}"
        )
    if timestamps.shape[0] != n:
        raise ValueError(
            f"timestamps length mismatch: expected {n}, got {timestamps.shape[0]}"
        )