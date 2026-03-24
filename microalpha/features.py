from __future__ import annotations

import numpy as np
import pandas as pd

from microalpha._cpp import compute_features_series
from microalpha.config import FeatureConfig


def compute_features(
    *,
    bid_prices: np.ndarray,
    bid_sizes: np.ndarray,
    ask_prices: np.ndarray,
    ask_sizes: np.ndarray,
    midprice: np.ndarray,
    timestamps: np.ndarray,
    cfg: FeatureConfig,
) -> np.ndarray:
    core = compute_features_series(
        bid_prices=bid_prices,
        bid_sizes=bid_sizes,
        ask_prices=ask_prices,
        ask_sizes=ask_sizes,
    )

    core_df = pd.DataFrame(
        core,
        columns=[
            "ofi_best",
            "ofi_best_norm",
            "queue_imbalance_best",
            "depth_imbalance_3",
            "depth_imbalance_5",
            "depth_imbalance_10",
            "spread",
            "microprice_deviation",
        ],
    )

    core_df[f"ofi_roll_sum_{cfg.ofi_window_raw}"] = (
        core_df["ofi_best"].rolling(cfg.ofi_window_raw, min_periods=1).sum()
    )

    for window in cfg.ofi_norm_windows:
        core_df[f"ofi_best_norm_roll_sum_{window}"] = (
            core_df["ofi_best_norm"].rolling(window, min_periods=1).sum()
        )

    midprice_series = pd.Series(midprice)
    mid_returns = midprice_series.diff().fillna(0.0)
    core_df[f"midprice_vol_{cfg.vol_window}"] = (
        mid_returns.rolling(cfg.vol_window, min_periods=1).std().fillna(0.0)
    )

    timestamps_dt = pd.to_datetime(timestamps, unit="s")
    event_count = pd.Series(1.0, index=timestamps_dt)
    intensity = event_count.rolling(cfg.intensity_window).sum()
    core_df[f"event_intensity_{cfg.intensity_window}"] = intensity.to_numpy(
        dtype=np.float64
    )

    return core_df.to_numpy(dtype=np.float64)


def make_feature_names(cfg: FeatureConfig) -> list[str]:
    names = [
        "ofi_best",
        "ofi_best_norm",
        "queue_imbalance_best",
        "depth_imbalance_3",
        "depth_imbalance_5",
        "depth_imbalance_10",
        "spread",
        "microprice_deviation",
        f"ofi_roll_sum_{cfg.ofi_window_raw}",
    ]

    names.extend([f"ofi_best_norm_roll_sum_{w}" for w in cfg.ofi_norm_windows])
    names.extend(
        [f"midprice_vol_{cfg.vol_window}", f"event_intensity_{cfg.intensity_window}"]
    )

    return names
