from __future__ import annotations

import numpy as np

from microalpha._cpp import compute_features_series
from microalpha.config import FeatureConfig


def _parse_intensity_window_seconds(window: str) -> float:
    if not window.endswith("s"):
        raise ValueError(
            f"Only second-based intensity windows are supported, got {window!r}"
        )

    value = window[:-1]
    try:
        seconds = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid intensity window {window!r}") from exc

    if seconds <= 0:
        raise ValueError(f"Intensity window must be positive, got {window!r}")

    return seconds


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
    if len(cfg.ofi_norm_windows) != 3:
        raise ValueError(
            "Expected exactly 3 ofi_norm_windows for the current C++ feature contract"
        )

    return compute_features_series(
        bid_prices=bid_prices,
        bid_sizes=bid_sizes,
        ask_prices=ask_prices,
        ask_sizes=ask_sizes,
        midprice=midprice,
        timestamps=timestamps,
        ofi_window_raw=cfg.ofi_window_raw,
        ofi_norm_window_1=cfg.ofi_norm_windows[0],
        ofi_norm_window_2=cfg.ofi_norm_windows[1],
        ofi_norm_window_3=cfg.ofi_norm_windows[2],
        vol_window=cfg.vol_window,
        intensity_window_seconds=_parse_intensity_window_seconds(cfg.intensity_window),
    )


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