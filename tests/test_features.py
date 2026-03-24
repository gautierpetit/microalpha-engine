import numpy as np

from microalpha.config import FeatureConfig
from microalpha.features import compute_features, make_feature_names


def test_make_feature_names_matches_feature_config() -> None:
    cfg = FeatureConfig(
        ofi_window_raw=50,
        ofi_norm_windows=[10, 50, 100],
        vol_window=50,
        intensity_window="1s",
    )

    feature_names = make_feature_names(cfg)

    assert feature_names == [
        "ofi_best",
        "ofi_best_norm",
        "queue_imbalance_best",
        "depth_imbalance_3",
        "depth_imbalance_5",
        "depth_imbalance_10",
        "spread",
        "microprice_deviation",
        "ofi_roll_sum_50",
        "ofi_best_norm_roll_sum_10",
        "ofi_best_norm_roll_sum_50",
        "ofi_best_norm_roll_sum_100",
        "midprice_vol_50",
        "event_intensity_1s",
    ]


def test_compute_features_shape_matches_make_feature_names(monkeypatch) -> None:
    n = 8
    cfg = FeatureConfig(
        ofi_window_raw=50,
        ofi_norm_windows=[10, 50, 100],
        vol_window=50,
        intensity_window="1s",
    )

    def fake_compute_features_series(**kwargs):
        # Return 8 core columns exactly as the C++ extension promises
        return np.zeros((n, 8), dtype=np.float64)

    monkeypatch.setattr(
        "microalpha.features.compute_features_series",
        fake_compute_features_series,
    )

    bid_prices = np.ones((n, 10))
    bid_sizes = np.ones((n, 10))
    ask_prices = np.ones((n, 10)) * 2
    ask_sizes = np.ones((n, 10))
    midprice = np.linspace(100.0, 101.0, n)
    timestamps = np.arange(n, dtype=float)

    X = compute_features(
        bid_prices=bid_prices,
        bid_sizes=bid_sizes,
        ask_prices=ask_prices,
        ask_sizes=ask_sizes,
        midprice=midprice,
        timestamps=timestamps,
        cfg=cfg,
    )

    feature_names = make_feature_names(cfg)

    assert X.shape == (n, len(feature_names))