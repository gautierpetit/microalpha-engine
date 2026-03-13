from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from microalpha.io import LobsterPaths, LobsterConfig, load_lobster, compute_tie_rate, time_stats
from microalpha.labels import create_directional_labels, summarize_labels, align_features_with_labels
from microalpha.features import compute_features, FEATURE_NAMES


def main() -> None:
    msg_path = Path(r"data\raw\LOBSTER_SampleFile_AAPL_2012-06-21_10\AAPL_2012-06-21_34200000_57600000_message_10.csv")
    ob_path = Path(r"data\raw\LOBSTER_SampleFile_AAPL_2012-06-21_10\AAPL_2012-06-21_34200000_57600000_orderbook_10.csv")

    paths = LobsterPaths(message_csv=msg_path, orderbook_csv=ob_path)
    cfg = LobsterConfig(levels=10, price_scale=10_000)

    data = load_lobster(paths, cfg=cfg, validate=True)

    print("\n=== LOBSTER LOAD OK ===")
    print(f"Events: {len(data.t):,}")
    print(f"Levels: {data.bid_prices.shape[1]}")
    print(f"Midprice: min={data.midprice.min():.4f}, max={data.midprice.max():.4f}")
    print(f"Spread:   min={data.spread.min():.6f}, p50={np.median(data.spread):.6f}, p95={np.percentile(data.spread,95):.6f}")

    plt.plot(data.t/3600, data.midprice)
    plt.xlabel("Time (hours)")
    plt.ylabel("Midprice")
    plt.title("Midprice Over Time")
    plt.show()

    ts = time_stats(data.t)
    print("\n=== TIME STATS ===")
    for k, v in ts.items():
        if "dt_" in k or "duration" in k or "t_" in k:
            print(f"{k:>12}: {v:.6f}")
        else:
            print(f"{k:>12}: {int(v):,}")

    for H in (500, 1000):
        p_tie, n_eff = compute_tie_rate(data.midprice, H)
        print(f"\n=== TIE RATE (H={H}) ===")
        print(f"n_effective: {n_eff:,}")
        print(f"p_tie:       {p_tie:.2%}")
        # Report implied drop if using binary-drop-ties
        print(f"binary-drop-ties would drop ~{p_tie:.2%} of samples")

    label_result = create_directional_labels(data.midprice, horizon=500, label_mode="binary_drop_ties")

    plt.hist(label_result.delta, bins=200)
    plt.xlabel("Forward Midprice Change")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Forward Midprice Changes H={label_result.horizon}")
    plt.show()

    print("\n=== LABEL SUMMARY (H=500, binary_drop_ties) ===")
    summary = summarize_labels(label_result)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:>15}: {v:.6f}")
        else:
            print(f"{k:>15}: {v}")
    
    #######
    X_raw = compute_features(
        data.bid_prices,
        data.bid_sizes,
        data.ask_prices,
        data.ask_sizes,
    )

    print("\n=== FEATURE MATRIX ===")
    print(f"shape: {X_raw.shape}")
    for i, name in enumerate(FEATURE_NAMES):
        col = X_raw[:, i]
        print(
            f"{name:>24}: "
            f"min={col.min():.6f}, "
            f"p005={np.percentile(col, 5):.6f}, "
            f"p50={np.median(col):.6f}, "
            f"p95={np.percentile(col, 95):.6f}, "
            f"max={col.max():.6f}"
        )

    X, y = align_features_with_labels(X_raw, label_result)

    print("\nDone.\n")


if __name__ == "__main__":
    main()