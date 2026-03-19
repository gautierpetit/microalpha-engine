from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from microalpha.features import FEATURE_NAMES, compute_features
from microalpha.io import LobsterConfig, LobsterPaths, load_lobster
from microalpha.labels import (
    align_features_with_labels,
    create_directional_labels,
)


FIG_DIR = Path("docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_forward_delta_histogram(delta: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(delta, bins=200)
    plt.title("Forward Midprice Delta Distribution")
    plt.xlabel("Forward midprice change")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_ofi_quantile_response(
    ofi_feature: np.ndarray,
    delta: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    n_bins: int = 20,
) -> None:
    df = pd.DataFrame(
        {
            "ofi": ofi_feature,
            "delta": delta,
            "y": y,
        }
    )

    # qcut can fail if too many duplicates; duplicates='drop' handles that
    df["ofi_bin"] = pd.qcut(df["ofi"], q=n_bins, duplicates="drop")

    grouped = df.groupby("ofi_bin", observed=True).agg(
        mean_ofi=("ofi", "mean"),
        mean_delta=("delta", "mean"),
        up_prob=("y", "mean"),
        count=("y", "size"),
    )

    fig, ax1 = plt.subplots(figsize=(11, 6))

    ax1.plot(grouped["mean_ofi"], grouped["mean_delta"], marker="o", label="Mean forward delta")
    ax1.set_xlabel("Mean OFI in quantile bin")
    ax1.set_ylabel("Mean forward midprice delta")
    ax1.set_title("OFI Quantile Response")

    ax2 = ax1.twinx()
    ax2.plot(grouped["mean_ofi"], grouped["up_prob"], marker="s", color="red", label="Up move probability")
    ax2.set_ylabel("Probability of up move")

    plt.tight_layout()
    plt.legend(loc="upper left")
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_class_conditional_boxplots(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    selected_features: list[str],
    out_path: Path,
) -> None:
    indices = [feature_names.index(name) for name in selected_features]

    n = len(indices)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 3.5 * n))
    if n == 1:
        axes = [axes]

    for ax, idx, name in zip(axes, indices, selected_features):
        down = X[y == 0, idx]
        up = X[y == 1, idx]

        ax.boxplot([down, up], tick_labels=["Down", "Up"], showfliers=False)
        ax.set_title(f"{name}: class-conditional distribution")
        ax.set_ylabel(name)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_correlation_heatmap(
    X: np.ndarray,
    feature_names: list[str],
    out_path: Path,
) -> None:
    corr = np.corrcoef(X, rowvar=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, aspect="auto")

    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticklabels(feature_names)
    ax.set_title("Feature Correlation Heatmap")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_time_slice(
    timestamps: np.ndarray,
    midprice: np.ndarray,
    ofi_roll_sum: np.ndarray,
    event_intensity: np.ndarray,
    out_path: Path,
    n_points: int = 5000,
) -> None:
    n = min(n_points, len(timestamps))
    t = timestamps[:n] - timestamps[0]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 9), sharex=True)

    axes[0].plot(t, midprice[:n])
    axes[0].set_ylabel("Midprice")
    axes[0].set_title("Time Slice Diagnostics")

    axes[1].plot(t, ofi_roll_sum[:n])
    axes[1].set_ylabel("OFI roll sum")

    axes[2].plot(t, event_intensity[:n])
    axes[2].set_ylabel("Event intensity")
    axes[2].set_xlabel("Seconds from start of slice")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def print_classwise_feature_means(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> None:
    print("\n=== CLASS-CONDITIONAL FEATURE MEANS ===")
    for i, name in enumerate(feature_names):
        mean_down = X[y == 0, i].mean()
        mean_up = X[y == 1, i].mean()
        diff = mean_up - mean_down
        print(
            f"{name:>24}: "
            f"mean_down={mean_down:.6f}, "
            f"mean_up={mean_up:.6f}, "
            f"diff(up-down)={diff:.6f}"
        )


def main() -> None:
    msg_path = Path(r"data\raw\LOBSTER_SampleFile_AAPL_2012-06-21_10\AAPL_2012-06-21_34200000_57600000_message_10.csv")
    ob_path = Path(r"data\raw\LOBSTER_SampleFile_AAPL_2012-06-21_10\AAPL_2012-06-21_34200000_57600000_orderbook_10.csv")

    paths = LobsterPaths(message_csv=msg_path, orderbook_csv=ob_path)
    cfg = LobsterConfig(levels=10, price_scale=10_000)

    data = load_lobster(paths, cfg=cfg, validate=True)

    X_raw = compute_features(
        bid_prices=data.bid_prices,
        bid_sizes=data.bid_sizes,
        ask_prices=data.ask_prices,
        ask_sizes=data.ask_sizes,
        midprice=data.midprice,
        timestamps=data.t,
    )

    label_result = create_directional_labels(
        midprice=data.midprice,
        horizon=500,
        label_mode="binary_drop_ties",
    )

    X, y = align_features_with_labels(X_raw, label_result)
    delta_aligned = label_result.delta[label_result.valid_mask]
    t_aligned = data.t[:-label_result.horizon][label_result.valid_mask]
    mid_aligned = data.midprice[:-label_result.horizon][label_result.valid_mask]

    print("\n=== ALIGNED DATASET ===")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print_classwise_feature_means(X, y, FEATURE_NAMES)

    plot_forward_delta_histogram(
        delta=label_result.delta,
        out_path=FIG_DIR / "forward_delta_histogram.png",
    )

    plot_ofi_quantile_response(
        ofi_feature=X[:, FEATURE_NAMES.index("ofi_roll_sum_50")],
        delta=delta_aligned,
        y=y,
        out_path=FIG_DIR / "ofi_quantile_response.png",
    )

    plot_class_conditional_boxplots(
        X=X,
        y=y,
        feature_names=FEATURE_NAMES,
        selected_features=[
            "ofi_best",
            "ofi_roll_sum_50",
            "queue_imbalance_best",
            "microprice_deviation",
        ],
        out_path=FIG_DIR / "class_conditional_boxplots.png",
    )

    plot_feature_correlation_heatmap(
        X=X,
        feature_names=FEATURE_NAMES,
        out_path=FIG_DIR / "feature_correlation_heatmap.png",
    )

    plot_time_slice(
        timestamps=t_aligned,
        midprice=mid_aligned,
        ofi_roll_sum=X[:, FEATURE_NAMES.index("ofi_roll_sum_50")],
        event_intensity=X[:, FEATURE_NAMES.index("event_intensity_1s")],
        out_path=FIG_DIR / "time_slice_diagnostics.png",
        n_points=5000,
    )

    print("\n=== DIAGNOSTIC PLOTS SAVED ===")
    for fname in [
        "forward_delta_histogram.png",
        "ofi_quantile_response.png",
        "class_conditional_boxplots.png",
        "feature_correlation_heatmap.png",
        "time_slice_diagnostics.png",
    ]:
        print(FIG_DIR / fname)

    print("\nDone.\n")


if __name__ == "__main__":
    main()