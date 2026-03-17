from __future__ import annotations

from pathlib import Path

from microalpha.features import FEATURE_NAMES, compute_features
from microalpha.io import LobsterConfig, LobsterPaths, load_lobster
from microalpha.labels import align_features_with_labels, create_directional_labels
from microalpha.models import (
    get_logistic_coefficients,
    summarize_split,
    time_train_test_split,
    train_model,
)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load raw LOBSTER data
    # ------------------------------------------------------------------
    msg_path = Path(
        r"data\raw\LOBSTER_SampleFile_AAPL_2012-06-21_10\AAPL_2012-06-21_34200000_57600000_message_10.csv"
    )
    ob_path = Path(
        r"data\raw\LOBSTER_SampleFile_AAPL_2012-06-21_10\AAPL_2012-06-21_34200000_57600000_orderbook_10.csv"
    )

    paths = LobsterPaths(message_csv=msg_path, orderbook_csv=ob_path)
    cfg = LobsterConfig(levels=10, price_scale=10_000)

    data = load_lobster(paths, cfg=cfg, validate=True)

    print("\n=== DATA LOAD COMPLETE ===")
    print(f"events: {len(data.t):,}")
    print(f"levels: {data.bid_prices.shape[1]}")

    # ------------------------------------------------------------------
    # 2. Compute raw + temporal features
    # ------------------------------------------------------------------
    X_raw = compute_features(
        bid_prices=data.bid_prices,
        bid_sizes=data.bid_sizes,
        ask_prices=data.ask_prices,
        ask_sizes=data.ask_sizes,
        midprice=data.midprice,
        timestamps=data.t,
    )

    print("\n=== FEATURE EXTRACTION COMPLETE ===")
    print(f"X_raw shape: {X_raw.shape}")

    # ------------------------------------------------------------------
    # 3. Create labels
    # ------------------------------------------------------------------
    label_result = create_directional_labels(
        midprice=data.midprice,
        horizon=500,
        label_mode="binary_drop_ties",
    )

    print("\n=== LABEL CREATION COMPLETE ===")
    print(f"horizon: {label_result.horizon}")
    print(f"label_mode: {label_result.label_mode}")
    print(f"tie_rate: {label_result.tie_rate:.4%}")
    print(f"n_raw: {label_result.n_raw:,}")
    print(f"n_final: {label_result.n_final:,}")

    # ------------------------------------------------------------------
    # 4. Align features with labels
    # ------------------------------------------------------------------
    X, y = align_features_with_labels(X_raw, label_result)

    print("\n=== FEATURE/LABEL ALIGNMENT COMPLETE ===")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # ------------------------------------------------------------------
    # 5. Time-ordered train/test split
    # ------------------------------------------------------------------
    split = time_train_test_split(X, y, train_fraction=0.7)
    split_summary = summarize_split(split)

    print("\n=== TRAIN / TEST SPLIT ===")
    for k, v in split_summary.items():
        if isinstance(v, float):
            print(f"{k:>16}: {v:.6f}")
        else:
            print(f"{k:>16}: {v}")

    # ------------------------------------------------------------------
    # 6. Train logistic regression
    # ------------------------------------------------------------------
    print("\n=== TRAINING LOGISTIC REGRESSION ===")
    logistic_model = train_model(
        X_train=split.X_train,
        y_train=split.y_train,
        model_name="logistic",
        random_state=42,
    )
    print("logistic regression training complete")

    # Inspect coefficients
    coef_pairs = get_logistic_coefficients(logistic_model, FEATURE_NAMES)

    print("\nTop logistic coefficients (sorted by |coefficient|):")
    for name, coef in coef_pairs:
        print(f"{name:>24}: {coef:+.6f}")

    # ------------------------------------------------------------------
    # 7. Train small MLP
    # ------------------------------------------------------------------
    print("\n=== TRAINING SMALL MLP ===")
    mlp_model = train_model(
        X_train=split.X_train,
        y_train=split.y_train,
        model_name="mlp",
        random_state=42,
    )
    print("small MLP training complete")

    mlp_clf = mlp_model.named_steps["clf"]
    print(f"MLP iterations: {mlp_clf.n_iter_}")
    print(f"MLP final loss: {mlp_clf.loss_:.6f}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()