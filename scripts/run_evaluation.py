from __future__ import annotations

from pathlib import Path

from microalpha.evaluation import (
    evaluate_binary_classifier,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_score_distribution,
    summarize_evaluation,
)
from microalpha.features import compute_features
from microalpha.io import LobsterConfig, LobsterPaths, load_lobster
from microalpha.labels import align_features_with_labels, create_directional_labels
from microalpha.models import (
    predict_classes,
    predict_probabilities,
    summarize_split,
    time_train_test_split,
    train_model,
)


FIG_DIR = Path("docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load data
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

    # ------------------------------------------------------------------
    # 2. Compute features
    # ------------------------------------------------------------------
    X_raw = compute_features(
        bid_prices=data.bid_prices,
        bid_sizes=data.bid_sizes,
        ask_prices=data.ask_prices,
        ask_sizes=data.ask_sizes,
        midprice=data.midprice,
        timestamps=data.t,
    )

    # ------------------------------------------------------------------
    # 3. Create labels and align
    # ------------------------------------------------------------------
    label_result = create_directional_labels(
        midprice=data.midprice,
        horizon=500,
        label_mode="binary_drop_ties",
    )

    X, y = align_features_with_labels(X_raw, label_result)

    # ------------------------------------------------------------------
    # 4. Split
    # ------------------------------------------------------------------
    split = time_train_test_split(X, y, train_fraction=0.7)

    print("\n=== EVALUATION SPLIT SUMMARY ===")
    for k, v in summarize_split(split).items():
        if isinstance(v, float):
            print(f"{k:>16}: {v:.6f}")
        else:
            print(f"{k:>16}: {v}")

    # ------------------------------------------------------------------
    # 5. Train models
    # ------------------------------------------------------------------
    logistic_model = train_model(
        X_train=split.X_train,
        y_train=split.y_train,
        model_name="logistic",
        random_state=42,
    )

    mlp_model = train_model(
        X_train=split.X_train,
        y_train=split.y_train,
        model_name="mlp",
        random_state=42,
    )

    # ------------------------------------------------------------------
    # 6. Predict on test set
    # ------------------------------------------------------------------
    logistic_y_pred = predict_classes(logistic_model, split.X_test)
    logistic_y_proba = predict_probabilities(logistic_model, split.X_test)

    mlp_y_pred = predict_classes(mlp_model, split.X_test)
    mlp_y_proba = predict_probabilities(mlp_model, split.X_test)

    # ------------------------------------------------------------------
    # 7. Evaluate
    # ------------------------------------------------------------------
    logistic_result = evaluate_binary_classifier(
        model_name="logistic",
        y_true=split.y_test,
        y_pred=logistic_y_pred,
        y_proba=logistic_y_proba,
    )

    mlp_result = evaluate_binary_classifier(
        model_name="mlp",
        y_true=split.y_test,
        y_pred=mlp_y_pred,
        y_proba=mlp_y_proba,
    )

    print("\n=== EVALUATION SUMMARY ===")
    for result in [logistic_result, mlp_result]:
        summary = summarize_evaluation(result)
        print(f"\nModel: {summary['model_name']}")
        print(f"  accuracy: {summary['accuracy']:.6f}")
        print(f"  roc_auc : {summary['roc_auc']:.6f}")
        print("  confusion_matrix:")
        print(result.confusion_matrix)

    # ------------------------------------------------------------------
    # 8. Save plots
    # ------------------------------------------------------------------
    plot_roc_curve(
        y_true=split.y_test,
        y_proba=logistic_result.y_proba,
        out_path=FIG_DIR / "roc_logistic.png",
        model_name="Logistic Regression",
    )
    plot_score_distribution(
        y_true=split.y_test,
        y_proba=logistic_result.y_proba,
        out_path=FIG_DIR / "score_distribution_logistic.png",
        model_name="Logistic Regression",
    )
    plot_confusion_matrix(
        cm=logistic_result.confusion_matrix,
        out_path=FIG_DIR / "confusion_matrix_logistic.png",
        model_name="Logistic Regression",
    )

    plot_roc_curve(
        y_true=split.y_test,
        y_proba=mlp_result.y_proba,
        out_path=FIG_DIR / "roc_mlp.png",
        model_name="Small MLP",
    )
    plot_score_distribution(
        y_true=split.y_test,
        y_proba=mlp_result.y_proba,
        out_path=FIG_DIR / "score_distribution_mlp.png",
        model_name="Small MLP",
    )
    plot_confusion_matrix(
        cm=mlp_result.confusion_matrix,
        out_path=FIG_DIR / "confusion_matrix_mlp.png",
        model_name="Small MLP",
    )

    print("\n=== EVALUATION PLOTS SAVED ===")
    for fname in [
        "roc_logistic.png",
        "score_distribution_logistic.png",
        "confusion_matrix_logistic.png",
        "roc_mlp.png",
        "score_distribution_mlp.png",
        "confusion_matrix_mlp.png",
    ]:
        print(FIG_DIR / fname)

    print("\nDone.\n")


if __name__ == "__main__":
    main()