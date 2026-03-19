from __future__ import annotations

from pathlib import Path

from microalpha.config import config_to_dict, load_experiment_config
from microalpha.evaluation import (
    evaluate_binary_classifier,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_score_distribution,
    summarize_evaluation,
)
from microalpha.features import TemporalFeatureConfig, compute_features, make_feature_names
from microalpha.io import LobsterConfig, LobsterPaths, load_lobster
from microalpha.labels import align_features_with_labels, create_directional_labels
from microalpha.models import (
    get_logistic_coefficients,
    predict_classes,
    predict_probabilities,
    summarize_split,
    time_train_test_split,
    train_model,
)
from microalpha.utils import make_artifact_dirs, make_run_id, save_json, setup_logger, stringify_metrics


def main() -> None:
    cfg = load_experiment_config("config/experiment.yaml")

    run_id = make_run_id(prefix=f"h{cfg.labels.horizon}_aapl")
    dirs = make_artifact_dirs(run_id)
    logger = setup_logger(dirs["logs"] / "run.log")

    logger.info("Starting experiment run: %s", run_id)
    save_json(config_to_dict(cfg), dirs["root"] / "config.json")

    lobster_paths = LobsterPaths(
        message_csv=Path(cfg.dataset.message_csv),
        orderbook_csv=Path(cfg.dataset.orderbook_csv),
    )
    lobster_cfg = LobsterConfig(
        levels=cfg.dataset.levels,
        price_scale=cfg.dataset.price_scale,
    )

    data = load_lobster(lobster_paths, cfg=lobster_cfg, validate=True)
    logger.info("Loaded data: %s events, %s levels", len(data.t), data.bid_prices.shape[1])

    feature_cfg = TemporalFeatureConfig(
        ofi_window_raw=cfg.features.ofi_window_raw,
        ofi_norm_windows=tuple(cfg.features.ofi_norm_windows),
        vol_window=cfg.features.vol_window,
        intensity_window=cfg.features.intensity_window,
    )
    feature_names = make_feature_names(feature_cfg)

    X_raw = compute_features(
        bid_prices=data.bid_prices,
        bid_sizes=data.bid_sizes,
        ask_prices=data.ask_prices,
        ask_sizes=data.ask_sizes,
        midprice=data.midprice,
        timestamps=data.t,
        cfg=feature_cfg,
    )
    logger.info("Computed features: X_raw shape=%s", X_raw.shape)

    label_result = create_directional_labels(
        midprice=data.midprice,
        horizon=cfg.labels.horizon,
        label_mode=cfg.labels.label_mode,
    )
    logger.info(
        "Created labels: mode=%s, horizon=%s, tie_rate=%.6f, n_final=%s",
        label_result.label_mode,
        label_result.horizon,
        label_result.tie_rate,
        label_result.n_final,
    )

    X, y = align_features_with_labels(X_raw, label_result)
    logger.info("Aligned dataset: X shape=%s, y shape=%s", X.shape, y.shape)

    split = time_train_test_split(X, y, train_fraction=cfg.split.train_fraction)
    split_summary = summarize_split(split)
    save_json(split_summary, dirs["root"] / "split_summary.json")
    logger.info("Split summary: %s", split_summary)

    logistic_model = train_model(
        X_train=split.X_train,
        y_train=split.y_train,
        model_name="logistic",
        random_state=cfg.models.logistic.random_state,
    )
    logger.info("Trained logistic regression")

    mlp_model = train_model(
        X_train=split.X_train,
        y_train=split.y_train,
        model_name="mlp",
        random_state=cfg.models.mlp.random_state,
    )
    logger.info("Trained MLP")

    logistic_y_pred = predict_classes(logistic_model, split.X_test)
    logistic_y_proba = predict_probabilities(logistic_model, split.X_test)

    mlp_y_pred = predict_classes(mlp_model, split.X_test)
    mlp_y_proba = predict_probabilities(mlp_model, split.X_test)

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

    metrics = {
        "logistic": stringify_metrics(summarize_evaluation(logistic_result)),
        "mlp": stringify_metrics(summarize_evaluation(mlp_result)),
    }
    save_json(metrics, dirs["root"] / "metrics.json")
    logger.info("Saved metrics")

    logistic_coefs = get_logistic_coefficients(logistic_model, feature_names)
    save_json(
        {"coefficients": [{"feature": f, "coefficient": c} for f, c in logistic_coefs]},
        dirs["root"] / "logistic_coefficients.json",
    )
    logger.info("Saved logistic coefficients")

    # plots
    plot_roc_curve(split.y_test, logistic_result.y_proba, dirs["figures"] / "roc_logistic.png", "Logistic Regression")
    plot_score_distribution(split.y_test, logistic_result.y_proba, dirs["figures"] / "score_distribution_logistic.png", "Logistic Regression")
    plot_confusion_matrix(logistic_result.confusion_matrix, dirs["figures"] / "confusion_matrix_logistic.png", "Logistic Regression")

    plot_roc_curve(split.y_test, mlp_result.y_proba, dirs["figures"] / "roc_mlp.png", "Small MLP")
    plot_score_distribution(split.y_test, mlp_result.y_proba, dirs["figures"] / "score_distribution_mlp.png", "Small MLP")
    plot_confusion_matrix(mlp_result.confusion_matrix, dirs["figures"] / "confusion_matrix_mlp.png", "Small MLP")

    logger.info("Saved evaluation plots")
    logger.info("Experiment complete: %s", run_id)

    print(f"\nRun complete: {run_id}")
    print(f"Artifacts saved under: {dirs['root']}\n")


if __name__ == "__main__":
    main()