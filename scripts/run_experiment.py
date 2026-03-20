from __future__ import annotations
from dataclasses import asdict

from microalpha.config import load_experiment_config
from microalpha.evaluation import (
    evaluate_binary_classifier,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_score_distribution,
    summarize_evaluation,
)
from microalpha.features import make_feature_names
from microalpha.models import (
    get_logistic_coefficients,
    predict_classes,
    predict_probabilities,
    summarize_split,
    train_model,
)
from microalpha.pipeline import (
    build_ticker_dataset,
    make_ticker_split_summary,
    split_and_pool_datasets,
)
from microalpha.utils import (
    make_artifact_dirs,
    make_dataset_prefix,
    make_run_id,
    save_json,
    setup_logger,
    stringify_metrics,
)


def main() -> None:
    cfg = load_experiment_config("config/experiment.yaml")

    symbols = [ticker.symbol for ticker in cfg.dataset.tickers]
    run_id = make_run_id(prefix=f"h{cfg.labels.horizon}_{make_dataset_prefix(symbols)}")
    dirs = make_artifact_dirs(run_id)
    logger = setup_logger(dirs["logs"] / "run.log")

    logger.info("Starting experiment run: %s", run_id)
    logger.info("Tickers: %s", symbols)
    save_json(asdict(cfg), dirs["root"] / "config.json")

    datasets = []
    for ticker_cfg in cfg.dataset.tickers:
        logger.info("Building dataset for %s", ticker_cfg.symbol)
        dataset = build_ticker_dataset(ticker_cfg=ticker_cfg, cfg=cfg)
        logger.info(
            "Built dataset for %s: n_events=%s, X shape=%s, y shape=%s, tie_rate=%.6f",
            dataset.symbol,
            dataset.n_events,
            dataset.X.shape,
            dataset.y.shape,
            dataset.label_summary["tie_rate"],
        )
        datasets.append(dataset)

    split, ticker_splits = split_and_pool_datasets(
        datasets=datasets,
        train_fraction=cfg.split.train_fraction,
    )

    split_summary = {
        **summarize_split(split),
        "pooled": len(ticker_splits) > 1,
        "n_tickers": len(ticker_splits),
        "symbols": [ts.symbol for ts in ticker_splits],
    }
    save_json(split_summary, dirs["root"] / "split_summary.json")
    save_json(
        {"tickers": [make_ticker_split_summary(ts) for ts in ticker_splits]},
        dirs["root"] / "ticker_summaries.json",
    )
    logger.info("Pooled split summary: %s", split_summary)

    feature_names = make_feature_names(cfg.features)

    logistic_model = train_model(
        X_train=split.X_train,
        y_train=split.y_train,
        model_name="logistic",
        logistic_random_state=cfg.models.logistic.random_state,
    )
    logger.info("Trained logistic regression")

    hist_gbdt_model = train_model(
        X_train=split.X_train,
        y_train=split.y_train,
        model_name="hist_gbdt",
        hist_gbdt_random_state=cfg.models.hist_gbdt.random_state,
        hist_gbdt_learning_rate=cfg.models.hist_gbdt.learning_rate,
        hist_gbdt_max_iter=cfg.models.hist_gbdt.max_iter,
        hist_gbdt_max_leaf_nodes=cfg.models.hist_gbdt.max_leaf_nodes,
        hist_gbdt_min_samples_leaf=cfg.models.hist_gbdt.min_samples_leaf,
        hist_gbdt_l2_regularization=cfg.models.hist_gbdt.l2_regularization,
    )
    logger.info("Trained HistGradientBoostingClassifier")

    logistic_y_pred = predict_classes(logistic_model, split.X_test)
    logistic_y_proba = predict_probabilities(logistic_model, split.X_test)

    hist_gbdt_y_pred = predict_classes(hist_gbdt_model, split.X_test)
    hist_gbdt_y_proba = predict_probabilities(hist_gbdt_model, split.X_test)

    logistic_result = evaluate_binary_classifier(
        model_name="logistic",
        y_true=split.y_test,
        y_pred=logistic_y_pred,
        y_proba=logistic_y_proba,
    )
    hist_gbdt_result = evaluate_binary_classifier(
        model_name="hist_gbdt",
        y_true=split.y_test,
        y_pred=hist_gbdt_y_pred,
        y_proba=hist_gbdt_y_proba,
    )

    metrics = {
        "logistic": stringify_metrics(summarize_evaluation(logistic_result)),
        "hist_gbdt": stringify_metrics(summarize_evaluation(hist_gbdt_result)),
    }
    save_json(metrics, dirs["root"] / "metrics.json")
    logger.info("Metrics: %s", metrics)

    logistic_coefs = get_logistic_coefficients(logistic_model, feature_names)
    save_json(
        {
            "coefficients": [
                {"feature": feature, "coefficient": coefficient}
                for feature, coefficient in logistic_coefs
            ]
        },
        dirs["root"] / "logistic_coefficients.json",
    )

    plot_roc_curve(
        split.y_test,
        logistic_result.y_proba,
        dirs["figures"] / "roc_logistic.png",
        "Logistic Regression",
    )
    plot_score_distribution(
        split.y_test,
        logistic_result.y_proba,
        dirs["figures"] / "score_distribution_logistic.png",
        "Logistic Regression",
    )
    plot_confusion_matrix(
        logistic_result.confusion_matrix,
        dirs["figures"] / "confusion_matrix_logistic.png",
        "Logistic Regression",
    )

    plot_roc_curve(
        split.y_test,
        hist_gbdt_result.y_proba,
        dirs["figures"] / "roc_hist_gbdt.png",
        "HistGradientBoosting",
    )
    plot_score_distribution(
        split.y_test,
        hist_gbdt_result.y_proba,
        dirs["figures"] / "score_distribution_hist_gbdt.png",
        "HistGradientBoosting",
    )
    plot_confusion_matrix(
        hist_gbdt_result.confusion_matrix,
        dirs["figures"] / "confusion_matrix_hist_gbdt.png",
        "HistGradientBoosting",
    )

    logger.info("Saved evaluation plots")
    logger.info("Experiment complete: %s", run_id)

    print(f"\nRun complete: {run_id}")
    print(f"Artifacts saved under: {dirs['root']}\n")


if __name__ == "__main__":
    main()