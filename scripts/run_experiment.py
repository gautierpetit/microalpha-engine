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
    iter_ticker_test_segments,
    make_ticker_split_summary,
    split_and_pool_datasets,
)
from microalpha.utils import (
    make_artifact_dirs,
    make_experiment_prefix,
    make_run_id,
    save_json,
    setup_logger,
    stringify_metrics,
)


def main() -> None:
    cfg = load_experiment_config("config/experiment.yaml")

    symbols = [ticker.symbol for ticker in cfg.dataset.tickers]
    prefix = make_experiment_prefix(
        horizon=cfg.labels.horizon,
        task_name=cfg.task.name,
        symbols=symbols,
    )
    run_id = make_run_id(prefix=prefix)

    dirs = make_artifact_dirs(run_id)
    logger = setup_logger(dirs["logs"] / "run.log")

    logger.info("Starting experiment run: %s", run_id)
    logger.info("Task: %s", cfg.task.name)
    logger.info("Tickers: %s", symbols)

    save_json(asdict(cfg), dirs["root"] / "config.json")

    datasets = []
    for ticker_cfg in cfg.dataset.tickers:
        logger.info("Building dataset for %s", ticker_cfg.symbol)
        dataset = build_ticker_dataset(ticker_cfg=ticker_cfg, cfg=cfg)
        logger.info(
            "Built dataset for %s: n_events=%s, X shape=%s, y shape=%s, tie_rate=%.6f, move_rate=%.6f",
            dataset.symbol,
            dataset.n_events,
            dataset.X.shape,
            dataset.y.shape,
            dataset.label_summary["tie_rate"],
            dataset.label_summary["move_rate"],
        )
        datasets.append(dataset)

    split, ticker_splits = split_and_pool_datasets(
        datasets=datasets,
        train_fraction=cfg.split.train_fraction,
    )

    split_summary = {
        **summarize_split(split),
        "task_name": cfg.task.name,
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
        "task_name": cfg.task.name,
        "logistic": stringify_metrics(summarize_evaluation(logistic_result)),
        "hist_gbdt": stringify_metrics(summarize_evaluation(hist_gbdt_result)),
    }
    save_json(metrics, dirs["root"] / "metrics.json")
    logger.info("Metrics: %s", metrics)

    pooled_ticker_metrics: dict[str, object] = {
        "task_name": cfg.task.name,
        "label_mode": cfg.labels.label_mode,
        "tickers": [],
    }

    for ticker_split, test_slice in iter_ticker_test_segments(ticker_splits):
        y_true_ticker = split.y_test[test_slice]

        logistic_result_ticker = evaluate_binary_classifier(
            model_name="logistic",
            y_true=y_true_ticker,
            y_pred=logistic_y_pred[test_slice],
            y_proba=logistic_y_proba[test_slice],
        )

        hist_gbdt_result_ticker = evaluate_binary_classifier(
            model_name="hist_gbdt",
            y_true=y_true_ticker,
            y_pred=hist_gbdt_y_pred[test_slice],
            y_proba=hist_gbdt_y_proba[test_slice],
        )

        pooled_ticker_metrics["tickers"].append(
            {
                "symbol": ticker_split.symbol,
                "n_test": ticker_split.split.n_test,
                "test_pos_rate": float(ticker_split.split.y_test.mean()),
                "logistic": stringify_metrics(summarize_evaluation(logistic_result_ticker)),
                "hist_gbdt": stringify_metrics(summarize_evaluation(hist_gbdt_result_ticker)),
            }
        )
    
    total_segment_n = sum(ts.split.n_test for ts in ticker_splits)
    if total_segment_n != split.y_test.shape[0]:
        raise ValueError(
            f"Per-ticker test segments sum to {total_segment_n}, "
            f"but pooled y_test has length {split.y_test.shape[0]}"
        )

    save_json(
        pooled_ticker_metrics,
        dirs["root"] / "pooled_ticker_metrics.json",
    )
    logger.info("Saved pooled per-ticker metrics")    

    feature_names = make_feature_names(cfg.features)
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
    logger.info("Saved logistic coefficients")

    plot_roc_curve(
        split.y_test,
        logistic_result.y_proba,
        dirs["figures"] / "roc_logistic.png",
        f"Logistic Regression ({cfg.task.name})",
    )
    plot_score_distribution(
        split.y_test,
        logistic_result.y_proba,
        dirs["figures"] / "score_distribution_logistic.png",
        f"Logistic Regression ({cfg.task.name})",
    )
    plot_confusion_matrix(
        logistic_result.confusion_matrix,
        dirs["figures"] / "confusion_matrix_logistic.png",
        f"Logistic Regression ({cfg.task.name})",
    )

    plot_roc_curve(
        split.y_test,
        hist_gbdt_result.y_proba,
        dirs["figures"] / "roc_hist_gbdt.png",
        f"HistGradientBoosting ({cfg.task.name})",
    )
    plot_score_distribution(
        split.y_test,
        hist_gbdt_result.y_proba,
        dirs["figures"] / "score_distribution_hist_gbdt.png",
        f"HistGradientBoosting ({cfg.task.name})",
    )
    plot_confusion_matrix(
        hist_gbdt_result.confusion_matrix,
        dirs["figures"] / "confusion_matrix_hist_gbdt.png",
        f"HistGradientBoosting ({cfg.task.name})",
    )

    logger.info("Saved evaluation plots")
    logger.info("Experiment complete: %s", run_id)

    print(f"\nRun complete: {run_id}")
    print(f"Artifacts saved under: {dirs['root']}\n")


if __name__ == "__main__":
    main()
