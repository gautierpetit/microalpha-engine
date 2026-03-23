from __future__ import annotations
from dataclasses import asdict

from microalpha.config import load_experiment_config
from microalpha.evaluation import (
    evaluate_binary_classifier,
    plot_confusion_matrix_comparison,
    plot_feature_importance_barh,
    plot_model_metric_comparison,
    plot_per_ticker_auc_comparison,
    plot_roc_comparison,
    plot_score_distribution_comparison,
    summarize_evaluation,
)
from microalpha.features import make_feature_names
from microalpha.models import (
    compute_permutation_importance,
    get_logistic_coefficients,
    predict_classes,
    predict_probabilities,
    save_trained_model,
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
    save_rows_csv,
    setup_logger,
    stringify_metrics,
)
from microalpha.diagnostics import (
    flatten_feature_importance,
    flatten_pooled_ticker_metrics,
    summarize_ticker_datasets,
    summarize_ticker_feature_diagnostics,
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

    

    dataset_diagnostics = summarize_ticker_datasets(datasets)
    save_json(dataset_diagnostics, dirs["root"] / "ticker_dataset_diagnostics.json")

    feature_names = make_feature_names(cfg.features)
    feature_diagnostics = summarize_ticker_feature_diagnostics(datasets, feature_names)
    save_json(feature_diagnostics, dirs["root"] / "ticker_feature_diagnostics.json")

    logger.info("Saved dataset and feature diagnostics")

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

    save_trained_model(logistic_model, dirs["models"] / "logistic.joblib")
    save_trained_model(hist_gbdt_model, dirs["models"] / "hist_gbdt.joblib")
    logger.info("Saved trained models")

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
                "logistic": stringify_metrics(
                    summarize_evaluation(logistic_result_ticker)
                ),
                "hist_gbdt": stringify_metrics(
                    summarize_evaluation(hist_gbdt_result_ticker)
                ),
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

    pooled_ticker_rows = flatten_pooled_ticker_metrics(pooled_ticker_metrics)
    save_rows_csv(
        pooled_ticker_rows,
        dirs["tables"] / "pooled_ticker_metrics.csv",
    )
    logger.info("Saved pooled per-ticker metrics")

    N_REPEATS = 10
    logistic_perm_importance = compute_permutation_importance(
        logistic_model,
        split.X_test,
        split.y_test,
        feature_names,
        scoring="roc_auc",
        n_repeats=N_REPEATS,
        random_state=cfg.models.logistic.random_state,
        n_jobs=-1,
    )
    hist_gbdt_perm_importance = compute_permutation_importance(
        hist_gbdt_model,
        split.X_test,
        split.y_test,
        feature_names,
        scoring="roc_auc",
        n_repeats=N_REPEATS,
        random_state=cfg.models.hist_gbdt.random_state,
        n_jobs=-1,
    )

    save_json(
        {
            "model_name": "logistic",
            "scoring": "roc_auc",
            "n_repeats": N_REPEATS,
            "importances": logistic_perm_importance,
        },
        dirs["root"] / "logistic_permutation_importance.json",
    )
    save_json(
        {
            "model_name": "hist_gbdt",
            "scoring": "roc_auc",
            "n_repeats": N_REPEATS,
            "importances": hist_gbdt_perm_importance,
        },
        dirs["root"] / "hist_gbdt_permutation_importance.json",
    )
    save_rows_csv(
        flatten_feature_importance(
            {
                "model_name": "logistic",
                "importances": logistic_perm_importance,
            }
        ),
        dirs["tables"] / "logistic_permutation_importance.csv",
    )

    save_rows_csv(
        flatten_feature_importance(
            {
                "model_name": "hist_gbdt",
                "importances": hist_gbdt_perm_importance,
            }
        ),
        dirs["tables"] / "hist_gbdt_permutation_importance.csv",
    )
    logger.info("Saved permutation importance artifacts")

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

    plot_roc_comparison(
        logistic_result,
        hist_gbdt_result,
        dirs["figures"] / "roc_comparison.png",
        )

    plot_score_distribution_comparison(
        logistic_result,
        hist_gbdt_result,
        dirs["figures"] / "score_distribution_comparison.png",
    )

    plot_confusion_matrix_comparison(
        logistic_result,
        hist_gbdt_result,
        dirs["figures"] / "confusion_matrix_comparison.png",
    )

    logger.info("Saved evaluation plots")
    TOP_N = 12
    plot_feature_importance_barh(
        logistic_perm_importance,
        dirs["figures"] / "feature_importance_logistic.png",
        "Logistic Regression",
        top_n=TOP_N,
    )
    plot_feature_importance_barh(
        hist_gbdt_perm_importance,
        dirs["figures"] / "feature_importance_hist_gbdt.png",
        "HistGradientBoosting",
        top_n=TOP_N,
    )
    plot_model_metric_comparison(
        metrics,
        dirs["figures"] / "model_comparison_roc_auc.png",
        metric_name="roc_auc",
    )

    plot_per_ticker_auc_comparison(
        pooled_ticker_metrics,
        dirs["figures"] / "per_ticker_auc_comparison.png",
    )
    logger.info("Saved comparison charts")
    logger.info("Saved feature importance plots")

    logger.info("Experiment complete: %s", run_id)

    print(f"\nRun complete: {run_id}")
    print(f"Artifacts saved under: {dirs['root']}\n")


if __name__ == "__main__":
    main()
