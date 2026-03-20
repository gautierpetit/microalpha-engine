from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from microalpha.config import ExperimentConfig, TickerConfig, config_to_dict, load_experiment_config
from microalpha.evaluation import (
    evaluate_binary_classifier,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_score_distribution,
    summarize_evaluation,
)
from microalpha.features import TemporalFeatureConfig, compute_features, make_feature_names
from microalpha.io import LobsterConfig, LobsterPaths, load_lobster
from microalpha.labels import align_features_with_labels, create_directional_labels, summarize_labels
from microalpha.models import (
    SplitResult,
    get_logistic_coefficients,
    predict_classes,
    predict_probabilities,
    summarize_split,
    time_train_test_split,
    train_model,
)
from microalpha.utils import make_artifact_dirs, make_run_id, save_json, setup_logger, stringify_metrics


@dataclass(frozen=True)
class TickerDataset:
    symbol: str
    X: np.ndarray
    y: np.ndarray
    n_events: int
    label_summary: dict[str, float | int | str]


@dataclass(frozen=True)
class TickerSplit:
    symbol: str
    split: SplitResult
    n_events: int
    label_summary: dict[str, float | int | str]


def build_ticker_dataset(
    ticker_cfg: TickerConfig,
    cfg: ExperimentConfig,
    feature_cfg: TemporalFeatureConfig,
) -> TickerDataset:
    lobster_paths = LobsterPaths(
        message_csv=Path(ticker_cfg.message_csv),
        orderbook_csv=Path(ticker_cfg.orderbook_csv),
    )
    lobster_cfg = LobsterConfig(
        levels=cfg.dataset.levels,
        price_scale=cfg.dataset.price_scale,
    )

    data = load_lobster(lobster_paths, cfg=lobster_cfg, validate=True)

    X_raw = compute_features(
        bid_prices=data.bid_prices,
        bid_sizes=data.bid_sizes,
        ask_prices=data.ask_prices,
        ask_sizes=data.ask_sizes,
        midprice=data.midprice,
        timestamps=data.t,
        cfg=feature_cfg,
    )

    label_result = create_directional_labels(
        midprice=data.midprice,
        horizon=cfg.labels.horizon,
        label_mode=cfg.labels.label_mode,
    )
    X, y = align_features_with_labels(X_raw, label_result)

    return TickerDataset(
        symbol=ticker_cfg.symbol,
        X=X,
        y=y,
        n_events=len(data.t),
        label_summary=summarize_labels(label_result),
    )


def split_and_pool_datasets(
    datasets: list[TickerDataset],
    train_fraction: float,
) -> tuple[SplitResult, list[TickerSplit]]:
    if not datasets:
        raise ValueError("datasets must be non-empty")

    ticker_splits: list[TickerSplit] = []
    for dataset in datasets:
        split = time_train_test_split(dataset.X, dataset.y, train_fraction=train_fraction)
        ticker_splits.append(
            TickerSplit(
                symbol=dataset.symbol,
                split=split,
                n_events=dataset.n_events,
                label_summary=dataset.label_summary,
            )
        )

    pooled_split = SplitResult(
        X_train=np.vstack([ticker_split.split.X_train for ticker_split in ticker_splits]),
        X_test=np.vstack([ticker_split.split.X_test for ticker_split in ticker_splits]),
        y_train=np.concatenate([ticker_split.split.y_train for ticker_split in ticker_splits]),
        y_test=np.concatenate([ticker_split.split.y_test for ticker_split in ticker_splits]),
        split_idx=-1,
        train_fraction=train_fraction,
        n_train=sum(ticker_split.split.n_train for ticker_split in ticker_splits),
        n_test=sum(ticker_split.split.n_test for ticker_split in ticker_splits),
    )
    return pooled_split, ticker_splits


def make_dataset_prefix(cfg: ExperimentConfig) -> str:
    symbols = [ticker.symbol.lower() for ticker in cfg.dataset.tickers]
    if len(symbols) == 1:
        return symbols[0]
    return f"pooled_{len(symbols)}t"


def make_ticker_split_summary(ticker_split: TickerSplit) -> dict[str, float | int | str]:
    return {
        "symbol": ticker_split.symbol,
        "n_events": ticker_split.n_events,
        **ticker_split.label_summary,
        **summarize_split(ticker_split.split),
    }


def main() -> None:
    cfg = load_experiment_config("config/experiment.yaml")

    run_id = make_run_id(prefix=f"h{cfg.labels.horizon}_{make_dataset_prefix(cfg)}")
    dirs = make_artifact_dirs(run_id)
    logger = setup_logger(dirs["logs"] / "run.log")

    logger.info("Starting experiment run: %s", run_id)
    logger.info("Tickers: %s", [ticker.symbol for ticker in cfg.dataset.tickers])
    save_json(config_to_dict(cfg), dirs["root"] / "config.json")

    feature_cfg = TemporalFeatureConfig(
        ofi_window_raw=cfg.features.ofi_window_raw,
        ofi_norm_windows=tuple(cfg.features.ofi_norm_windows),
        vol_window=cfg.features.vol_window,
        intensity_window=cfg.features.intensity_window,
    )
    feature_names = make_feature_names(feature_cfg)

    datasets: list[TickerDataset] = []
    for ticker_cfg in cfg.dataset.tickers:
        logger.info("Building dataset for %s", ticker_cfg.symbol)
        dataset = build_ticker_dataset(
            ticker_cfg=ticker_cfg,
            cfg=cfg,
            feature_cfg=feature_cfg,
        )
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
        "symbols": [ticker_split.symbol for ticker_split in ticker_splits],
    }
    save_json(split_summary, dirs["root"] / "split_summary.json")
    save_json(
        {"tickers": [make_ticker_split_summary(ticker_split) for ticker_split in ticker_splits]},
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
    logger.info("Saved logistic coefficients")

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