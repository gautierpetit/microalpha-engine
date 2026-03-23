from __future__ import annotations

from typing import Any

import numpy as np

from microalpha.pipeline import TickerDataset


def summarize_feature_matrix(
    X: np.ndarray,
    feature_names: list[str],
    *,
    percentiles: tuple[float, ...] = (0.01, 0.05, 0.50, 0.95, 0.99),
) -> list[dict[str, float | str]]:
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"feature_names length {len(feature_names)} does not match X.shape[1]={X.shape[1]}"
        )

    rows: list[dict[str, float | str]] = []
    for j, feature in enumerate(feature_names):
        col = X[:, j]
        row: dict[str, float | str] = {
            "feature": feature,
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }
        for q in percentiles:
            row[f"p{int(q * 100):02d}"] = float(np.quantile(col, q))
        rows.append(row)

    return rows


def summarize_ticker_feature_diagnostics(
    datasets: list[TickerDataset],
    feature_names: list[str],
) -> dict[str, Any]:
    return {
        "tickers": [
            {
                "symbol": ds.symbol,
                "n_rows": int(ds.X.shape[0]),
                "n_features": int(ds.X.shape[1]),
                "features": summarize_feature_matrix(ds.X, feature_names),
            }
            for ds in datasets
        ]
    }


def flatten_pooled_ticker_metrics(
    pooled_ticker_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ticker in pooled_ticker_metrics["tickers"]:
        rows.append(
            {
                "symbol": ticker["symbol"],
                "n_test": ticker["n_test"],
                "test_pos_rate": ticker["test_pos_rate"],
                "logistic_accuracy": ticker["logistic"]["accuracy"],
                "logistic_roc_auc": ticker["logistic"]["roc_auc"],
                "hist_gbdt_accuracy": ticker["hist_gbdt"]["accuracy"],
                "hist_gbdt_roc_auc": ticker["hist_gbdt"]["roc_auc"],
            }
        )
    return rows


def flatten_feature_importance(
    importance_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "model_name": importance_payload["model_name"],
            "feature": row["feature"],
            "importance_mean": row["importance_mean"],
            "importance_std": row["importance_std"],
        }
        for row in importance_payload["importances"]
    ]
