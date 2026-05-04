from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class EvaluationResult:
    model_name: str
    accuracy: float
    roc_auc: float
    confusion_matrix: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray


def evaluate_binary_classifier(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> EvaluationResult:
    """
    Evaluate a binary classifier using standard classification metrics.

    Parameters
    ----------
    model_name : str
        Name used for reporting.
    y_true : np.ndarray
        True binary labels of shape (N,).
    y_pred : np.ndarray
        Predicted binary labels of shape (N,).
    y_proba : np.ndarray
        Predicted positive-class probabilities of shape (N,).

    Returns
    -------
    EvaluationResult
    """
    _validate_eval_inputs(y_true, y_pred, y_proba)

    acc = float(accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_proba))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return EvaluationResult(
        model_name=model_name,
        accuracy=acc,
        roc_auc=auc,
        confusion_matrix=cm,
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )


def summarize_evaluation(result: EvaluationResult) -> dict[str, float | str]:
    """
    Compact summary for printing/logging.
    """
    return {
        "model_name": result.model_name,
        "accuracy": result.accuracy,
        "roc_auc": result.roc_auc,
    }


def plot_feature_importance_barh(
    importance_rows: list[dict[str, float | str]],
    out_path: Path,
    model_name: str,
    *,
    top_n: int = 12,
) -> None:
    """
    Save a horizontal bar chart of top permutation feature importances.
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}")

    rows = importance_rows[:top_n]
    if not rows:
        raise ValueError("importance_rows must be non-empty")

    features = [str(row["feature"]) for row in rows][::-1]
    means = [float(row["importance_mean"]) for row in rows][::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(features, means)
    plt.xlabel("Permutation importance (mean ROC AUC decrease)")
    plt.ylabel("Feature")
    plt.title(f"Top {min(top_n, len(rows))} Feature Importances - {model_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_model_metric_comparison(
    metrics_payload: dict,
    out_path: Path,
    *,
    metric_name: str = "roc_auc",
) -> None:
    model_names = ["logistic", "hist_gbdt"]
    values = [float(metrics_payload[m][metric_name]) for m in model_names]

    plt.figure(figsize=(7, 5))
    plt.bar(model_names, values)
    plt.ylabel(metric_name.upper())
    plt.title(f"Model Comparison - {metric_name.upper()}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_per_ticker_auc_comparison(
    pooled_ticker_metrics: dict,
    out_path: Path,
) -> None:
    tickers = [row["symbol"] for row in pooled_ticker_metrics["tickers"]]
    logistic_auc = [
        float(row["logistic"]["roc_auc"]) for row in pooled_ticker_metrics["tickers"]
    ]
    hist_auc = [
        float(row["hist_gbdt"]["roc_auc"]) for row in pooled_ticker_metrics["tickers"]
    ]

    x = np.arange(len(tickers))
    width = 0.38

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, logistic_auc, width, label="logistic")
    plt.bar(x + width / 2, hist_auc, width, label="hist_gbdt")
    plt.xticks(x, tickers)
    plt.ylabel("ROC AUC")
    plt.title("Per-Ticker ROC AUC (Pooled-Trained Models)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_comparison(
    logistic_result: EvaluationResult,
    hist_gbdt_result: EvaluationResult,
    out_path: Path,
) -> None:
    fpr_log, tpr_log, _ = roc_curve(logistic_result.y_true, logistic_result.y_proba)
    auc_log = auc(fpr_log, tpr_log)

    fpr_gbdt, tpr_gbdt, _ = roc_curve(hist_gbdt_result.y_true, hist_gbdt_result.y_proba)
    auc_gbdt = auc(fpr_gbdt, tpr_gbdt)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_log, tpr_log, label=f"Logistic (AUC={auc_log:.3f})")
    plt.plot(fpr_gbdt, tpr_gbdt, label=f"HistGBDT (AUC={auc_gbdt:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_score_distribution_comparison(
    logistic_result: EvaluationResult,
    hist_gbdt_result: EvaluationResult,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    y_true_log = logistic_result.y_true
    y_proba_log = logistic_result.y_proba

    axes[0].hist(y_proba_log[y_true_log == 0], bins=50, alpha=0.7, label="Class 0")
    axes[0].hist(y_proba_log[y_true_log == 1], bins=50, alpha=0.7, label="Class 1")
    axes[0].set_title("Logistic Regression")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    y_true_gbdt = hist_gbdt_result.y_true
    y_proba_gbdt = hist_gbdt_result.y_proba

    axes[1].hist(y_proba_gbdt[y_true_gbdt == 0], bins=50, alpha=0.7, label="Class 0")
    axes[1].hist(y_proba_gbdt[y_true_gbdt == 1], bins=50, alpha=0.7, label="Class 1")
    axes[1].set_title("HistGradientBoosting")
    axes[1].set_xlabel("Predicted Probability")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig.suptitle("Score Distribution Comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix_comparison(
    logistic_result: EvaluationResult,
    hist_gbdt_result: EvaluationResult,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    plots = [
        (axes[0], logistic_result, "Logistic Regression"),
        (axes[1], hist_gbdt_result, "HistGradientBoosting"),
    ]

    for ax, result, title in plots:
        display = ConfusionMatrixDisplay(
            confusion_matrix=result.confusion_matrix,
            display_labels=[0, 1],
        )
        display.plot(
            ax=ax,
            values_format="d",
            colorbar=False,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

    fig.suptitle("Confusion Matrix Comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _validate_eval_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> None:
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D, got shape {y_true.shape}")
    if y_pred.ndim != 1:
        raise ValueError(f"y_pred must be 1D, got shape {y_pred.shape}")
    if y_proba.ndim != 1:
        raise ValueError(f"y_proba must be 1D, got shape {y_proba.shape}")

    n = y_true.shape[0]
    if y_pred.shape[0] != n or y_proba.shape[0] != n:
        raise ValueError(
            f"Input length mismatch: len(y_true)={n}, "
            f"len(y_pred)={y_pred.shape[0]}, len(y_proba)={y_proba.shape[0]}"
        )
