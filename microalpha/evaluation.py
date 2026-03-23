from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
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
    cm = confusion_matrix(y_true, y_pred)

    return EvaluationResult(
        model_name=model_name,
        accuracy=acc,
        roc_auc=auc,
        confusion_matrix=cm,
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


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_path: Path,
    model_name: str,
) -> None:
    """
    Save ROC curve plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_score_distribution(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_path: Path,
    model_name: str,
) -> None:
    """
    Save predicted score distributions split by true class.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(y_proba[y_true == 0], bins=100, alpha=0.7, label="True class = 0")
    plt.hist(y_proba[y_true == 1], bins=100, alpha=0.7, label="True class = 1")
    plt.xlabel("Predicted probability of up move")
    plt.ylabel("Count")
    plt.title(f"Score Distribution - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    out_path: Path,
    model_name: str,
) -> None:
    """
    Save confusion matrix heatmap using matplotlib only.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, aspect="auto")
    plt.colorbar(label="Count")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title(f"Confusion Matrix - {model_name}")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


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
            f"Input length mismatch: len(y_true)={n}, len(y_pred)={y_pred.shape[0]}, len(y_proba)={y_proba.shape[0]}"
        )


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
    logistic_auc = [float(row["logistic"]["roc_auc"]) for row in pooled_ticker_metrics["tickers"]]
    hist_auc = [float(row["hist_gbdt"]["roc_auc"]) for row in pooled_ticker_metrics["tickers"]]

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