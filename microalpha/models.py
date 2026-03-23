from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from pathlib import Path

import joblib
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ModelName = Literal["logistic", "hist_gbdt"]


@dataclass(frozen=True)
class SplitResult:
    """
    Container for time-ordered train/test split.
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    split_idx: int
    train_fraction: float
    n_train: int
    n_test: int


def time_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_fraction: float = 0.7,
) -> SplitResult:
    """
    Perform a strict time-ordered train/test split.

    No shuffling is applied.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (N, F).
    y : np.ndarray
        Label array of shape (N,).
    train_fraction : float
        Fraction of observations assigned to the training set.

    Returns
    -------
    SplitResult
        Structured split output.
    """
    _validate_X_y(X, y)

    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    n = X.shape[0]
    split_idx = int(n * train_fraction)

    if split_idx <= 0 or split_idx >= n:
        raise ValueError(
            f"Invalid split_idx={split_idx} for n={n}. Check train_fraction={train_fraction}."
        )

    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    return SplitResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        split_idx=split_idx,
        train_fraction=train_fraction,
        n_train=X_train.shape[0],
        n_test=X_test.shape[0],
    )


def build_logistic_model(
    random_state: int = 42,
) -> Pipeline:
    """
    Build standardized logistic regression pipeline.

    Notes
    -----
    - Standardization is mandatory due to large feature scale differences.
    - class_weight='balanced' is NOT used because class imbalance is mild.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    l1_ratio=0,  # L2 penalty only
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_hist_gbdt_model(
    learning_rate: float = 0.05,
    max_iter: int = 200,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 50,
    l2_regularization: float = 0.0,
    random_state: int = 42,
):
    """
    Build standardized HistGradientBoostingClassifier.
    """
    # HistGradientBoosting does not need feature scaling
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_state,
    )


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: ModelName,
    *,
    logistic_random_state: int = 42,
    hist_gbdt_random_state: int = 42,
    hist_gbdt_learning_rate: float = 0.05,
    hist_gbdt_max_iter: int = 200,
    hist_gbdt_max_leaf_nodes: int = 31,
    hist_gbdt_min_samples_leaf: int = 50,
    hist_gbdt_l2_regularization: float = 0.0,
):
    """
    Train a model by name.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    model_name : {"logistic", "hist_gbdt"}
        Model selection.
    Returns
    -------
    Trained model instance.
    """
    _validate_X_y(X_train, y_train)

    if model_name == "logistic":
        model = build_logistic_model(random_state=logistic_random_state)

    elif model_name == "hist_gbdt":
        model = build_hist_gbdt_model(
            learning_rate=hist_gbdt_learning_rate,
            max_iter=hist_gbdt_max_iter,
            max_leaf_nodes=hist_gbdt_max_leaf_nodes,
            min_samples_leaf=hist_gbdt_min_samples_leaf,
            l2_regularization=hist_gbdt_l2_regularization,
            random_state=hist_gbdt_random_state,
        )

    else:
        raise ValueError(
            f"Unsupported model_name={model_name!r}. Expected 'logistic' or 'hist_gbdt'."
        )

    model.fit(X_train, y_train)
    return model


def predict_probabilities(model, X: np.ndarray) -> np.ndarray:
    """
    Return positive-class probabilities for binary classification.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    proba = model.predict_proba(X)

    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ValueError(
            f"Expected binary-class probability matrix of shape (N, 2), got {proba.shape}"
        )

    return proba[:, 1]


def predict_classes(model, X: np.ndarray) -> np.ndarray:
    """
    Return predicted class labels.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    return model.predict(X)


def get_logistic_coefficients(
    model: Pipeline,
    feature_names: list[str],
) -> list[tuple[str, float]]:
    """
    Extract logistic regression coefficients from the trained pipeline.

    Returns
    -------
    list of (feature_name, coefficient)
        Sorted by absolute coefficient magnitude descending.
    """
    clf = model.named_steps["clf"]

    if not isinstance(clf, LogisticRegression):
        raise TypeError("Model is not a logistic regression pipeline.")

    coef = clf.coef_
    if coef.shape[0] != 1:
        raise ValueError(
            f"Expected binary logistic coefficients of shape (1, F), got {coef.shape}"
        )

    coef_1d = coef[0]

    if len(feature_names) != coef_1d.shape[0]:
        raise ValueError(
            f"feature_names length {len(feature_names)} does not match coefficient length {coef_1d.shape[0]}"
        )

    pairs = list(zip(feature_names, coef_1d.tolist()))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs

def save_trained_model(model, out_path: str | Path) -> None:
    """
    Persist a trained sklearn-compatible model to disk with joblib.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)


def compute_permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    scoring: str = "roc_auc",
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
) -> list[dict[str, float | str]]:
    """
    Compute permutation feature importance on a held-out dataset.

    Returns
    -------
    list[dict]
        Sorted descending by mean importance. Each item contains:
        - feature
        - importance_mean
        - importance_std
    """
    _validate_X_y(X, y)

    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"feature_names length {len(feature_names)} does not match X.shape[1]={X.shape[1]}"
        )

    result = permutation_importance(
        estimator=model,
        X=X,
        y=y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    rows = [
        {
            "feature": feature_names[i],
            "importance_mean": float(result.importances_mean[i]),
            "importance_std": float(result.importances_std[i]),
        }
        for i in range(len(feature_names))
    ]
    rows.sort(key=lambda row: row["importance_mean"], reverse=True)
    return rows


def summarize_split(split: SplitResult) -> dict[str, float | int]:
    """
    Small summary dictionary for logging/debugging.
    """
    return {
        "split_idx": split.split_idx,
        "train_fraction": split.train_fraction,
        "n_train": split.n_train,
        "n_test": split.n_test,
        "train_pos_rate": float(np.mean(split.y_train)),
        "test_pos_rate": float(np.mean(split.y_test)),
    }


def _validate_X_y(X: np.ndarray, y: np.ndarray) -> None:
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array")
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have same number of rows, got {X.shape[0]} and {y.shape[0]}"
        )
    if X.shape[0] == 0:
        raise ValueError("X and y must be non-empty")
