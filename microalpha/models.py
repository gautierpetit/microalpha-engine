from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ModelName = Literal["logistic", "mlp"]


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
                    l1_ratio=0, # L2 penalty only
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_mlp_model(
    hidden_layer_sizes: tuple[int, ...] = (32,),
    max_iter: int = 200,
    alpha: float = 1e-4,
    learning_rate_init: float = 1e-3,
    batch_size: int = 256,
    random_state: int = 42,
) -> Pipeline:
    """
    Build standardized small MLP pipeline.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    batch_size=batch_size,
                    learning_rate_init=learning_rate_init,
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: ModelName,
    *,
    logistic_random_state: int = 42,
    mlp_random_state: int = 42,
    mlp_hidden_layer_sizes: tuple[int, ...] = (32,),
    mlp_max_iter: int = 200,
    mlp_alpha: float = 1e-4,
    mlp_learning_rate_init: float = 1e-3,
    mlp_batch_size: int = 256,
):
    """
    Train a model by name.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    model_name : {"logistic", "mlp"}
        Model selection.

    Returns
    -------
    fitted model
        sklearn Pipeline object.
    """
    _validate_X_y(X_train, y_train)

    if model_name == "logistic":
        model = build_logistic_model(random_state=logistic_random_state)

    elif model_name == "mlp":
        model = build_mlp_model(
            hidden_layer_sizes=mlp_hidden_layer_sizes,
            max_iter=mlp_max_iter,
            alpha=mlp_alpha,
            learning_rate_init=mlp_learning_rate_init,
            batch_size=mlp_batch_size,
            random_state=mlp_random_state,
        )

    else:
        raise ValueError(
            f"Unsupported model_name={model_name!r}. Expected 'logistic' or 'mlp'."
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
        raise ValueError(f"Expected binary logistic coefficients of shape (1, F), got {coef.shape}")

    coef_1d = coef[0]

    if len(feature_names) != coef_1d.shape[0]:
        raise ValueError(
            f"feature_names length {len(feature_names)} does not match coefficient length {coef_1d.shape[0]}"
        )

    pairs = list(zip(feature_names, coef_1d.tolist()))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs


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
        raise ValueError(f"X and y must have same number of rows, got {X.shape[0]} and {y.shape[0]}")
    if X.shape[0] == 0:
        raise ValueError("X and y must be non-empty")