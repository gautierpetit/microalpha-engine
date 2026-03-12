from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


LabelMode = Literal["binary_drop_ties", "three_class"]


@dataclass(frozen=True)
class LabelResult:
    """
    Container for label outputs.

    Attributes
    ----------
    y : np.ndarray
        Final label array after applying the chosen label mode.
    delta : np.ndarray
        Forward midprice change over the horizon, before any filtering.
        Shape: (N - H,)
    valid_mask : np.ndarray
        Boolean mask applied to delta (and later to features) to produce y.
        For binary_drop_ties: delta != 0
        For three_class: all True
    horizon : int
        Event horizon used for labeling.
    label_mode : str
        Labeling mode used.
    tie_rate : float
        Fraction of zero deltas in the raw delta series.
    n_raw : int
        Number of raw label candidates, equal to N - H.
    n_final : int
        Number of final labels after applying valid_mask.
    """
    y: np.ndarray
    delta: np.ndarray
    valid_mask: np.ndarray
    horizon: int
    label_mode: str
    tie_rate: float
    n_raw: int
    n_final: int


def compute_forward_midprice_delta(midprice: np.ndarray, horizon: int) -> np.ndarray:
    """
    Compute forward midprice change over an event horizon H.

    Parameters
    ----------
    midprice : np.ndarray
        Midprice series of shape (N,).
    horizon : int
        Number of events ahead.

    Returns
    -------
    np.ndarray
        Forward delta array of shape (N - H,), where:
        delta[t] = midprice[t + H] - midprice[t]
    """
    _validate_midprice_and_horizon(midprice, horizon)

    m0 = midprice[:-horizon]
    m1 = midprice[horizon:]
    delta = m1 - m0
    return delta


def create_directional_labels(
    midprice: np.ndarray,
    horizon: int,
    label_mode: LabelMode = "binary_drop_ties",
) -> LabelResult:
    """
    Create directional labels from forward midprice changes.

    Modes
    -----
    binary_drop_ties:
        y = 1 if delta > 0
        y = 0 if delta < 0
        ties (delta == 0) are dropped

    three_class:
        y = 2 if delta > 0
        y = 1 if delta == 0
        y = 0 if delta < 0

    Parameters
    ----------
    midprice : np.ndarray
        Midprice series of shape (N,).
    horizon : int
        Number of events ahead.
    label_mode : {"binary_drop_ties", "three_class"}
        Labeling policy.

    Returns
    -------
    LabelResult
        Structured output including labels, raw deltas, mask, and diagnostics.
    """
    delta = compute_forward_midprice_delta(midprice, horizon)
    tie_mask = delta == 0.0
    tie_rate = float(np.mean(tie_mask))
    n_raw = int(delta.shape[0])

    if label_mode == "binary_drop_ties":
        valid_mask = ~tie_mask
        y = (delta[valid_mask] > 0).astype(np.int8)

    elif label_mode == "three_class":
        valid_mask = np.ones_like(delta, dtype=bool)
        y = np.empty_like(delta, dtype=np.int8)
        y[delta < 0] = 0
        y[delta == 0] = 1
        y[delta > 0] = 2

    else:
        raise ValueError(
            f"Unsupported label_mode={label_mode!r}. "
            "Expected 'binary_drop_ties' or 'three_class'."
        )

    n_final = int(y.shape[0])

    return LabelResult(
        y=y,
        delta=delta,
        valid_mask=valid_mask,
        horizon=horizon,
        label_mode=label_mode,
        tie_rate=tie_rate,
        n_raw=n_raw,
        n_final=n_final,
    )


def align_features_with_labels(
    features: np.ndarray,
    label_result: LabelResult,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align raw feature matrix with labels.

    Expected raw feature shape: (N, F), where N matches the original midprice length.
    Since labels are built from delta over horizon H, only the first (N - H) rows
    can be used. Then valid_mask is applied.

    Parameters
    ----------
    features : np.ndarray
        Raw feature matrix of shape (N, F).
    label_result : LabelResult
        Result returned by create_directional_labels().

    Returns
    -------
    X : np.ndarray
        Aligned feature matrix.
    y : np.ndarray
        Final labels.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")

    n_expected = label_result.n_raw + label_result.horizon
    if features.shape[0] != n_expected:
        raise ValueError(
            f"Feature row count mismatch: expected {n_expected}, got {features.shape[0]}"
        )

    X_raw = features[:-label_result.horizon]
    X = X_raw[label_result.valid_mask]
    y = label_result.y

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Aligned feature/label mismatch: X has {X.shape[0]} rows, y has {y.shape[0]}"
        )

    return X, y


def summarize_labels(label_result: LabelResult) -> dict[str, float | int | str]:
    """
    Return a compact summary dictionary for reporting or metadata logging.
    """
    summary: dict[str, float | int | str] = {
        "horizon": label_result.horizon,
        "label_mode": label_result.label_mode,
        "tie_rate": label_result.tie_rate,
        "n_raw": label_result.n_raw,
        "n_final": label_result.n_final,
    }

    classes, counts = np.unique(label_result.y, return_counts=True)
    for c, cnt in zip(classes.tolist(), counts.tolist()):
        summary[f"class_{c}_count"] = int(cnt)
        summary[f"class_{c}_pct"] = float(cnt) / float(label_result.n_final)

    return summary


def _validate_midprice_and_horizon(midprice: np.ndarray, horizon: int) -> None:
    if not isinstance(midprice, np.ndarray):
        raise TypeError("midprice must be a numpy array")
    if midprice.ndim != 1:
        raise ValueError(f"midprice must be 1D, got shape {midprice.shape}")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if midprice.shape[0] <= horizon:
        raise ValueError(
            f"midprice length must be greater than horizon; got len={midprice.shape[0]}, horizon={horizon}"
        )