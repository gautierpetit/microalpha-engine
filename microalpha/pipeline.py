from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

import numpy as np

from microalpha.config import ExperimentConfig, TickerConfig
from microalpha.features import compute_features
from microalpha.io import LobsterPaths, load_lobster
from microalpha.labels import (
    align_features_with_labels,
    create_labels,
    summarize_labels,
)
from microalpha.models import SplitResult, summarize_split, time_train_test_split


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
) -> TickerDataset:
    data = load_lobster(
        LobsterPaths(
            message_csv=ticker_cfg.message_csv,
            orderbook_csv=ticker_cfg.orderbook_csv,
        ),
        levels=cfg.dataset.levels,
        price_scale=cfg.dataset.price_scale,
        validate=True,
    )

    X_raw = compute_features(
        bid_prices=data.bid_prices,
        bid_sizes=data.bid_sizes,
        ask_prices=data.ask_prices,
        ask_sizes=data.ask_sizes,
        midprice=data.midprice,
        timestamps=data.t,
        cfg=cfg.features,
    )

    label_result = create_labels(
        midprice=data.midprice,
        horizon=cfg.labels.horizon,
        task_name=cfg.task.name,
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
        split = time_train_test_split(
            dataset.X, dataset.y, train_fraction=train_fraction
        )
        ticker_splits.append(
            TickerSplit(
                symbol=dataset.symbol,
                split=split,
                n_events=dataset.n_events,
                label_summary=dataset.label_summary,
            )
        )

    pooled_split = SplitResult(
        X_train=np.vstack([ts.split.X_train for ts in ticker_splits]),
        X_test=np.vstack([ts.split.X_test for ts in ticker_splits]),
        y_train=np.concatenate([ts.split.y_train for ts in ticker_splits]),
        y_test=np.concatenate([ts.split.y_test for ts in ticker_splits]),
        split_idx=-1,
        train_fraction=train_fraction,
        n_train=sum(ts.split.n_train for ts in ticker_splits),
        n_test=sum(ts.split.n_test for ts in ticker_splits),
    )

    return pooled_split, ticker_splits


def make_ticker_split_summary(
    ticker_split: TickerSplit,
) -> dict[str, float | int | str]:
    return {
        "symbol": ticker_split.symbol,
        "n_events": ticker_split.n_events,
        **ticker_split.label_summary,
        **summarize_split(ticker_split.split),
    }


def iter_ticker_test_segments(
    ticker_splits: list[TickerSplit],
) -> Iterator[tuple[TickerSplit, slice]]:
    """
    Yield (ticker_split, slice) pairs mapping each ticker to its segment inside
    the pooled test arrays.

    Assumes pooled X_test / y_test were built by concatenating ticker test sets
    in the same order as `ticker_splits`, which is how split_and_pool_datasets() behaves.
    """
    start = 0
    for ticker_split in ticker_splits:
        stop = start + ticker_split.split.n_test
        yield ticker_split, slice(start, stop)
        start = stop
