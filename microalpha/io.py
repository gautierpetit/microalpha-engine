from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from microalpha.config import TickerConfig


@dataclass(frozen=True)
class LobsterData:
    t: np.ndarray
    bid_prices: np.ndarray
    bid_sizes: np.ndarray
    ask_prices: np.ndarray
    ask_sizes: np.ndarray
    midprice: np.ndarray


def load_lobster(
    paths: TickerConfig,
    *,
    levels: int,
    price_scale: int,
    validate: bool = True,
) -> LobsterData:
    message_df = pd.read_csv(paths.message_csv, header=None)
    orderbook_df = pd.read_csv(paths.orderbook_csv, header=None)

    expected_orderbook_cols = 4 * levels
    if orderbook_df.shape[1] != expected_orderbook_cols:
        raise ValueError(
            f"Expected {expected_orderbook_cols} order book columns for {levels} "
            f"levels, got {orderbook_df.shape[1]}"
        )

    if len(message_df) != len(orderbook_df):
        raise ValueError(
            f"Message and order book row count mismatch: "
            f"{len(message_df)} vs {len(orderbook_df)}"
        )

    t = message_df.iloc[:, 0].to_numpy(dtype=np.float64)

    orderbook = orderbook_df.to_numpy(dtype=np.float64)

    ask_prices = orderbook[:, 0::4] / price_scale
    ask_sizes = orderbook[:, 1::4]
    bid_prices = orderbook[:, 2::4] / price_scale
    bid_sizes = orderbook[:, 3::4]

    midprice = 0.5 * (bid_prices[:, 0] + ask_prices[:, 0])

    if validate:
        if not np.all(np.isfinite(midprice)):
            raise ValueError("Non-finite midprice values detected")
        if not np.all(ask_prices[:, 0] >= bid_prices[:, 0]):
            raise ValueError("Best ask below best bid detected")

    return LobsterData(
        t=t,
        bid_prices=bid_prices,
        bid_sizes=bid_sizes,
        ask_prices=ask_prices,
        ask_sizes=ask_sizes,
        midprice=midprice,
    )
