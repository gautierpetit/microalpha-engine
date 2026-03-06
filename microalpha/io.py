from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LobsterPaths:
    message_csv: Path
    orderbook_csv: Path


@dataclass(frozen=True)
class LobsterConfig:
    levels: int = 10
    price_scale: int = 10_000  # LOBSTER prices are typically price * 10,000
    dtype_float: str = "float64"


@dataclass(frozen=True)
class LobsterData:
    # Raw aligned event-time series
    t: np.ndarray  # shape (N,), seconds from midnight
    event_type: np.ndarray  # shape (N,), int
    order_id: np.ndarray  # shape (N,), int
    event_size: np.ndarray  # shape (N,), int
    event_price: np.ndarray  # shape (N,), float
    event_direction: np.ndarray  # shape (N,), int (typically 1 buy / -1 sell)

    bid_prices: np.ndarray  # shape (N, L)
    bid_sizes: np.ndarray  # shape (N, L)
    ask_prices: np.ndarray  # shape (N, L)
    ask_sizes: np.ndarray  # shape (N, L)

    midprice: np.ndarray  # shape (N,)
    spread: np.ndarray  # shape (N,)


def _read_lobster_message_csv(path: Path) -> pd.DataFrame:
    """
    LOBSTER message file columns (standard):
      1) Time (seconds after midnight)
      2) Type
      3) Order ID
      4) Size
      5) Price (price * 10,000)
      6) Direction (1=buy, -1=sell)  [for trades; still present in file]
    """
    df = pd.read_csv(path, header=None)
    if df.shape[1] != 6:
        raise ValueError(f"Expected 6 columns in message CSV, got {df.shape[1]} at {path}")
    df.columns = ["time", "type", "order_id", "size", "price", "direction"]
    return df


def _read_lobster_orderbook_csv(path: Path, levels: int) -> pd.DataFrame:
    """
    LOBSTER orderbook file columns (standard):
      For each level i=1..L:
        AskPrice_i, AskSize_i, BidPrice_i, BidSize_i
    Total columns = 4*L
    """
    df = pd.read_csv(path, header=None)
    expected_cols = 4 * levels
    if df.shape[1] != expected_cols:
        raise ValueError(
            f"Expected {expected_cols} columns (4*levels) in orderbook CSV, got {df.shape[1]} at {path}"
        )

    cols = []
    for i in range(1, levels + 1):
        cols += [f"ask_price_{i}", f"ask_size_{i}", f"bid_price_{i}", f"bid_size_{i}"]
    df.columns = cols
    return df


def load_lobster(
    paths: LobsterPaths,
    cfg: Optional[LobsterConfig] = None,
    nrows: Optional[int] = None,
    validate: bool = True,
) -> LobsterData:
    """
    Load and validate aligned LOBSTER message + orderbook CSVs.
    Returns numpy arrays suitable for C++ feature engine.
    """
    cfg = cfg or LobsterConfig()

    msg = _read_lobster_message_csv(paths.message_csv)
    ob = _read_lobster_orderbook_csv(paths.orderbook_csv, cfg.levels)

    if nrows is not None:
        msg = msg.iloc[:nrows].copy()
        ob = ob.iloc[:nrows].copy()

    if validate:
        if len(msg) != len(ob):
            raise ValueError(f"Row mismatch: message rows={len(msg)} orderbook rows={len(ob)}")

        # time monotonic (non-decreasing is fine, but should not go backwards)
        t = msg["time"].to_numpy()
        if np.any(np.diff(t) < 0):
            idx = int(np.where(np.diff(t) < 0)[0][0])
            raise ValueError(f"Timestamps not monotonic at index {idx}: {t[idx]} -> {t[idx+1]}")

    # Convert orderbook to arrays
    L = cfg.levels
    # Ask/Bid arrays in shape (N, L)
    ask_prices = np.column_stack([ob[f"ask_price_{i}"].to_numpy() for i in range(1, L + 1)]).astype(cfg.dtype_float)
    ask_sizes = np.column_stack([ob[f"ask_size_{i}"].to_numpy() for i in range(1, L + 1)]).astype(cfg.dtype_float)
    bid_prices = np.column_stack([ob[f"bid_price_{i}"].to_numpy() for i in range(1, L + 1)]).astype(cfg.dtype_float)
    bid_sizes = np.column_stack([ob[f"bid_size_{i}"].to_numpy() for i in range(1, L + 1)]).astype(cfg.dtype_float)

    # Scale prices
    ask_prices = ask_prices / cfg.price_scale
    bid_prices = bid_prices / cfg.price_scale

    # Message fields
    t = msg["time"].to_numpy(dtype=cfg.dtype_float)
    event_type = msg["type"].to_numpy(dtype=np.int32)
    order_id = msg["order_id"].to_numpy(dtype=np.int64)
    event_size = msg["size"].to_numpy(dtype=np.int32)
    event_price = msg["price"].to_numpy(dtype=cfg.dtype_float) / cfg.price_scale
    event_direction = msg["direction"].to_numpy(dtype=np.int32)

    # Midprice and spread from best level
    bid1 = bid_prices[:, 0]
    ask1 = ask_prices[:, 0]
    midprice = 0.5 * (bid1 + ask1)
    spread = ask1 - bid1

    if validate:
        # Spread sanity
        if np.any(spread < -1e-12):
            bad = np.where(spread < -1e-12)[0][:5]
            raise ValueError(f"Negative spread detected at indices {bad.tolist()} (check price scaling / parsing).")

        # Best bid should be <= best ask always
        if np.any(bid1 > ask1 + 1e-12):
            bad = np.where(bid1 > ask1 + 1e-12)[0][:5]
            raise ValueError(f"bid1 > ask1 at indices {bad.tolist()} (parsing error).")

        # Level monotonicity: bids should be non-increasing in price, asks non-decreasing
        # (allow equality due to tick rounding / empty depth)
        if np.any(np.diff(bid_prices, axis=1) > 1e-12):
            # bid_prices[:,0] >= bid_prices[:,1] >= ...
            bad = np.where(np.diff(bid_prices, axis=1) > 1e-12)[0][:5]
            raise ValueError(f"Bid levels not non-increasing at some rows (examples: {bad.tolist()}).")

        if np.any(np.diff(ask_prices, axis=1) < -1e-12):
            # ask_prices[:,0] <= ask_prices[:,1] <= ...
            bad = np.where(np.diff(ask_prices, axis=1) < -1e-12)[0][:5]
            raise ValueError(f"Ask levels not non-decreasing at some rows (examples: {bad.tolist()}).")

    return LobsterData(
        t=t,
        event_type=event_type,
        order_id=order_id,
        event_size=event_size,
        event_price=event_price,
        event_direction=event_direction,
        bid_prices=bid_prices,
        bid_sizes=bid_sizes,
        ask_prices=ask_prices,
        ask_sizes=ask_sizes,
        midprice=midprice,
        spread=spread,
    )


def compute_tie_rate(midprice: np.ndarray, horizon: int) -> Tuple[float, int]:
    """
    Tie rate for delta midprice over event horizon H:
      p_tie = Pr(m_{t+H} == m_t)
    Returns (p_tie, n_effective) where n_effective = len(midprice) - H.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if len(midprice) <= horizon:
        raise ValueError("midprice length must be > horizon")

    m0 = midprice[:-horizon]
    m1 = midprice[horizon:]
    delta = m1 - m0

    # Midprice is float; compare with exact equality is usually ok here because prices are tick-based
    ties = np.sum(delta == 0.0)
    n = delta.shape[0]
    return float(ties) / float(n), int(n)


def time_stats(t: np.ndarray) -> Dict[str, float]:
    """
    Basic stats for event timestamps.
    """
    dt = np.diff(t)
    return {
        "n_events": float(t.shape[0]),
        "t_start": float(t[0]),
        "t_end": float(t[-1]),
        "duration_sec": float(t[-1] - t[0]),
        "dt_median": float(np.median(dt)),
        "dt_p95": float(np.percentile(dt, 95)),
        "dt_p99": float(np.percentile(dt, 99)),
    }