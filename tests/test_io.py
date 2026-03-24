from pathlib import Path

import pandas as pd
import pytest

from microalpha.config import TickerConfig
from microalpha.io import load_lobster


def test_load_lobster_smoke(tmp_path: Path) -> None:
    message_path = tmp_path / "message.csv"
    orderbook_path = tmp_path / "orderbook.csv"

    message_df = pd.DataFrame(
        [
            [34200.0, 1, 100, 10, 1, 1],
            [34200.1, 1, 101, 20, 1, 1],
            [34200.2, 1, 102, 30, 1, 1],
        ]
    )
    message_df.to_csv(message_path, header=False, index=False)

    # 1 level => 4 columns: ask_p, ask_q, bid_p, bid_q
    orderbook_df = pd.DataFrame(
        [
            [10001, 10, 9999, 12],
            [10002, 11, 10000, 13],
            [10003, 12, 10001, 14],
        ]
    )
    orderbook_df.to_csv(orderbook_path, header=False, index=False)

    ticker_cfg = TickerConfig(
        symbol="TEST",
        message_csv=message_path,
        orderbook_csv=orderbook_path,
    )

    data = load_lobster(
        ticker_cfg,
        levels=1,
        price_scale=10000,
        validate=True,
    )

    assert data.t.shape == (3,)
    assert data.bid_prices.shape == (3, 1)
    assert data.ask_prices.shape == (3, 1)
    assert data.bid_sizes.shape == (3, 1)
    assert data.ask_sizes.shape == (3, 1)
    assert data.midprice.shape == (3,)
    assert data.midprice[0] == (0.9999 + 1.0001) / 2


def test_load_lobster_raises_on_bad_column_count(tmp_path: Path) -> None:
    message_path = tmp_path / "message.csv"
    orderbook_path = tmp_path / "orderbook.csv"

    pd.DataFrame([[0.0], [1.0]]).to_csv(message_path, header=False, index=False)
    pd.DataFrame([[1, 2, 3], [4, 5, 6]]).to_csv(
        orderbook_path, header=False, index=False
    )

    ticker_cfg = TickerConfig(
        symbol="TEST",
        message_csv=message_path,
        orderbook_csv=orderbook_path,
    )

    with pytest.raises(ValueError, match="Expected 4 order book columns"):
        load_lobster(
            ticker_cfg,
            levels=1,
            price_scale=10000,
            validate=False,
        )
