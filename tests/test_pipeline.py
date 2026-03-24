import numpy as np

from microalpha.pipeline import (
    TickerDataset,
    iter_ticker_test_segments,
    split_and_pool_datasets,
)


def test_split_and_pool_datasets_preserves_per_ticker_test_segments() -> None:
    ds1 = TickerDataset(
        symbol="AAA",
        X=np.arange(20).reshape(10, 2),
        y=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8),
        n_events=10,
        label_summary={"tie_rate": 0.0, "move_rate": 1.0},
    )
    ds2 = TickerDataset(
        symbol="BBB",
        X=np.arange(20, 40).reshape(10, 2),
        y=np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1], dtype=np.int8),
        n_events=10,
        label_summary={"tie_rate": 0.0, "move_rate": 1.0},
    )

    pooled_split, ticker_splits = split_and_pool_datasets(
        [ds1, ds2],
        train_fraction=0.7,
    )

    assert pooled_split.n_train == 14
    assert pooled_split.n_test == 6

    # Reconstruct test segments from pooled arrays
    slices = list(iter_ticker_test_segments(ticker_splits))
    assert len(slices) == 2

    (ts1, sl1), (ts2, sl2) = slices
    assert ts1.symbol == "AAA"
    assert ts2.symbol == "BBB"

    np.testing.assert_array_equal(pooled_split.X_test[sl1], ts1.split.X_test)
    np.testing.assert_array_equal(pooled_split.y_test[sl1], ts1.split.y_test)

    np.testing.assert_array_equal(pooled_split.X_test[sl2], ts2.split.X_test)
    np.testing.assert_array_equal(pooled_split.y_test[sl2], ts2.split.y_test)
