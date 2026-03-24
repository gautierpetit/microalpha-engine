import numpy as np

from microalpha.labels import (
    align_features_with_labels,
    compute_forward_midprice_delta,
    create_directional_labels,
    create_movement_labels,
)


def test_compute_forward_midprice_delta() -> None:
    midprice = np.array([100.0, 101.0, 100.5, 102.0, 101.0])
    delta = compute_forward_midprice_delta(midprice, horizon=2)

    expected = np.array([0.5, 1.0, 0.5])
    np.testing.assert_allclose(delta, expected)


def test_directional_labels_drop_ties_and_align_features() -> None:
    midprice = np.array([100.0, 100.0, 101.0, 101.0, 102.0], dtype=float)
    # horizon=1 => deltas = [0, 1, 0, 1]
    label_result = create_directional_labels(
        midprice,
        horizon=1,
        label_mode="binary_drop_ties",
    )

    assert label_result.n_raw == 4
    assert label_result.n_final == 2
    assert label_result.tie_rate == 0.5
    np.testing.assert_array_equal(label_result.y, np.array([1, 1], dtype=np.int8))

    features = np.arange(5 * 3).reshape(5, 3)
    X, y = align_features_with_labels(features, label_result)

    # Features trimmed to first 4 rows, then keep rows 1 and 3
    expected_X = np.vstack([features[1], features[3]])
    np.testing.assert_array_equal(X, expected_X)
    np.testing.assert_array_equal(y, np.array([1, 1], dtype=np.int8))


def test_directional_labels_keep_ties_as_zero() -> None:
    midprice = np.array([100.0, 100.0, 101.0, 101.0, 102.0], dtype=float)
    label_result = create_directional_labels(
        midprice,
        horizon=1,
        label_mode="binary_keep_ties_as_zero",
    )

    assert label_result.n_raw == 4
    assert label_result.n_final == 4
    assert label_result.tie_rate == 0.5
    np.testing.assert_array_equal(label_result.y, np.array([0, 1, 0, 1], dtype=np.int8))


def test_movement_labels_keep_all_rows() -> None:
    midprice = np.array([100.0, 100.0, 101.0, 101.0, 102.0], dtype=float)
    label_result = create_movement_labels(
        midprice,
        horizon=1,
        label_mode="binary",
    )

    assert label_result.n_raw == 4
    assert label_result.n_final == 4
    assert label_result.tie_rate == 0.5
    np.testing.assert_array_equal(label_result.y, np.array([0, 1, 0, 1], dtype=np.int8))
