#pragma once

#include <cstddef>
#include <vector>

namespace microalpha {

struct FeatureMatrix {
    std::size_t n_rows{};
    std::size_t n_cols{};
    std::vector<double> data;  // row-major, shape (n_rows, n_cols)
};

/**
 * Compute event-level microstructure features from aligned order book arrays.
 *
 * Inputs are flattened row-major arrays of shape (n_rows, levels):
 * - bid_prices
 * - bid_sizes
 * - ask_prices
 * - ask_sizes
 *
 * Output columns:
 *   0 -> ofi_best
 *   1 -> ofi_best_norm
 *   2 -> queue_imbalance_best
 *   3 -> depth_imbalance_3
 *   4 -> depth_imbalance_5
 *   5 -> depth_imbalance_10
 *   6 -> spread
 *   7 -> microprice_deviation
 */
FeatureMatrix compute_features_series(
    const double* bid_prices,
    const double* bid_sizes,
    const double* ask_prices,
    const double* ask_sizes,
    std::size_t n_rows,
    std::size_t levels
);

}  // namespace microalpha