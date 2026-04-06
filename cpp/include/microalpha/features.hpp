#pragma once

#include <cstddef>
#include <vector>

namespace microalpha {

struct FeatureMatrix {
    std::size_t n_rows;
    std::size_t n_cols;
    std::vector<double> data;
};

FeatureMatrix compute_features_series(
    const double* bid_prices,
    const double* bid_sizes,
    const double* ask_prices,
    const double* ask_sizes,
    const double* midprice,
    const double* timestamps,
    std::size_t n_rows,
    std::size_t levels,
    std::size_t ofi_window_raw,
    std::size_t ofi_norm_window_1,
    std::size_t ofi_norm_window_2,
    std::size_t ofi_norm_window_3,
    std::size_t vol_window,
    double intensity_window_seconds
);

}  // namespace microalpha