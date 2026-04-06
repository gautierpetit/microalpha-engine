#include "microalpha/features.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <format>
#include <stdexcept>

namespace py = pybind11;

namespace {

void validate_2d_array(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& arr,
    const char* name
) {
    if (arr.ndim() != 2) {
        throw std::invalid_argument(
            std::format("{} must be 2D, got ndim={}", name, arr.ndim())
        );
    }
}

void validate_1d_array(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& arr,
    const char* name
) {
    if (arr.ndim() != 1) {
        throw std::invalid_argument(
            std::format("{} must be 1D, got ndim={}", name, arr.ndim())
        );
    }
}

void validate_same_shape(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& a,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& b,
    const char* a_name,
    const char* b_name
) {
    if (a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1)) {
        throw std::invalid_argument(
            std::format(
                "Shape mismatch: {} has shape ({}, {}), while {} has shape ({}, {})",
                a_name, a.shape(0), a.shape(1),
                b_name, b.shape(0), b.shape(1)
            )
        );
    }
}

void validate_same_length(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& arr,
    std::size_t expected_len,
    const char* name
) {
    if (static_cast<std::size_t>(arr.shape(0)) != expected_len) {
        throw std::invalid_argument(
            std::format(
                "{} length mismatch: expected {}, got {}",
                name,
                expected_len,
                arr.shape(0)
            )
        );
    }
}

}  // namespace

py::array_t<double> py_compute_features_series(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& bid_prices,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& bid_sizes,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& ask_prices,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& ask_sizes,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& midprice,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& timestamps,
    const std::size_t ofi_window_raw,
    const std::size_t ofi_norm_window_1,
    const std::size_t ofi_norm_window_2,
    const std::size_t ofi_norm_window_3,
    const std::size_t vol_window,
    const double intensity_window_seconds
) {
    validate_2d_array(bid_prices, "bid_prices");
    validate_2d_array(bid_sizes, "bid_sizes");
    validate_2d_array(ask_prices, "ask_prices");
    validate_2d_array(ask_sizes, "ask_sizes");
    validate_1d_array(midprice, "midprice");
    validate_1d_array(timestamps, "timestamps");

    validate_same_shape(bid_prices, bid_sizes, "bid_prices", "bid_sizes");
    validate_same_shape(bid_prices, ask_prices, "bid_prices", "ask_prices");
    validate_same_shape(bid_prices, ask_sizes, "bid_prices", "ask_sizes");

    const std::size_t n_rows = static_cast<std::size_t>(bid_prices.shape(0));
    const std::size_t levels = static_cast<std::size_t>(bid_prices.shape(1));

    validate_same_length(midprice, n_rows, "midprice");
    validate_same_length(timestamps, n_rows, "timestamps");

    const double* bid_prices_ptr = static_cast<const double*>(bid_prices.data());
    const double* bid_sizes_ptr = static_cast<const double*>(bid_sizes.data());
    const double* ask_prices_ptr = static_cast<const double*>(ask_prices.data());
    const double* ask_sizes_ptr = static_cast<const double*>(ask_sizes.data());
    const double* midprice_ptr = static_cast<const double*>(midprice.data());
    const double* timestamps_ptr = static_cast<const double*>(timestamps.data());

    microalpha::FeatureMatrix result = microalpha::compute_features_series(
        bid_prices_ptr,
        bid_sizes_ptr,
        ask_prices_ptr,
        ask_sizes_ptr,
        midprice_ptr,
        timestamps_ptr,
        n_rows,
        levels,
        ofi_window_raw,
        ofi_norm_window_1,
        ofi_norm_window_2,
        ofi_norm_window_3,
        vol_window,
        intensity_window_seconds
    );

    py::array_t<double> out({result.n_rows, result.n_cols});
    double* out_ptr = static_cast<double*>(out.mutable_data());

    std::copy(result.data.begin(), result.data.end(), out_ptr);

    return out;
}

PYBIND11_MODULE(_cpp, m) {
    m.doc() = "C++ microstructure feature engine for microalpha";

    m.def(
        "compute_features_series",
        &py_compute_features_series,
        py::arg("bid_prices"),
        py::arg("bid_sizes"),
        py::arg("ask_prices"),
        py::arg("ask_sizes"),
        py::arg("midprice"),
        py::arg("timestamps"),
        py::arg("ofi_window_raw"),
        py::arg("ofi_norm_window_1"),
        py::arg("ofi_norm_window_2"),
        py::arg("ofi_norm_window_3"),
        py::arg("vol_window"),
        py::arg("intensity_window_seconds"),
        "Compute event-level microstructure features from order book arrays."
    );
}