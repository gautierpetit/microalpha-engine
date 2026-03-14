#include "microalpha/features.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <sstream>
#include <format>
#include <stdexcept>
#include <vector>

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

}  // namespace

py::array_t<double> py_compute_features_series(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& bid_prices,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& bid_sizes,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& ask_prices,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& ask_sizes
) {
    validate_2d_array(bid_prices, "bid_prices");
    validate_2d_array(bid_sizes, "bid_sizes");
    validate_2d_array(ask_prices, "ask_prices");
    validate_2d_array(ask_sizes, "ask_sizes");

    validate_same_shape(bid_prices, bid_sizes, "bid_prices", "bid_sizes");
    validate_same_shape(bid_prices, ask_prices, "bid_prices", "ask_prices");
    validate_same_shape(bid_prices, ask_sizes, "bid_prices", "ask_sizes");

    const std::size_t n_rows = static_cast<std::size_t>(bid_prices.shape(0));
    const std::size_t levels = static_cast<std::size_t>(bid_prices.shape(1));

    // Input arrays are guaranteed contiguous by py::array::c_style |forcecast,
    // so we can pass raw data pointers safely.
    const double* bid_prices_ptr = static_cast<const double*>(bid_prices.data());
    const double* bid_sizes_ptr = static_cast<const double*>(bid_sizes.data());
    const double* ask_prices_ptr = static_cast<const double*>(ask_prices.data());
    const double* ask_sizes_ptr = static_cast<const double*>(ask_sizes.data());

    microalpha::FeatureMatrix result = microalpha::compute_features_series(
        bid_prices_ptr,
        bid_sizes_ptr,
        ask_prices_ptr,
        ask_sizes_ptr,
        n_rows,
        levels
    );

    py::array_t<double> out({result.n_rows, result.n_cols});
    double* out_ptr = static_cast<double*>(out.mutable_data());
    // result.data is already a flat row-major buffer compatible with NumPy
    // so we can copy the entire block directly
    std::copy(
        result.data.begin(),
        result.data.end(),
        out_ptr
    );

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
        "Compute event-level microstructure features from order book arrays."
    );
}