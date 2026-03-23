#include "microalpha/features.hpp"

#include <cmath>
#include <stdexcept>

namespace microalpha {
namespace {

inline std::size_t idx(const std::size_t row, const std::size_t col, const std::size_t n_cols) {
    return row * n_cols + col;
}

inline double safe_divide(const double num, const double den) {
    if (std::abs(den) < 1e-15) {
        return 0.0;
    }
    return num / den;
}

inline double compute_best_level_ofi(
    const double* bid_prices,
    const double* bid_sizes,
    const double* ask_prices,
    const double* ask_sizes,
    const std::size_t t,
    const std::size_t levels
) {
    const double bid_p_prev = bid_prices[idx(t - 1, 0, levels)];
    const double bid_q_prev = bid_sizes[idx(t - 1, 0, levels)];
    const double ask_p_prev = ask_prices[idx(t - 1, 0, levels)];
    const double ask_q_prev = ask_sizes[idx(t - 1, 0, levels)];

    const double bid_p_curr = bid_prices[idx(t, 0, levels)];
    const double bid_q_curr = bid_sizes[idx(t, 0, levels)];
    const double ask_p_curr = ask_prices[idx(t, 0, levels)];
    const double ask_q_curr = ask_sizes[idx(t, 0, levels)];

    double delta_bid = 0.0;
    if (bid_p_curr > bid_p_prev) {
        delta_bid = bid_q_curr;
    } else if (bid_p_curr == bid_p_prev) {
        // Exact comparison is intentional:
        // LOBSTER prices originate from discrete integer ticks and scaling
        // preserves equality for this pipeline, so same-price events can be
        // treated exactly.
        delta_bid = bid_q_curr - bid_q_prev;
    } else {  // bid_p_curr < bid_p_prev
        delta_bid = -bid_q_prev;
    }

    double delta_ask = 0.0;
    if (ask_p_curr < ask_p_prev) {
        delta_ask = ask_q_curr;
    } else if (ask_p_curr == ask_p_prev) {
        // Same comment about exact comparison applies here as well.
        delta_ask = ask_q_curr - ask_q_prev;
    } else {  // ask_p_curr > ask_p_prev
        delta_ask = -ask_q_prev;
    }

    return delta_bid - delta_ask;
}

inline double compute_best_level_ofi_normalized(
    const double ofi_best,
    const double* bid_sizes,
    const double* ask_sizes,
    const std::size_t t,
    const std::size_t levels
) {
    const double bid_q = bid_sizes[idx(t, 0, levels)];
    const double ask_q = ask_sizes[idx(t, 0, levels)];
    return safe_divide(ofi_best, bid_q + ask_q);
}

inline double compute_queue_imbalance_best(
    const double* bid_sizes,
    const double* ask_sizes,
    const std::size_t t,
    const std::size_t levels
) {
    const double bid_q = bid_sizes[idx(t, 0, levels)];
    const double ask_q = ask_sizes[idx(t, 0, levels)];
    return safe_divide(bid_q, bid_q + ask_q);
}

inline double compute_depth_imbalance_k(
    const double* bid_sizes,
    const double* ask_sizes,
    const std::size_t t,
    const std::size_t levels,
    const std::size_t k
) {
    if (k == 0 || k > levels) {
        throw std::invalid_argument("Invalid depth imbalance level k");
    }

    double bid_sum = 0.0;
    double ask_sum = 0.0;

    for (std::size_t level = 0; level < k; ++level) {
        bid_sum += bid_sizes[idx(t, level, levels)];
        ask_sum += ask_sizes[idx(t, level, levels)];
    }

    return safe_divide(bid_sum - ask_sum, bid_sum + ask_sum);
}

inline double compute_spread(
    const double* bid_prices,
    const double* ask_prices,
    const std::size_t t,
    const std::size_t levels
) {
    const double bid_p = bid_prices[idx(t, 0, levels)];
    const double ask_p = ask_prices[idx(t, 0, levels)];
    return ask_p - bid_p;
}

inline double compute_microprice_deviation(
    const double* bid_prices,
    const double* bid_sizes,
    const double* ask_prices,
    const double* ask_sizes,
    const std::size_t t,
    const std::size_t levels
) {
    const double bid_p = bid_prices[idx(t, 0, levels)];
    const double ask_p = ask_prices[idx(t, 0, levels)];
    const double bid_q = bid_sizes[idx(t, 0, levels)];
    const double ask_q = ask_sizes[idx(t, 0, levels)];

    const double midprice = 0.5 * (bid_p + ask_p);
    const double microprice = safe_divide(ask_p * bid_q + bid_p * ask_q, bid_q + ask_q);

    return microprice - midprice;
}

}  // namespace

FeatureMatrix compute_features_series(
    const double* bid_prices,
    const double* bid_sizes,
    const double* ask_prices,
    const double* ask_sizes,
    const std::size_t n_rows,
    const std::size_t levels
) {
    if (bid_prices == nullptr || bid_sizes == nullptr || ask_prices == nullptr || ask_sizes == nullptr) {
        throw std::invalid_argument("Null input pointer passed to compute_features_series");
    }
    if (n_rows == 0) {
        throw std::invalid_argument("n_rows must be > 0");
    }
    if (levels < 10) {
        throw std::invalid_argument("levels must be >= 10 for this feature set");
    }

    constexpr std::size_t n_features = 8;

    FeatureMatrix out;
    out.n_rows = n_rows;
    out.n_cols = n_features;
    out.data.resize(n_rows * n_features, 0.0);

    for (std::size_t t = 0; t < n_rows; ++t) {
        const std::size_t base = t * n_features;

        double ofi_best = 0.0;
        if (t > 0) {
            ofi_best = compute_best_level_ofi(
                bid_prices, bid_sizes, ask_prices, ask_sizes, t, levels
            );
        }

        const double ofi_best_norm = compute_best_level_ofi_normalized(
            ofi_best, bid_sizes, ask_sizes, t, levels
        );

        const double qi_best = compute_queue_imbalance_best(
            bid_sizes, ask_sizes, t, levels
        );

        const double di_3 = compute_depth_imbalance_k(
            bid_sizes, ask_sizes, t, levels, 3
        );

        const double di_5 = compute_depth_imbalance_k(
            bid_sizes, ask_sizes, t, levels, 5
        );

        const double di_10 = compute_depth_imbalance_k(
            bid_sizes, ask_sizes, t, levels, 10
        );

        const double spread = compute_spread(
            bid_prices, ask_prices, t, levels
        );

        const double microprice_dev = compute_microprice_deviation(
            bid_prices, bid_sizes, ask_prices, ask_sizes, t, levels
        );

        out.data[base + 0] = ofi_best;
        out.data[base + 1] = ofi_best_norm;
        out.data[base + 2] = qi_best;
        out.data[base + 3] = di_3;
        out.data[base + 4] = di_5;
        out.data[base + 5] = di_10;
        out.data[base + 6] = spread;
        out.data[base + 7] = microprice_dev;
    }

    return out;
}

}  // namespace microalpha