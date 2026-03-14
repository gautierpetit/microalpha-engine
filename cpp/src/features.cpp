#include "microalpha/features.hpp"

#include <cmath>
#include <stdexcept>

namespace microalpha {
namespace {

inline std::size_t idx(std::size_t row, std::size_t col, std::size_t n_cols) {
    return row * n_cols + col;
}

inline double safe_divide(double num, double den) {
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
    std::size_t t,
    std::size_t levels
) {
    // Best level only
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
        // Exact comparison is intentional here:
        // LOB prices come from discrete tick-grid data, so equality of best quotes
        // should be treated as an exact state comparison, not a fuzzy float comparison.
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

inline double compute_queue_imbalance_best(
    const double* bid_sizes,
    const double* ask_sizes,
    std::size_t t,
    std::size_t levels
) {
    const double bid_q = bid_sizes[idx(t, 0, levels)];
    const double ask_q = ask_sizes[idx(t, 0, levels)];
    return safe_divide(bid_q, bid_q + ask_q);
}

inline double compute_depth_imbalance(
    const double* bid_sizes,
    const double* ask_sizes,
    std::size_t t,
    std::size_t levels
) {
    double bid_sum = 0.0;
    double ask_sum = 0.0;

    for (std::size_t level = 0; level < levels; ++level) {
        bid_sum += bid_sizes[idx(t, level, levels)];
        ask_sum += ask_sizes[idx(t, level, levels)];
    }

    return safe_divide(bid_sum - ask_sum, bid_sum + ask_sum);
}

inline double compute_spread(
    const double* bid_prices,
    const double* ask_prices,
    std::size_t t,
    std::size_t levels
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
    std::size_t t,
    std::size_t levels
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
    std::size_t n_rows,
    std::size_t levels
) {
    if (bid_prices == nullptr || bid_sizes == nullptr || ask_prices == nullptr || ask_sizes == nullptr) {
        throw std::invalid_argument("Null input pointer passed to compute_features_series");
    }
    if (n_rows == 0) {
        throw std::invalid_argument("n_rows must be > 0");
    }
    if (levels == 0) {
        throw std::invalid_argument("levels must be > 0");
    }

    constexpr std::size_t n_features = 5;
    FeatureMatrix out;
    out.n_rows = n_rows;
    out.n_cols = n_features;
    out.data.resize(n_rows * n_features, 0.0);

    // t = 0 has no previous row, so OFI is left at 0.0 by construction.
    for (std::size_t t = 0; t < n_rows; ++t) {
        const std::size_t base = t * n_features;

        double ofi = 0.0;
        if (t > 0) {
            ofi = compute_best_level_ofi(
                bid_prices, bid_sizes, ask_prices, ask_sizes, t, levels
            );
        }

        const double qi = compute_queue_imbalance_best(bid_sizes, ask_sizes, t, levels);
        const double di = compute_depth_imbalance(bid_sizes, ask_sizes, t, levels);
        const double spr = compute_spread(bid_prices, ask_prices, t, levels);
        const double mpd = compute_microprice_deviation(
            bid_prices, bid_sizes, ask_prices, ask_sizes, t, levels
        );

        out.data[base + 0] = ofi;
        out.data[base + 1] = qi;
        out.data[base + 2] = di;
        out.data[base + 3] = spr;
        out.data[base + 4] = mpd;
    }

    return out;
}

}  // namespace microalpha