#include "microalpha/features.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

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

struct RollingSum {
    explicit RollingSum(const std::size_t window) : window(window), values(window, 0.0) {
        if (window == 0) {
            throw std::invalid_argument("RollingSum window must be > 0");
        }
    }

    double push(const double x) {
        if (count < window) {
            values[count] = x;
            sum += x;
            ++count;
            return sum;
        }

        const std::size_t pos = index % window;
        sum -= values[pos];
        values[pos] = x;
        sum += x;
        ++index;
        return sum;
    }

    std::size_t window;
    std::vector<double> values;
    double sum = 0.0;
    std::size_t count = 0;
    std::size_t index = 0;
};


struct RollingSampleStd {
    explicit RollingSampleStd(const std::size_t window) : window(window), values(window, 0.0) {
        if (window == 0) {
            throw std::invalid_argument("RollingSampleStd window must be > 0");
        }
    }

    double push(const double x) {
        if (count < window) {
            values[count] = x;
            sum += x;
            sumsq += x * x;
            ++count;
        } else {
            const std::size_t pos = index % window;
            const double old = values[pos];
            sum -= old;
            sumsq -= old * old;

            values[pos] = x;
            sum += x;
            sumsq += x * x;
            ++index;
        }

        const std::size_t n = std::min(count, window);
        if (n < 2) {
            return 0.0;
        }

        const double mean = sum / static_cast<double>(n);
        double ss = sumsq - static_cast<double>(n) * mean * mean;

        // Numerical guard
        if (ss < 0.0 && std::abs(ss) < 1e-12) {
            ss = 0.0;
        }
        if (ss < 0.0) {
            throw std::runtime_error("Negative rolling sum-of-squares encountered");
        }

        const double var = ss / static_cast<double>(n - 1);  // sample variance, ddof=1
        return std::sqrt(var);
    }

    std::size_t window;
    std::vector<double> values;
    double sum = 0.0;
    double sumsq = 0.0;
    std::size_t count = 0;
    std::size_t index = 0;
};

inline std::vector<double> compute_event_intensity(
    const double* timestamps,
    const std::size_t n_rows,
    const double window_seconds
) {
    if (timestamps == nullptr) {
        throw std::invalid_argument("timestamps must not be null");
    }
    if (window_seconds <= 0.0) {
        throw std::invalid_argument("window_seconds must be > 0");
    }

    std::vector<double> intensity(n_rows, 0.0);

    std::size_t left = 0;
    for (std::size_t right = 0; right < n_rows; ++right) {
        const double t_right = timestamps[right];

        while (left < right && (t_right - timestamps[left]) > window_seconds) {
            ++left;
        }

        intensity[right] = static_cast<double>(right - left + 1);
    }

    return intensity;
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
    const double* midprice,
    const double* timestamps,
    const std::size_t n_rows,
    const std::size_t levels,
    const std::size_t ofi_window_raw,
    const std::size_t ofi_norm_window_1,
    const std::size_t ofi_norm_window_2,
    const std::size_t ofi_norm_window_3,
    const std::size_t vol_window,
    const double intensity_window_seconds
) {
    if (
        bid_prices == nullptr || bid_sizes == nullptr ||
        ask_prices == nullptr || ask_sizes == nullptr ||
        midprice == nullptr || timestamps == nullptr
    ) {
        throw std::invalid_argument("Null input pointer passed to compute_features_series");
    }
    if (n_rows == 0) {
        throw std::invalid_argument("n_rows must be > 0");
    }
    if (levels < 10) {
        throw std::invalid_argument("levels must be >= 10 for this feature set");
    }
    if (ofi_window_raw == 0) {
        throw std::invalid_argument("ofi_window_raw must be > 0");
    }
    if (ofi_norm_window_1 == 0 || ofi_norm_window_2 == 0 || ofi_norm_window_3 == 0) {
        throw std::invalid_argument("ofi_norm windows must be > 0");
    }
    if (vol_window == 0) {
        throw std::invalid_argument("vol_window must be > 0");
    }
    if (intensity_window_seconds <= 0.0) {
        throw std::invalid_argument("intensity_window_seconds must be > 0");
    }

    constexpr std::size_t n_features = 14;

    FeatureMatrix out;
    out.n_rows = n_rows;
    out.n_cols = n_features;
    out.data.resize(n_rows * n_features, 0.0);

    RollingSum ofi_sum_raw(ofi_window_raw);
    RollingSum ofi_norm_sum_1(ofi_norm_window_1);
    RollingSum ofi_norm_sum_2(ofi_norm_window_2);
    RollingSum ofi_norm_sum_3(ofi_norm_window_3);
    RollingSampleStd mid_return_std(vol_window);

    const std::vector<double> event_intensity = compute_event_intensity(
        timestamps, n_rows, intensity_window_seconds
    );

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

        const double ofi_roll_sum = ofi_sum_raw.push(ofi_best);
        const double ofi_norm_roll_sum_1 = ofi_norm_sum_1.push(ofi_best_norm);
        const double ofi_norm_roll_sum_2 = ofi_norm_sum_2.push(ofi_best_norm);
        const double ofi_norm_roll_sum_3 = ofi_norm_sum_3.push(ofi_best_norm);

        double mid_return = 0.0;
        if (t > 0) {
            mid_return = midprice[t] - midprice[t - 1];
        }
        const double midprice_vol = mid_return_std.push(mid_return);

        out.data[base + 0] = ofi_best;
        out.data[base + 1] = ofi_best_norm;
        out.data[base + 2] = qi_best;
        out.data[base + 3] = di_3;
        out.data[base + 4] = di_5;
        out.data[base + 5] = di_10;
        out.data[base + 6] = spread;
        out.data[base + 7] = microprice_dev;

        out.data[base + 8] = ofi_roll_sum;
        out.data[base + 9] = ofi_norm_roll_sum_1;
        out.data[base + 10] = ofi_norm_roll_sum_2;
        out.data[base + 11] = ofi_norm_roll_sum_3;
        out.data[base + 12] = midprice_vol;
        out.data[base + 13] = event_intensity[t]; 
    }

    return out;
}

}  // namespace microalpha