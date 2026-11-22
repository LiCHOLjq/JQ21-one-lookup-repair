#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern "C" EXPORT void expected_rows(
    const double* weights, const int n, double* lookahead, double* marginals
) {
    const int states = 1 << n;
    const int full = states - 1;
    std::vector<std::vector<int>> layers(n + 1), available(states);
    std::vector<double> prefix(states), suffix(states);

    for (int mask = 0; mask < states; ++mask) {
        for (int slot = 0; slot < n; ++slot) {
            if (!(mask & (1 << slot))) {
                available[mask].push_back(slot);
            }
        }
        layers[n - static_cast<int>(available[mask].size())].push_back(mask);
    }
    prefix[0] = 1.0;
    suffix[full] = 1.0;
    for (int i = 0; i < n; ++i) {
        for (int mask : layers[i]) {
            for (int slot : available[mask]) {
                prefix[mask | (1 << slot)] += prefix[mask] * weights[i * n + slot];
            }
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        for (int mask : layers[i]) {
            for (int slot : available[mask]) {
                const double tail = weights[i * n + slot] * suffix[mask | (1 << slot)];
                suffix[mask] += tail;
                marginals[i * n + slot] += prefix[mask] * tail;
            }
        }
    }
    const double normalizer = prefix[full];
    for (int edge = 0; edge < n * n; ++edge) {
        marginals[edge] /= normalizer;
    }

    std::vector<double> marked(states * n), joint(n * n);
    std::fill(lookahead, lookahead + n, 1.0);
    for (int i = 0; i < n - 1; ++i) {
        std::fill(marked.begin(), marked.end(), 0.0);
        // The second state coordinate records the answer for first row i.
        for (int mask : layers[i]) {
            for (int answer : available[mask]) {
                const int next = mask | (1 << answer);
                marked[next * n + answer] = prefix[mask] * weights[i * n + answer];
            }
        }
        for (int k = i + 1; k < n; ++k) {
            std::fill(joint.begin(), joint.end(), 0.0);
            for (int mask : layers[k]) {
                for (int slot : available[mask]) {
                    const int next = mask | (1 << slot);
                    const double weight = weights[k * n + slot];
                    double* destination = &marked[next * n];
                    const double* source = &marked[mask * n];
                    double* pair_mass = &joint[slot * n];
                    for (int answer = 0; answer < n; ++answer) {
                        const double step = source[answer] * weight;
                        destination[answer] += step;
                        pair_mass[answer] += step * suffix[next];
                    }
                }
            }
            for (int slot = 0; slot < n; ++slot) {
                double best_for_i = 0.0;
                double best_for_k = 0.0;
                for (int other = 0; other < n; ++other) {
                    best_for_i = std::max(best_for_i, joint[other * n + slot]);
                    best_for_k = std::max(best_for_k, joint[slot * n + other]);
                }
                lookahead[i] += best_for_i / normalizer;
                lookahead[k] += best_for_k / normalizer;
            }
        }
    }
}

extern "C" EXPORT void decoder_accuracy(
    const double* scores, const std::int64_t* truth, const int n, const double tolerance,
    double* row_accuracy, std::int64_t* counts_out, double* optima
) {
    const int states = 1 << n;
    std::vector<std::vector<int>> allocated(states);
    for (int mask = 0; mask < states; ++mask) {
        for (int slot = 0; slot < n; ++slot) {
            if (mask & (1 << slot)) {
                allocated[mask].push_back(slot);
            }
        }
    }
    std::vector<double> best(states);
    std::vector<std::int64_t> counts(states), correct(states);
    for (int action = 0; action <= n; ++action) {
        const int forced_slot = action < n ? static_cast<int>(truth[action]) : -1;
        best[0] = 0.0;
        counts[0] = 1;
        correct[0] = 0;
        for (int mask = 1; mask < states; ++mask) {
            const int row = static_cast<int>(allocated[mask].size()) - 1;
            counts[mask] = 0;
            correct[mask] = 0;
            best[mask] = -std::numeric_limits<double>::infinity();
            for (int slot : allocated[mask]) {
                if ((row == action) != (slot == forced_slot)) {
                    continue;
                }
                const int previous = mask ^ (1 << slot);
                if (counts[previous] == 0) {
                    continue;
                }
                best[mask] = std::max(best[mask], best[previous] + scores[row * n + slot]);
            }
            if (!std::isfinite(best[mask])) {
                continue;
            }
            for (int slot : allocated[mask]) {
                if ((row == action) != (slot == forced_slot)) {
                    continue;
                }
                const int previous = mask ^ (1 << slot);
                const double candidate = best[previous] + scores[row * n + slot];
                if (counts[previous] && best[mask] - candidate <= tolerance) {
                    counts[mask] += counts[previous];
                    correct[mask] += correct[previous];
                    if (slot == truth[row]) {
                        correct[mask] += counts[previous];
                    }
                }
            }
        }
        optima[action] = best.back();
        counts_out[action] = counts.back();
        row_accuracy[action] = static_cast<double>(correct.back()) / counts.back() / n;
    }
}
