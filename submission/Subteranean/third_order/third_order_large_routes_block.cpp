// Route-focused third-order hybrid evaluator for Subterranean-2.0.
// Generated from sthird_fast(1).cpp. The geometric tail is unchanged;
// only exact-prefix route selection/aggregation is replaced by a bounded
// beam plus short-tail proxy.

#ifndef SUBTERRANEAN_THIRD_ORDER_VERIFY_OPTIMIZED
#define SUBTERRANEAN_THIRD_ORDER_VERIFY_OPTIMIZED 0
#endif

#ifndef SUBTERRANEAN_THIRD_ORDER_SELF_TEST_ONLY
#define SUBTERRANEAN_THIRD_ORDER_SELF_TEST_ONLY 0
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

static constexpr int N = 257;
static constexpr int W = 5;
static constexpr int ORDER = 3;
static constexpr int CUBE = 1 << ORDER;          // 8 cube vertices
static constexpr int DIM = 1 << CUBE;            // 256 Walsh coordinates
static constexpr int FINAL_COORD = 127;           // 0b01111111
static constexpr int INV12 = 150;                 // 12^{-1} mod 257

struct Mask {
    std::array<uint64_t, W> w{};
    bool operator==(const Mask& o) const noexcept { return w == o.w; }
};

struct MaskHash {
    size_t operator()(const Mask& m) const noexcept {
        uint64_t h = 0x9e3779b97f4a7c15ULL;
        for (uint64_t x : m.w) {
            x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
            x ^= x >> 27; x *= 0x94d049bb133111ebULL;
            x ^= x >> 31;
            h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return static_cast<size_t>(h);
    }
};

static inline int modN(long long x) {
    const long long r = x % N;
    return static_cast<int>(r < 0 ? r + N : r);
}

static inline void xor_bit(Mask& m, int support_i) {
    support_i = modN(support_i);
    const int pos = N - 1 - support_i;
    m.w[pos >> 6] ^= 1ULL << (pos & 63);
}

static inline int test_bit(const Mask& m, int support_i) {
    support_i = modN(support_i);
    const int pos = N - 1 - support_i;
    return static_cast<int>((m.w[pos >> 6] >> (pos & 63)) & 1ULL);
}

static inline bool is_zero(const Mask& m) {
    return (m.w[0] | m.w[1] | m.w[2] | m.w[3] | m.w[4]) == 0ULL;
}

static inline Mask mask_xor(const Mask& a, const Mask& b) {
    Mask r;
    for (int i = 0; i < W; ++i) r.w[i] = a.w[i] ^ b.w[i];
    return r;
}

static inline int highest_pos(const Mask& m) {
    for (int wi = W - 1; wi >= 0; --wi) {
        const uint64_t x = m.w[wi];
        if (x != 0ULL) return wi * 64 + 63 - __builtin_clzll(x);
    }
    return -1;
}

static inline int popcount_mask(const Mask& m) {
    int s = 0;
    for (uint64_t x : m.w) s += __builtin_popcountll(x);
    return s;
}

static std::vector<int> active_bits(const Mask& m) {
    std::vector<int> out;
    out.reserve(popcount_mask(m));
    for (int wi = 0; wi < W; ++wi) {
        uint64_t x = m.w[wi];
        while (x != 0ULL) {
            const int b = __builtin_ctzll(x);
            const int pos = wi * 64 + b;
            if (pos < N) out.push_back(N - 1 - pos);
            x &= x - 1;
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

static Mask set_support(const std::vector<int>& xs) {
    Mask m;
    for (int x : xs) {
        if (x < 0 || x >= N) {
            throw std::runtime_error("support index outside [0,257)");
        }
        xor_bit(m, x);
    }
    return m;
}

static Mask parse_support(std::string s) {
    for (char& c : s) {
        if (c == '[' || c == ']') c = ' ';
    }
    std::stringstream ss(s);
    std::vector<int> vals;
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        std::stringstream tt(tok);
        int x = 0;
        if (tt >> x) vals.push_back(x);
    }
    return set_support(vals);
}

static std::string fmt_support(const Mask& m, int max_len = 40) {
    const auto v = active_bits(m);
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.size() && i < static_cast<size_t>(max_len); ++i) {
        if (i != 0) os << ",";
        os << v[i];
    }
    if (v.size() > static_cast<size_t>(max_len)) {
        os << ",...(+" << (v.size() - max_len) << ")";
    }
    os << "]";
    return os.str();
}

static inline bool masks_intersect(const Mask& a, const Mask& b) noexcept {
    for (int i = 0; i < W; ++i) {
        if ((a.w[i] & b.w[i]) != 0ULL) return true;
    }
    return false;
}

static inline bool triple_pairwise_intersects(
    const Mask& d0,
    const Mask& d1,
    const Mask& d2
) noexcept {
    return masks_intersect(d0, d1) ||
           masks_intersect(d0, d2) ||
           masks_intersect(d1, d2);
}

// Forward differential propagation through Subterranean L.
static Mask L_transform(const Mask& in) {
    Mask out;
    for (int p : active_bits(in)) {
        xor_bit(out, modN(static_cast<long long>(INV12) * p));
        xor_bit(out, modN(static_cast<long long>(INV12) * (p - 3)));
        xor_bit(out, modN(static_cast<long long>(INV12) * (p - 8)));
    }
    return out;
}

static std::vector<Mask> gf2_independent_basis(const std::vector<Mask>& columns) {
    std::unordered_map<int, Mask> by_lead;
    std::vector<int> insertion_order;
    by_lead.reserve(columns.size() * 2 + 1);
    insertion_order.reserve(columns.size());

    for (const Mask& c : columns) {
        Mask x = c;
        while (!is_zero(x)) {
            const int lead = highest_pos(x);
            const auto it = by_lead.find(lead);
            if (it == by_lead.end()) {
                by_lead.emplace(lead, x);
                insertion_order.push_back(lead);
                break;
            }
            x = mask_xor(x, it->second);
        }
    }

    std::vector<Mask> basis;
    basis.reserve(insertion_order.size());
    for (int lead : insertion_order) basis.push_back(by_lead.at(lead));
    return basis;
}

// Exact chi output-difference distribution for one input difference.
static std::vector<std::pair<Mask, double>> chi_diff_distribution(const Mask& delta) {
    Mask base;
    for (int i = 0; i < N; ++i) {
        const int di = test_bit(delta, i);
        const int d1 = test_bit(delta, i + 1);
        const int d2 = test_bit(delta, i + 2);
        if (di ^ d2 ^ (d1 & d2)) xor_bit(base, i);
    }

    std::vector<Mask> columns;
    columns.reserve(2 * popcount_mask(delta));
    for (int j = 0; j < N; ++j) {
        Mask col;
        if (test_bit(delta, j - 1)) xor_bit(col, j - 2);
        if (test_bit(delta, j + 1)) xor_bit(col, j - 1);
        if (!is_zero(col)) columns.push_back(col);
    }

    const auto basis = gf2_independent_basis(columns);
    const int rank = static_cast<int>(basis.size());
    if (rank >= 63) throw std::runtime_error("chi basis too large to enumerate");

    const double probability = std::ldexp(1.0, -rank);
    std::vector<std::pair<Mask, double>> outputs;
    outputs.reserve(1ULL << rank);
    outputs.push_back({base, probability});
    for (const Mask& b : basis) {
        const size_t old_size = outputs.size();
        for (size_t k = 0; k < old_size; ++k) {
            outputs.push_back({mask_xor(outputs[k].first, b), probability});
        }
    }
    return outputs;
}

static inline const std::vector<std::pair<Mask, double>>& chi_cached_ref(
    const Mask& d,
    std::unordered_map<Mask, std::vector<std::pair<Mask, double>>, MaskHash>& cache
) {
    const auto it = cache.find(d);
    if (it != cache.end()) return it->second;
    auto result = chi_diff_distribution(d);
    return cache.emplace(d, std::move(result)).first->second;
}

static inline Mask L_cached(
    const Mask& d,
    std::unordered_map<Mask, Mask, MaskHash>& cache
) {
    const auto it = cache.find(d);
    if (it != cache.end()) return it->second;
    const Mask out = L_transform(d);
    cache.emplace(d, out);
    return out;
}

// -------------------- third-order geometric tail --------------------

using Row256 = std::array<double, DIM>;
using Gamma = std::array<Row256, N>;
using CorrVec = std::array<double, N>;
using DTable = std::array<Row256, N>;

// True BLOCK_SIZE=2 enlarged-unit engine for third-order Subterranean-2.0.
// Requires the including translation unit to define:
//   N=257, DIM=256, FINAL_COORD=127, Mask, test_bit(), modN(), CorrVec.
#ifndef SUBTERRANEAN_THIRD_ORDER_TRUE_BLOCK2_ENGINE_HPP
#define SUBTERRANEAN_THIRD_ORDER_TRUE_BLOCK2_ENGINE_HPP

#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <random>

class TrueBlock2ThirdOrderEngine {
public:
    static constexpr int BLOCK_SIZE = 2;
    static constexpr int ALPHABET = 256;
    static constexpr int FULL_DIM = 65536;

    using Factor = std::vector<double>;
    using Factors = std::vector<Factor>;

    TrueBlock2ThirdOrderEngine() {
        build_dual_table();
        build_layout();
        build_linear_plans();
    }

    CorrVec tail_correlations(
        const Mask& d0,
        const Mask& d1,
        const Mask& d2,
        int tail_rounds
    ) const {
        if (tail_rounds <= 0) {
            throw std::runtime_error("block2 tail_rounds must be positive");
        }
        Factors factors = initial_factors(d0, d1, d2);
        for (int round = 0; round < tail_rounds; ++round) {
            const bool last = (round + 1 == tail_rounds);
            if (last) return final_chi_single_bits(factors);
            factors = pass_chi_full(factors);
            factors = pass_linear_full(factors);
        }
        throw std::runtime_error("unreachable block2 tail loop");
    }

    void self_test() const {
        std::cout << "Third-order true BLOCK_SIZE=2 self-tests\n";
        test_fwt_round_trip();
        test_local_chi_against_sparse_bruteforce();
        test_linear_plans();
        std::cout << "  true block2 engine: PASS\n" << std::flush;
    }

private:
    struct BlockSpec {
        int start = 0;
        int length = 0;
        int dimension = 0;
    };

    struct LinearPlan {
        std::vector<int> source_blocks;
        std::vector<uint16_t> source_masks;
        int dimension = 0;
    };

    std::array<std::array<uint8_t, ALPHABET>, ALPHABET> dual_{};
    std::vector<BlockSpec> blocks_;
    std::array<int, N> block_of_{};
    std::array<int, N> local_of_{};
    std::vector<LinearPlan> linear_plans_;

    static inline int parity8(unsigned x) noexcept {
        return __builtin_popcount(x) & 1;
    }

    static void fwt_dense(double* values, int n) {
        for (int half = 1; half < n; half <<= 1) {
            for (int start = 0; start < n; start += 2 * half) {
                for (int offset = 0; offset < half; ++offset) {
                    const double left = values[start + offset];
                    const double right = values[start + offset + half];
                    values[start + offset] = left + right;
                    values[start + offset + half] = left - right;
                }
            }
        }
    }

    static int rep_to_vertices(int r) noexcept {
        const int x = (r >> 7) & 1;
        int out = 0;
        for (int subset = 0; subset < 8; ++subset) {
            const int value = subset == 0
                ? x
                : (x ^ ((r >> (7 - subset)) & 1));
            out |= value << (7 - subset);
        }
        return out;
    }

    static int vertices_to_rep(int vertices) noexcept {
        const int x = (vertices >> 7) & 1;
        int out = x << 7;
        for (int subset = 1; subset < 8; ++subset) {
            const int derivative =
                x ^ ((vertices >> (7 - subset)) & 1);
            out |= derivative << (7 - subset);
        }
        return out;
    }

    static int h_rep(int middle, int right) noexcept {
        const int m = rep_to_vertices(middle);
        const int r = rep_to_vertices(right);
        return vertices_to_rep(((~m) & r) & 0xff);
    }

    static int chi_rep(int left, int middle, int right) noexcept {
        return left ^ h_rep(middle, right);
    }

    void build_dual_table() {
        // dual_[b][v] is the unique mask m satisfying
        //   <v,h(b,c)> = <m,c>  for every c.
        for (int b = 0; b < ALPHABET; ++b) {
            for (int v = 0; v < ALPHABET; ++v) {
                int mask = 0;
                for (int bit = 0; bit < 8; ++bit) {
                    const int basis = 1 << bit;
                    if (parity8(static_cast<unsigned>(v & h_rep(b, basis)))) {
                        mask |= basis;
                    }
                }
                dual_[b][v] = static_cast<uint8_t>(mask);
            }
        }
    }

    void build_layout() {
        int start = 0;
        while (start < N) {
            const int length = std::min(BLOCK_SIZE, N - start);
            const int block = static_cast<int>(blocks_.size());
            blocks_.push_back({
                start,
                length,
                length == 2 ? FULL_DIM : ALPHABET
            });
            for (int local = 0; local < length; ++local) {
                block_of_[start + local] = block;
                local_of_[start + local] = local;
            }
            start += length;
        }
        if (blocks_.size() != 129 || blocks_.back().length != 1) {
            throw std::runtime_error("unexpected block2 layout");
        }
    }

    void build_linear_plans() {
        struct BasisTerm {
            int source_block = 0;
            uint16_t source_mask = 0;
        };

        linear_plans_.resize(blocks_.size());
        for (int block = 0;
             block < static_cast<int>(blocks_.size());
             ++block) {
            const BlockSpec& spec = blocks_[block];
            std::vector<int> sources;
            std::vector<std::array<BasisTerm, 3>> basis(8 * spec.length);

            for (int local = 0; local < spec.length; ++local) {
                const int output_position = spec.start + local;
                const int positions[3] = {
                    modN(12LL * output_position),
                    modN(12LL * output_position + 3),
                    modN(12LL * output_position + 8)
                };
                for (int component = 0; component < 8; ++component) {
                    for (int term = 0; term < 3; ++term) {
                        const int source_position = positions[term];
                        const int source_block = block_of_[source_position];
                        const int source_local = local_of_[source_position];
                        if (std::find(sources.begin(), sources.end(), source_block)
                            == sources.end()) {
                            sources.push_back(source_block);
                        }
                        basis[8 * local + component][term] = {
                            source_block,
                            static_cast<uint16_t>(
                                1u << (8 * source_local + component)
                            )
                        };
                    }
                }
            }

            LinearPlan plan;
            plan.source_blocks = sources;
            plan.dimension = spec.dimension;
            const size_t source_count = sources.size();
            plan.source_masks.assign(
                static_cast<size_t>(plan.dimension) * source_count,
                0
            );

            for (int coordinate = 1;
                 coordinate < plan.dimension;
                 ++coordinate) {
                const int low_bit =
                    __builtin_ctz(static_cast<unsigned>(coordinate));
                const int previous = coordinate & (coordinate - 1);
                for (size_t j = 0; j < source_count; ++j) {
                    plan.source_masks[
                        static_cast<size_t>(coordinate) * source_count + j
                    ] = plan.source_masks[
                        static_cast<size_t>(previous) * source_count + j
                    ];
                }
                for (const BasisTerm& term : basis[low_bit]) {
                    const auto it = std::find(
                        sources.begin(), sources.end(), term.source_block
                    );
                    const size_t index =
                        static_cast<size_t>(it - sources.begin());
                    plan.source_masks[
                        static_cast<size_t>(coordinate) * source_count + index
                    ] ^= term.source_mask;
                }
            }
            linear_plans_[block] = std::move(plan);
        }
    }

    static Factor initial_row(int a, int b, int c) {
        Factor row(ALPHABET, 0.0);
        const int d01 = a ^ b;
        const int d02 = a ^ c;
        const int d12 = b ^ c;
        const int d012 = a ^ b ^ c;
        for (int x = 0; x <= 1; ++x) {
            const int state =
                (x << 7) |
                (a << 6) |
                (b << 5) |
                (d01 << 4) |
                (c << 3) |
                (d02 << 2) |
                (d12 << 1) |
                d012;
            row[state] = 1.0;
        }
        fwt_dense(row.data(), ALPHABET);
        for (double& value : row) value *= 0.5;
        return row;
    }

    Factors initial_factors(
        const Mask& d0,
        const Mask& d1,
        const Mask& d2
    ) const {
        Factors factors(blocks_.size());
#pragma omp parallel for schedule(static)
        for (int block = 0;
             block < static_cast<int>(blocks_.size());
             ++block) {
            const BlockSpec& spec = blocks_[block];
            std::array<Factor, 2> rows;
            for (int local = 0; local < spec.length; ++local) {
                const int position = spec.start + local;
                rows[local] = initial_row(
                    test_bit(d0, position),
                    test_bit(d1, position),
                    test_bit(d2, position)
                );
            }
            if (spec.length == 1) {
                factors[block] = std::move(rows[0]);
            } else {
                Factor factor(FULL_DIM);
                for (int second = 0; second < ALPHABET; ++second) {
                    for (int first = 0; first < ALPHABET; ++first) {
                        factor[first | (second << 8)] =
                            rows[0][first] * rows[1][second];
                    }
                }
                factors[block] = std::move(factor);
            }
        }
        return factors;
    }

    static Factor correlation_to_pmf(const Factor& correlation) {
        Factor pmf = correlation;
        fwt_dense(pmf.data(), static_cast<int>(pmf.size()));
        const double scale = 1.0 / static_cast<double>(pmf.size());
        for (double& value : pmf) value *= scale;
        return pmf;
    }

    static Factor marginal_single(
        const Factor& pmf,
        int block_length,
        int local
    ) {
        if (block_length == 1) return pmf;
        Factor marginal(ALPHABET, 0.0);
        for (int second = 0; second < ALPHABET; ++second) {
            for (int first = 0; first < ALPHABET; ++first) {
                marginal[local == 0 ? first : second] +=
                    pmf[first | (second << 8)];
            }
        }
        return marginal;
    }

    Factor right_context_pmf(
        const Factors& pmfs,
        int target_block
    ) const {
        const BlockSpec& target = blocks_[target_block];
        const int position1 = (target.start + target.length) % N;
        const int position2 = (position1 + 1) % N;
        const int block1 = block_of_[position1];
        const int block2 = block_of_[position2];
        const int local1 = local_of_[position1];
        const int local2 = local_of_[position2];

        if (block1 == block2 && blocks_[block1].length == 2
            && local1 == 0 && local2 == 1) {
            return pmfs[block1];
        }

        const Factor marginal1 = marginal_single(
            pmfs[block1], blocks_[block1].length, local1
        );
        const Factor marginal2 = marginal_single(
            pmfs[block2], blocks_[block2].length, local2
        );
        Factor context(FULL_DIM);
        for (int second = 0; second < ALPHABET; ++second) {
            for (int first = 0; first < ALPHABET; ++first) {
                context[first | (second << 8)] =
                    marginal1[first] * marginal2[second];
            }
        }
        return context;
    }

    Factor chi_pair_correlation(
        const Factor& target_pmf,
        const Factor& context_pmf
    ) const {
        Factor target_partial(FULL_DIM);
        Factor context_partial(FULL_DIM);
        Factor boundary_transform(FULL_DIM);
        Factor output(FULL_DIM, 0.0);
        std::array<double, ALPHABET> temporary{};

        // Partial Walsh transform of P(x0,x1) in x0.
        for (int x1 = 0; x1 < ALPHABET; ++x1) {
            for (int x0 = 0; x0 < ALPHABET; ++x0) {
                temporary[x0] = target_pmf[x0 | (x1 << 8)];
            }
            fwt_dense(temporary.data(), ALPHABET);
            for (int v0 = 0; v0 < ALPHABET; ++v0) {
                target_partial[v0 | (x1 << 8)] = temporary[v0];
            }
        }

        // For each x2, transform Q(x2,x3) in x3.
        for (int x2 = 0; x2 < ALPHABET; ++x2) {
            for (int x3 = 0; x3 < ALPHABET; ++x3) {
                temporary[x3] = context_pmf[x2 | (x3 << 8)];
            }
            fwt_dense(temporary.data(), ALPHABET);
            for (int mask = 0; mask < ALPHABET; ++mask) {
                context_partial[mask | (x2 << 8)] = temporary[mask];
            }
        }

        // B[v1,x2] = sum_x3 Q(x2,x3)(-1)^<v1,h(x2,x3)>,
        // then Walsh transform B in x2.
        for (int v1 = 0; v1 < ALPHABET; ++v1) {
            for (int x2 = 0; x2 < ALPHABET; ++x2) {
                temporary[x2] = context_partial[
                    dual_[x2][v1] | (x2 << 8)
                ];
            }
            fwt_dense(temporary.data(), ALPHABET);
            for (int mask = 0; mask < ALPHABET; ++mask) {
                boundary_transform[mask | (v1 << 8)] = temporary[mask];
            }
        }

        // Exact contraction for all 256^2 output coordinates.
        for (int v1 = 0; v1 < ALPHABET; ++v1) {
            const double* boundary =
                &boundary_transform[static_cast<size_t>(v1) << 8];
            for (int x1 = 0; x1 < ALPHABET; ++x1) {
                const double sign = parity8(
                    static_cast<unsigned>(v1 & x1)
                ) ? -1.0 : 1.0;
                const double* partial =
                    &target_partial[static_cast<size_t>(x1) << 8];
                for (int v0 = 0; v0 < ALPHABET; ++v0) {
                    output[v0 | (v1 << 8)] +=
                        partial[v0] * sign * boundary[dual_[x1][v0]];
                }
            }
        }
        output[0] = 1.0;
        return output;
    }

    Factor chi_single_correlation(
        const Factor& target_pmf,
        const Factor& context_pmf
    ) const {
        Factor target_transform = target_pmf;
        fwt_dense(target_transform.data(), ALPHABET);

        Factor context_partial(FULL_DIM);
        std::array<double, ALPHABET> temporary{};
        for (int middle = 0; middle < ALPHABET; ++middle) {
            for (int right = 0; right < ALPHABET; ++right) {
                temporary[right] =
                    context_pmf[middle | (right << 8)];
            }
            fwt_dense(temporary.data(), ALPHABET);
            for (int mask = 0; mask < ALPHABET; ++mask) {
                context_partial[mask | (middle << 8)] = temporary[mask];
            }
        }

        Factor output(ALPHABET, 0.0);
        for (int v = 0; v < ALPHABET; ++v) {
            double contraction = 0.0;
            for (int middle = 0; middle < ALPHABET; ++middle) {
                contraction += context_partial[
                    dual_[middle][v] | (middle << 8)
                ];
            }
            output[v] = target_transform[v] * contraction;
        }
        output[0] = 1.0;
        return output;
    }

    Factors pass_chi_full(const Factors& correlations) const {
        Factors pmfs(correlations.size());
#pragma omp parallel for schedule(static)
        for (int block = 0;
             block < static_cast<int>(correlations.size());
             ++block) {
            pmfs[block] = correlation_to_pmf(correlations[block]);
        }

        Factors output(correlations.size());
#pragma omp parallel for schedule(dynamic, 1)
        for (int block = 0;
             block < static_cast<int>(blocks_.size());
             ++block) {
            const Factor context = right_context_pmf(pmfs, block);
            output[block] = blocks_[block].length == 2
                ? chi_pair_correlation(pmfs[block], context)
                : chi_single_correlation(pmfs[block], context);
        }
        return output;
    }

    Factors pass_linear_full(const Factors& correlations) const {
        Factors output(blocks_.size());
#pragma omp parallel for schedule(dynamic, 1)
        for (int block = 0;
             block < static_cast<int>(blocks_.size());
             ++block) {
            const LinearPlan& plan = linear_plans_[block];
            const size_t source_count = plan.source_blocks.size();
            Factor factor(plan.dimension, 1.0);
            for (int coordinate = 1;
                 coordinate < plan.dimension;
                 ++coordinate) {
                const uint16_t* masks = &plan.source_masks[
                    static_cast<size_t>(coordinate) * source_count
                ];
                double value = 1.0;
                for (size_t source = 0;
                     source < source_count;
                     ++source) {
                    value *= correlations[
                        plan.source_blocks[source]
                    ][masks[source]];
                    if (value == 0.0) break;
                }
                factor[coordinate] = value;
            }
            output[block] = std::move(factor);
        }
        return output;
    }

    CorrVec final_chi_single_bits(const Factors& correlations) const {
        Factors pmfs(correlations.size());
#pragma omp parallel for schedule(static)
        for (int block = 0;
             block < static_cast<int>(correlations.size());
             ++block) {
            pmfs[block] = correlation_to_pmf(correlations[block]);
        }

        CorrVec output{};
        const int v = FINAL_COORD;
#pragma omp parallel for schedule(static)
        for (int position = 0; position < N; ++position) {
            const int position1 = (position + 1) % N;
            const int position2 = (position + 2) % N;
            const int block0 = block_of_[position];
            const int block1 = block_of_[position1];
            const int block2 = block_of_[position2];
            const int local0 = local_of_[position];
            const int local1 = local_of_[position1];
            const int local2 = local_of_[position2];

            if (block0 == block1) {
                const Factor right = marginal_single(
                    pmfs[block2], blocks_[block2].length, local2
                );
                Factor right_transform = right;
                fwt_dense(right_transform.data(), ALPHABET);
                const Factor& pair = pmfs[block0];
                double correlation = 0.0;
                for (int middle = 0; middle < ALPHABET; ++middle) {
                    const double boundary =
                        right_transform[dual_[middle][v]];
                    for (int left = 0; left < ALPHABET; ++left) {
                        const int index =
                            (local0 == 0 && local1 == 1)
                                ? left | (middle << 8)
                                : middle | (left << 8);
                        correlation += pair[index]
                            * (parity8(static_cast<unsigned>(v & left))
                                ? -1.0 : 1.0)
                            * boundary;
                    }
                }
                output[position] = correlation;
            } else if (block1 == block2) {
                const Factor left = marginal_single(
                    pmfs[block0], blocks_[block0].length, local0
                );
                double left_correlation = 0.0;
                for (int state = 0; state < ALPHABET; ++state) {
                    left_correlation += left[state]
                        * (parity8(static_cast<unsigned>(v & state))
                            ? -1.0 : 1.0);
                }
                const Factor& pair = pmfs[block1];
                double nonlinear = 0.0;
                for (int middle = 0; middle < ALPHABET; ++middle) {
                    for (int right = 0; right < ALPHABET; ++right) {
                        const int index =
                            (local1 == 0 && local2 == 1)
                                ? middle | (right << 8)
                                : right | (middle << 8);
                        nonlinear += pair[index]
                            * (parity8(static_cast<unsigned>(
                                v & h_rep(middle, right)
                            )) ? -1.0 : 1.0);
                    }
                }
                output[position] = left_correlation * nonlinear;
            } else {
                const Factor left = marginal_single(
                    pmfs[block0], blocks_[block0].length, local0
                );
                const Factor middle = marginal_single(
                    pmfs[block1], blocks_[block1].length, local1
                );
                const Factor right = marginal_single(
                    pmfs[block2], blocks_[block2].length, local2
                );
                double left_correlation = 0.0;
                for (int state = 0; state < ALPHABET; ++state) {
                    left_correlation += left[state]
                        * (parity8(static_cast<unsigned>(v & state))
                            ? -1.0 : 1.0);
                }
                double nonlinear = 0.0;
                for (int m = 0; m < ALPHABET; ++m) {
                    for (int r = 0; r < ALPHABET; ++r) {
                        nonlinear += middle[m] * right[r]
                            * (parity8(static_cast<unsigned>(
                                v & h_rep(m, r)
                            )) ? -1.0 : 1.0);
                    }
                }
                output[position] = left_correlation * nonlinear;
            }
        }
        return output;
    }

    void test_fwt_round_trip() const {
        Factor values(FULL_DIM);
        for (int i = 0; i < FULL_DIM; ++i) {
            values[i] = std::sin(0.01 * i) + 0.25 * std::cos(0.03 * i);
        }
        const Factor original = values;
        fwt_dense(values.data(), FULL_DIM);
        fwt_dense(values.data(), FULL_DIM);
        const double scale = 1.0 / FULL_DIM;
        double maximum = 0.0;
        for (int i = 0; i < FULL_DIM; ++i) {
            maximum = std::max(
                maximum,
                std::fabs(values[i] * scale - original[i])
            );
        }
        std::cout << "  FWT round-trip max error : "
                  << std::scientific << maximum << "\n";
        if (maximum > 2e-12) {
            throw std::runtime_error("block2 FWT self-test failed");
        }
    }

    void test_local_chi_against_sparse_bruteforce() const {
        Factor target(FULL_DIM, 0.0);
        Factor context(FULL_DIM, 0.0);
        const std::array<int, 4> target_states = {
            0x0000, 0x0183, 0x7f22, 0xa55a
        };
        const std::array<int, 4> context_states = {
            0x0000, 0x1307, 0x42c1, 0xff10
        };
        for (int state : target_states) target[state] += 0.25;
        for (int state : context_states) context[state] += 0.25;

        const Factor obtained = chi_pair_correlation(target, context);
        Factor expected_pmf(FULL_DIM, 0.0);
        for (int target_state : target_states) {
            const int x0 = target_state & 0xff;
            const int x1 = (target_state >> 8) & 0xff;
            for (int context_state : context_states) {
                const int x2 = context_state & 0xff;
                const int x3 = (context_state >> 8) & 0xff;
                const int y0 = chi_rep(x0, x1, x2);
                const int y1 = chi_rep(x1, x2, x3);
                expected_pmf[y0 | (y1 << 8)] += 1.0 / 16.0;
            }
        }
        fwt_dense(expected_pmf.data(), FULL_DIM);
        double maximum = 0.0;
        for (int i = 0; i < FULL_DIM; ++i) {
            maximum = std::max(
                maximum,
                std::fabs(obtained[i] - expected_pmf[i])
            );
        }
        std::cout << "  local chi max error      : "
                  << std::scientific << maximum << "\n";
        if (maximum > 2e-12) {
            throw std::runtime_error("block2 local chi self-test failed");
        }
    }

    void test_linear_plans() const {
        for (int block = 0;
             block < static_cast<int>(blocks_.size());
             block += 17) {
            const LinearPlan& plan = linear_plans_[block];
            const int coordinate =
                std::min(plan.dimension - 1, 0xa55a);
            std::vector<uint16_t> expected(plan.source_blocks.size(), 0);
            for (int local = 0; local < blocks_[block].length; ++local) {
                const int output_position = blocks_[block].start + local;
                for (int component = 0; component < 8; ++component) {
                    if (((coordinate >> (8 * local + component)) & 1) == 0) {
                        continue;
                    }
                    const int positions[3] = {
                        modN(12LL * output_position),
                        modN(12LL * output_position + 3),
                        modN(12LL * output_position + 8)
                    };
                    for (int source_position : positions) {
                        const int source_block = block_of_[source_position];
                        const int source_local = local_of_[source_position];
                        const auto it = std::find(
                            plan.source_blocks.begin(),
                            plan.source_blocks.end(),
                            source_block
                        );
                        expected[static_cast<size_t>(
                            it - plan.source_blocks.begin()
                        )] ^= static_cast<uint16_t>(
                            1u << (8 * source_local + component)
                        );
                    }
                }
            }
            const size_t source_count = plan.source_blocks.size();
            for (size_t source = 0; source < source_count; ++source) {
                const uint16_t obtained = plan.source_masks[
                    static_cast<size_t>(coordinate) * source_count + source
                ];
                if (obtained != expected[source]) {
                    throw std::runtime_error("block2 linear-plan self-test failed");
                }
            }
        }
        std::cout << "  linear plans             : PASS\n";
    }
};

#endif

static TrueBlock2ThirdOrderEngine TRUE_BLOCK2_ENGINE;

static std::array<std::array<int, 3>, N> L_ROWS;

alignas(64) static std::array<uint8_t, DIM> P_MAP_TABLE;
alignas(64) static std::array<uint8_t, DIM> P_INV_TABLE;
alignas(64) static std::array<uint8_t, DIM> PT_MAP_TABLE;
alignas(64) static std::array<std::array<uint8_t, DIM>, DIM>
    SOURCE_COORD_TABLE;
alignas(64) static std::array<std::array<int8_t, DIM>, DIM>
    SIGN_TABLE;
static std::array<int, N> NEXT1_POSITION;
static std::array<int, N> NEXT2_POSITION;

static void init_tables() {
    for (int i = 0; i < N; ++i) {
        // Pullback of one output mask bit through L:
        // L^T e_i = e_{12i} + e_{12i+3} + e_{12i+8}.
        L_ROWS[i] = {
            modN(12LL * i),
            modN(12LL * i + 3),
            modN(12LL * i + 8)
        };

        NEXT1_POSITION[i] = (i + 1 == N) ? 0 : i + 1;
        NEXT2_POSITION[i] =
            (i + 2 < N) ? i + 2 : i + 2 - N;
    }
}

static void fwt_inplace(Row256& row) {
    for (int h = 1; h < DIM; h <<= 1) {
        for (int i = 0; i < DIM; i += 2 * h) {
            for (int j = i; j < i + h; ++j) {
                const double x = row[j];
                const double y = row[j + h];
                row[j] = x + y;
                row[j + h] = x - y;
            }
        }
    }
}

// P changes the first (MSB) derivative-representation coordinate into the
// parity of all eight coordinates and leaves the remaining seven unchanged.
static inline int p_map8(int x) noexcept {
    const int low = x & 0x7f;
    const int high = __builtin_popcount(static_cast<unsigned>(x)) & 1;
    return (high << 7) | low;
}

static inline int p_inv8(int x) noexcept {
    const int low = x & 0x7f;
    const int q0 = (x >> 7) & 1;
    const int original_high = q0 ^ (__builtin_popcount(static_cast<unsigned>(low)) & 1);
    return (original_high << 7) | low;
}

// Q=P^T in the standard Walsh dot product.  For b=(b0,b1,...,b7),
// Q(b)=(b0,b0 xor b1,...,b0 xor b7).
static inline int pt_map8(int x) noexcept {
    return ((x >> 7) & 1) ? (x ^ 0x7f) : x;
}

static void init_fast_chi_tables() {
    for (int coordinate = 0;
         coordinate < DIM;
         ++coordinate) {
        P_MAP_TABLE[coordinate] =
            static_cast<uint8_t>(
                p_map8(coordinate)
            );

        P_INV_TABLE[coordinate] =
            static_cast<uint8_t>(
                p_inv8(coordinate)
            );

        PT_MAP_TABLE[coordinate] =
            static_cast<uint8_t>(
                pt_map8(coordinate)
            );
    }

    for (int output_coordinate = 0;
         output_coordinate < DIM;
         ++output_coordinate) {
        const int transformed_output =
            P_MAP_TABLE[output_coordinate];

        for (int boundary = 0;
             boundary < DIM;
             ++boundary) {
            SOURCE_COORD_TABLE[
                output_coordinate
            ][boundary] =
                P_INV_TABLE[
                    boundary
                    & transformed_output
                ];

            SIGN_TABLE[
                output_coordinate
            ][boundary] =
                (
                    __builtin_popcount(
                        static_cast<unsigned>(
                            transformed_output
                            & boundary
                        )
                    ) & 1
                )
                ? static_cast<int8_t>(-1)
                : static_cast<int8_t>(1);
        }
    }
}

static Gamma init_gamma_third_order(
    const Mask& d0,
    const Mask& d1,
    const Mask& d2
) {
    Gamma gamma{};
    for (int i = 0; i < N; ++i) {
        const int a = test_bit(d0, i);
        const int b = test_bit(d1, i);
        const int c = test_bit(d2, i);
        const int d01 = a ^ b;
        const int d02 = a ^ c;
        const int d12 = b ^ c;
        const int d012 = a ^ b ^ c;

        // Natural order of cube subsets:
        // (x,d0,d1,d01,d2,d02,d12,d012).
        Row256& row = gamma[i];
        row.fill(0.0);
        for (int x = 0; x <= 1; ++x) {
            const int index =
                (x    << 7) |
                (a    << 6) |
                (b    << 5) |
                (d01  << 4) |
                (c    << 3) |
                (d02  << 2) |
                (d12  << 1) |
                d012;
            row[index] = 1.0;
        }
        fwt_inplace(row);
        for (double& z : row) z *= 0.5;
    }
    return gamma;
}

// Exact reduced form of the third-order chi transfer.
//
// For each row j define
//   d_j[b] = 2^-8 * W(gamma_j)[P^T b].
// The rel=2 and rel=0 local matrices are rank one.  In the cyclic chain,
// all unmodified rel=2 factors therefore collapse to scalar products.
// The remaining rel=1 factor is evaluated through
//   sum_b d[b] (-1)^{P(v)��a + (P(v)&a)��b}
// = (-1)^{P(v)��a} gamma[P^{-1}(P(v)&a)].
// This gives exactly the same trace as the direct 256x256 matrix product.
#if SUBTERRANEAN_THIRD_ORDER_VERIFY_OPTIMIZED
static Gamma pass_chi_third_order_reference_formula(const Gamma& gamma, bool only_final_coord) {
    DTable dtable{};

    // One 256-point FWT per state bit; reused for every output coordinate v.
#pragma omp parallel for schedule(static)
    for (int j = 0; j < N; ++j) {
        Row256 transformed = gamma[j];
        fwt_inplace(transformed);
        for (int b = 0; b < DIM; ++b) {
            dtable[j][b] = transformed[pt_map8(b)] / static_cast<double>(DIM);
        }
    }

    Gamma sigma{};
    for (int i = 0; i < N; ++i) sigma[i][0] = 1.0;

    const int v_begin = only_final_coord ? FINAL_COORD : 1;
    const int v_end = only_final_coord ? FINAL_COORD + 1 : DIM;

#pragma omp parallel for schedule(dynamic, 1)
    for (int v = v_begin; v < v_end; ++v) {
        const int V = p_map8(v);

        for (int i = 0; i < N; ++i) {
            const int j1 = (i + 1) % N;
            const int j2 = (i + 2) % N;

            // Every valid gamma row has coordinate 0 exactly equal to 1.
            // Therefore all untouched rel=2 rank-one factors collapse to 1.
            const double rel0_sum = gamma[i][v];
            double rel1_contraction = 0.0;
            for (int a = 0; a < DIM; ++a) {
                const double sign = (__builtin_popcount(
                    static_cast<unsigned>(V & a)) & 1) ? -1.0 : 1.0;
                const int source_coordinate = p_inv8(a & V);
                rel1_contraction +=
                    dtable[j2][a] * sign * gamma[j1][source_coordinate];
            }

            sigma[i][v] = rel0_sum * rel1_contraction;
        }
    }
    return sigma;
}

#endif

static Gamma pass_chi_third_order(
    const Gamma& gamma,
    bool only_final_coordinate
) {
    DTable dtable{};

    constexpr double inverse_dimension =
        1.0 / static_cast<double>(DIM);

    // One 256-point FWT per state bit.
#pragma omp parallel for schedule(static)
    for (int position = 0;
         position < N;
         ++position) {
        Row256 transformed =
            gamma[position];

        fwt_inplace(transformed);

        double* destination =
            dtable[position].data();

        for (int boundary = 0;
             boundary < DIM;
             ++boundary) {
            destination[boundary] =
                transformed[
                    PT_MAP_TABLE[boundary]
                ]
                * inverse_dimension;
        }
    }

    Gamma output{};

    for (int position = 0;
         position < N;
         ++position) {
        output[position][0] = 1.0;
    }

    const int coordinate_begin =
        only_final_coordinate
            ? FINAL_COORD
            : 1;

    const int coordinate_end =
        only_final_coordinate
            ? FINAL_COORD + 1
            : DIM;

    // Static scheduling is balanced: each output coordinate performs the same
    // N*DIM contraction work.
#pragma omp parallel for schedule(static)
    for (int output_coordinate = coordinate_begin;
         output_coordinate < coordinate_end;
         ++output_coordinate) {
        const uint8_t* source_coordinates =
            SOURCE_COORD_TABLE[
                output_coordinate
            ].data();

        const int8_t* signs =
            SIGN_TABLE[
                output_coordinate
            ].data();

        for (int position = 0;
             position < N;
             ++position) {
            const double* drow =
                dtable[
                    NEXT2_POSITION[position]
                ].data();

            const double* next_gamma =
                gamma[
                    NEXT1_POSITION[position]
                ].data();

            double contraction = 0.0;

#pragma omp simd reduction(+:contraction)
            for (int boundary = 0;
                 boundary < DIM;
                 ++boundary) {
                contraction +=
                    drow[boundary]
                    * static_cast<double>(
                        signs[boundary]
                    )
                    * next_gamma[
                        source_coordinates[boundary]
                    ];
            }

            output[position][
                output_coordinate
            ] =
                gamma[position][
                    output_coordinate
                ]
                * contraction;
        }
    }

    return output;
}

static Gamma pass_linear_third_order(const Gamma& gamma) {
    Gamma sigma{};
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        sigma[i][0] = 1.0;
        const int a = L_ROWS[i][0];
        const int b = L_ROWS[i][1];
        const int c = L_ROWS[i][2];
        for (int v = 1; v < DIM; ++v) {
            sigma[i][v] = gamma[a][v] * gamma[b][v] * gamma[c][v];
        }
    }
    return sigma;
}


#if SUBTERRANEAN_THIRD_ORDER_VERIFY_OPTIMIZED

static double maximum_gamma_difference(
    const Gamma& left,
    const Gamma& right,
    bool only_final_coordinate,
    int& worst_position,
    int& worst_coordinate
) {
    double maximum = 0.0;
    worst_position = 0;
    worst_coordinate = 0;

    const int coordinate_begin =
        only_final_coordinate
            ? FINAL_COORD
            : 0;

    const int coordinate_end =
        only_final_coordinate
            ? FINAL_COORD + 1
            : DIM;

    for (int position = 0;
         position < N;
         ++position) {
        for (int coordinate = coordinate_begin;
             coordinate < coordinate_end;
             ++coordinate) {
            const double difference =
                std::fabs(
                    left[position][coordinate]
                    - right[position][coordinate]
                );

            if (difference > maximum) {
                maximum = difference;
                worst_position = position;
                worst_coordinate = coordinate;
            }
        }
    }

    return maximum;
}

static void verify_subterranean_linear_layer() {
    for (int input_position = 0;
         input_position < N;
         ++input_position) {
        Mask basis;
        xor_bit(basis, input_position);

        const Mask formula =
            L_transform(basis);

        Mask explicit_image;

        for (int output_position = 0;
             output_position < N;
             ++output_position) {
            const bool coefficient =
                input_position
                    == modN(12LL * output_position)
                || input_position
                    == modN(
                        12LL * output_position + 3
                    )
                || input_position
                    == modN(
                        12LL * output_position + 8
                    );

            if (coefficient) {
                xor_bit(
                    explicit_image,
                    output_position
                );
            }
        }

        if (!(formula == explicit_image)) {
            throw std::runtime_error(
                "Subterranean forward L self-test failed"
            );
        }
    }

    for (int output_position = 0;
         output_position < N;
         ++output_position) {
        for (int input_position = 0;
             input_position < N;
             ++input_position) {
            const bool expected =
                input_position
                    == modN(12LL * output_position)
                || input_position
                    == modN(
                        12LL * output_position + 3
                    )
                || input_position
                    == modN(
                        12LL * output_position + 8
                    );

            const bool obtained =
                input_position
                    == L_ROWS[output_position][0]
                || input_position
                    == L_ROWS[output_position][1]
                || input_position
                    == L_ROWS[output_position][2];

            if (expected != obtained) {
                throw std::runtime_error(
                    "Subterranean L^T self-test failed"
                );
            }
        }
    }

    std::cout
        << "  forward L and L^T rows: PASS\n"
        << std::flush;
}

static void run_optimized_chi_self_tests() {
    std::cout
        << "Subterranean-2.0 third-order optimized chi self-tests\n"
        << std::flush;

    verify_subterranean_linear_layer();

    const std::array<
        std::array<std::string, 3>,
        3
    > direction_triples = {{
        {{"0", "2", "37"}},
        {{"0", "32", "64"}},
        {{"0,7", "11,93", "37,128"}}
    }};

    for (const auto& directions :
         direction_triples) {
        const Mask d0 =
            parse_support(directions[0]);
        const Mask d1 =
            parse_support(directions[1]);
        const Mask d2 =
            parse_support(directions[2]);

        const Gamma initial =
            init_gamma_third_order(
                d0,
                d1,
                d2
            );

        const Gamma reference_first =
            pass_chi_third_order_reference_formula(
                initial,
                false
            );

        const Gamma optimized_first =
            pass_chi_third_order(
                initial,
                false
            );

        int worst_position = 0;
        int worst_coordinate = 0;

        const double first_difference =
            maximum_gamma_difference(
                reference_first,
                optimized_first,
                false,
                worst_position,
                worst_coordinate
            );

        std::cout
            << "  initial gamma D=["
            << directions[0]
            << "],["
            << directions[1]
            << "],["
            << directions[2]
            << "] max_diff="
            << std::scientific
            << std::setprecision(6)
            << first_difference
            << " at ("
            << worst_position
            << ","
            << worst_coordinate
            << ")\n";

        if (first_difference > 2e-12) {
            throw std::runtime_error(
                "optimized chi mismatch on initial gamma"
            );
        }

        const Gamma post_linear =
            pass_linear_third_order(
                reference_first
            );

        const Gamma reference_second =
            pass_chi_third_order_reference_formula(
                post_linear,
                false
            );

        const Gamma optimized_second =
            pass_chi_third_order(
                post_linear,
                false
            );

        const double second_difference =
            maximum_gamma_difference(
                reference_second,
                optimized_second,
                false,
                worst_position,
                worst_coordinate
            );

        std::cout
            << "  post-linear gamma max_diff="
            << std::scientific
            << std::setprecision(6)
            << second_difference
            << " at ("
            << worst_position
            << ","
            << worst_coordinate
            << ")\n";

        if (second_difference > 2e-12) {
            throw std::runtime_error(
                "optimized chi mismatch on post-linear gamma"
            );
        }

        const Gamma reference_final =
            pass_chi_third_order_reference_formula(
                post_linear,
                true
            );

        const Gamma optimized_final =
            pass_chi_third_order(
                post_linear,
                true
            );

        const double final_difference =
            maximum_gamma_difference(
                reference_final,
                optimized_final,
                true,
                worst_position,
                worst_coordinate
            );

        std::cout
            << "  final-coordinate-only max_diff="
            << std::scientific
            << std::setprecision(6)
            << final_difference
            << " at ("
            << worst_position
            << ","
            << worst_coordinate
            << ")\n";

        if (final_difference > 2e-12) {
            throw std::runtime_error(
                "optimized final-coordinate mismatch"
            );
        }
    }

    std::cout
        << "  scalar rank-one vs optimized rank-one: PASS\n"
        << std::flush;
}

#endif

static CorrVec tail_correlations_third_order_bitwise(
    const Mask& d0,
    const Mask& d1,
    const Mask& d2,
    int tail_rounds
) {
    Gamma gamma = init_gamma_third_order(d0, d1, d2);
    for (int r = 0; r < tail_rounds; ++r) {
        const bool last = (r == tail_rounds - 1);
        gamma = pass_chi_third_order(gamma, last);
        if (!last) gamma = pass_linear_third_order(gamma);
    }

    CorrVec out{};
    for (int i = 0; i < N; ++i) out[i] = gamma[i][FINAL_COORD];
    return out;
}

static CorrVec tail_correlations_third_order(
    const Mask& d0,
    const Mask& d1,
    const Mask& d2,
    int tail_rounds
) {
    return TRUE_BLOCK2_ENGINE.tail_correlations(d0, d1, d2, tail_rounds);
}

static std::pair<int, double> max_abs_position(const CorrVec& values) {
    int best_position = 0;
    double best_value = values[0];
    double best_abs = std::fabs(best_value);
    for (int i = 1; i < N; ++i) {
        const double a = std::fabs(values[i]);
        if (a > best_abs) {
            best_abs = a;
            best_position = i;
            best_value = values[i];
        }
    }
    return {best_position, best_value};
}

static double log2_abs(double x) {
    return x == 0.0 ? -INFINITY : std::log2(std::fabs(x));
}


// -------------------- route-focused experiment parameters --------------------
//
// This version intentionally uses a documented approximation in the exact
// prefix.  It does NOT enumerate every chi-output triple.  Instead it uses:
//   1. an exact probability cutoff applied before a Cartesian product;
//   2. a bounded low-weight beam for each individual chi output space;
//   3. a bounded beam for compatible D0/D1 pairs;
//   4. endpoint aggregation and a global front beam after every exact round;
//   5. a short geometric-tail proxy, followed by a full-tail Top-K evaluation.
//
// Increase the beam widths below and check convergence of the final log2(abs)
// value before using the result as a reported experimental value.

static constexpr int FIXED_THREADS = 16;
static constexpr int FIXED_ROUNDS = 5;
static constexpr int FIXED_FRONT_ROUNDS = 1;
static const std::string FIXED_INPUT_D0 = "0";
static const std::string FIXED_INPUT_D1 = "32";
static const std::string FIXED_INPUT_D2 = "254";



// Strict probability pruning.  Because |C_tail| <= 1, a route with prefix
// probability below 2^FRONT_LOG2_CUTOFF can contribute at most that amount.
static constexpr double FRONT_LOG2_CUTOFF = -40.0;

static constexpr std::size_t DIRECTION_TOP_K = 256;
static constexpr int DIRECTION_EXACT_RANK_LIMIT = 18;

static constexpr std::size_t PAIR_TOP_K = 8192;
static constexpr std::size_t PER_PARENT_TOP_K = 16384;
static constexpr std::size_t FRONT_BEAM_WIDTH = 2000000;

static constexpr std::size_t PROXY_CANDIDATE_TOP_K = 2000000;
static constexpr std::size_t FINAL_ROUTE_TOP_K = 2000000;
static constexpr int PROXY_TAIL_ROUNDS = 3;

// Structural score.  Smaller endpoint weights are usually more likely to
// retain a large geometric correlation after further rounds.
static constexpr double ENDPOINT_WEIGHT_PENALTY = 0.20;
static constexpr double CHI_OUTPUT_WEIGHT_PENALTY = 0.05;

static constexpr double FIXED_PROGRESS_EVERY = 10.0;
static const std::string FIXED_OUTPUT_JSON = "subterranean_third_order_true_block2_large_routes_5r_result.json";
static const std::string FIXED_ROUTE_OUTPUT = "subterranean_third_order_true_block2_selected_routes.txt";

static inline double front_probability_cutoff() {
    return std::isfinite(FRONT_LOG2_CUTOFF)
        ? std::exp2(FRONT_LOG2_CUTOFF)
        : 0.0;
}

static inline bool probability_survives(double p) {
    static const double cutoff = front_probability_cutoff();
    return p >= cutoff;
}

struct FrontTask {
    Mask d0;
    Mask d1;
    Mask d2;
    double probability = 0.0;
    double path_cost = 0.0;
    double structural_score = -INFINITY;
};

struct TripleKey {
    Mask d0;
    Mask d1;
    Mask d2;
    bool operator==(const TripleKey& o) const noexcept {
        return d0 == o.d0 && d1 == o.d1 && d2 == o.d2;
    }
};

struct TripleKeyHash {
    size_t operator()(const TripleKey& t) const noexcept {
        const MaskHash h;
        const size_t a = h(t.d0);
        const size_t b = h(t.d1);
        const size_t c = h(t.d2);
        size_t x = a;
        x ^= b + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
        x ^= c + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
        return x;
    }
};

struct DirectionCandidate {
    Mask chi_output;
    Mask next_difference;
    double probability = 0.0;
    double cost = 0.0;
};

struct PairCandidate {
    DirectionCandidate c0;
    DirectionCandidate c1;
    double cost = 0.0;
};

struct AggregatedRoute {
    double probability = 0.0;
    double best_path_cost = std::numeric_limits<double>::infinity();
};

struct RouteStatistics {
    long double raw_cartesian_routes = 0.0L;
    unsigned long long examined_routes = 0;
    unsigned long long pruned_conflicts = 0;
    unsigned long long pruned_probability_subtrees = 0;
};

struct RankedRoute {
    FrontTask task;
    double proxy_correlation = 0.0;
    int proxy_position = 0;
    double proxy_score = -INFINITY;
};

static inline double route_structural_score(
    double probability,
    double path_cost
) {
    if (probability <= 0.0) return -INFINITY;
    return std::log2(probability)
         - ENDPOINT_WEIGHT_PENALTY * path_cost;
}

static inline long double power_of_two_long_double(int rank) {
    return std::ldexp(1.0L, rank);
}

template <class T, class Better>
static void keep_best(std::vector<T>& values, std::size_t limit, Better better) {
    if (values.size() <= limit) {
        std::sort(values.begin(), values.end(), better);
        return;
    }
    std::nth_element(
        values.begin(),
        values.begin() + static_cast<std::ptrdiff_t>(limit),
        values.end(),
        better
    );
    values.resize(limit);
    std::sort(values.begin(), values.end(), better);
}

// Build the affine chi output space, but retain only the candidates with the
// lowest cheap structural cost.  For rank <= DIRECTION_EXACT_RANK_LIMIT the
// affine space is enumerated exactly before Top-K selection.  For larger rank,
// a bounded beam is applied after every basis vector.
static std::vector<DirectionCandidate> selected_chi_outputs(
    const Mask& delta,
    std::unordered_map<Mask, Mask, MaskHash>& l_cache,
    int& rank_out,
    long double& raw_output_count
) {
    Mask base;
    for (int i = 0; i < N; ++i) {
        const int di = test_bit(delta, i);
        const int d1 = test_bit(delta, i + 1);
        const int d2 = test_bit(delta, i + 2);
        if (di ^ d2 ^ (d1 & d2)) xor_bit(base, i);
    }

    std::vector<Mask> columns;
    columns.reserve(2 * popcount_mask(delta));
    for (int j = 0; j < N; ++j) {
        Mask col;
        if (test_bit(delta, j - 1)) xor_bit(col, j - 2);
        if (test_bit(delta, j + 1)) xor_bit(col, j - 1);
        if (!is_zero(col)) columns.push_back(col);
    }

    const std::vector<Mask> basis = gf2_independent_basis(columns);
    const int rank = static_cast<int>(basis.size());
    rank_out = rank;
    raw_output_count = power_of_two_long_double(rank);
    if (rank >= 63) {
        throw std::runtime_error("chi rank too large for double probability");
    }

    const double output_probability = std::ldexp(1.0, -rank);

    auto get_linear = [&](const Mask& x) {
        const auto it = l_cache.find(x);
        if (it != l_cache.end()) return it->second;
        const Mask y = L_transform(x);
        l_cache.emplace(x, y);
        return y;
    };

    std::vector<DirectionCandidate> beam;
    beam.reserve(DIRECTION_TOP_K * 2 + 2);
    const Mask base_next = get_linear(base);
    beam.push_back({
        base,
        base_next,
        output_probability,
        static_cast<double>(popcount_mask(base_next))
        + CHI_OUTPUT_WEIGHT_PENALTY
          * static_cast<double>(popcount_mask(base))
    });

    const bool exact_before_selection =
        rank <= DIRECTION_EXACT_RANK_LIMIT;

    for (const Mask& basis_vector : basis) {
        const Mask next_basis = get_linear(basis_vector);
        const std::size_t old_size = beam.size();
        beam.reserve(old_size * 2);
        for (std::size_t i = 0; i < old_size; ++i) {
            DirectionCandidate candidate = beam[i];
            candidate.chi_output =
                mask_xor(candidate.chi_output, basis_vector);
            candidate.next_difference =
                mask_xor(candidate.next_difference, next_basis);
            candidate.cost =
                static_cast<double>(popcount_mask(candidate.next_difference))
                + CHI_OUTPUT_WEIGHT_PENALTY
                  * static_cast<double>(popcount_mask(candidate.chi_output));
            beam.push_back(std::move(candidate));
        }

        if (!exact_before_selection && beam.size() > DIRECTION_TOP_K) {
            keep_best(
                beam,
                DIRECTION_TOP_K,
                [](const DirectionCandidate& a, const DirectionCandidate& b) {
                    if (a.cost != b.cost) return a.cost < b.cost;
                    return MaskHash{}(a.chi_output) < MaskHash{}(b.chi_output);
                }
            );
        }
    }

    keep_best(
        beam,
        DIRECTION_TOP_K,
        [](const DirectionCandidate& a, const DirectionCandidate& b) {
            if (a.cost != b.cost) return a.cost < b.cost;
            return MaskHash{}(a.chi_output) < MaskHash{}(b.chi_output);
        }
    );

    for (DirectionCandidate& candidate : beam) {
        candidate.probability = output_probability;
    }
    return beam;
}

static std::vector<FrontTask> aggregate_and_trim_routes(
    const std::vector<FrontTask>& routes,
    std::size_t limit
) {
    std::unordered_map<TripleKey, AggregatedRoute, TripleKeyHash> aggregate;
    aggregate.reserve(routes.size() * 2 + 16);

    for (const FrontTask& route : routes) {
        AggregatedRoute& value =
            aggregate[{route.d0, route.d1, route.d2}];
        value.probability += route.probability;
        value.best_path_cost =
            std::min(value.best_path_cost, route.path_cost);
    }

    std::vector<FrontTask> compact;
    compact.reserve(aggregate.size());
    for (const auto& item : aggregate) {
        FrontTask route;
        route.d0 = item.first.d0;
        route.d1 = item.first.d1;
        route.d2 = item.first.d2;
        route.probability = item.second.probability;
        route.path_cost = item.second.best_path_cost;
        route.structural_score = route_structural_score(
            route.probability,
            route.path_cost
        );
        compact.push_back(route);
    }

    keep_best(
        compact,
        limit,
        [](const FrontTask& a, const FrontTask& b) {
            if (a.structural_score != b.structural_score) {
                return a.structural_score > b.structural_score;
            }
            return a.probability > b.probability;
        }
    );
    return compact;
}

static std::vector<FrontTask> expand_front_beam_one_round(
    const std::vector<FrontTask>& parents,
    RouteStatistics& statistics,
    std::unordered_map<Mask, std::vector<DirectionCandidate>, MaskHash>& direction_cache,
    std::unordered_map<Mask, Mask, MaskHash>& l_cache
) {
    std::vector<FrontTask> all_children;
    all_children.reserve(parents.size() * PER_PARENT_TOP_K);

    auto get_candidates = [&](const Mask& delta) -> const std::vector<DirectionCandidate>& {
        const auto it = direction_cache.find(delta);
        if (it != direction_cache.end()) return it->second;
        int rank = 0;
        long double raw_count = 0.0L;
        std::vector<DirectionCandidate> selected =
            selected_chi_outputs(delta, l_cache, rank, raw_count);
        return direction_cache.emplace(delta, std::move(selected)).first->second;
    };

    for (std::size_t parent_index = 0;
         parent_index < parents.size();
         ++parent_index) {
        const FrontTask& parent = parents[parent_index];
        const auto& outputs0 = get_candidates(parent.d0);
        const auto& outputs1 = get_candidates(parent.d1);
        const auto& outputs2 = get_candidates(parent.d2);

        // The retained output vectors all preserve the exact per-output chi
        // probability, even though only a subset of outputs is represented.
        const double child_probability =
            parent.probability
            * outputs0.front().probability
            * outputs1.front().probability
            * outputs2.front().probability;

        // Recover raw affine-space sizes from p=2^-rank.
        const int rank0 = static_cast<int>(std::llround(-std::log2(outputs0.front().probability)));
        const int rank1 = static_cast<int>(std::llround(-std::log2(outputs1.front().probability)));
        const int rank2 = static_cast<int>(std::llround(-std::log2(outputs2.front().probability)));
        statistics.raw_cartesian_routes +=
            power_of_two_long_double(rank0 + rank1 + rank2);

        if (!probability_survives(child_probability)) {
            ++statistics.pruned_probability_subtrees;
            continue;
        }

        std::vector<PairCandidate> pairs;
        pairs.reserve(outputs0.size() * outputs1.size());
        for (const DirectionCandidate& c0 : outputs0) {
            for (const DirectionCandidate& c1 : outputs1) {
                if (masks_intersect(c0.chi_output, c1.chi_output)) {
                    statistics.pruned_conflicts +=
                        static_cast<unsigned long long>(outputs2.size());
                    continue;
                }
                pairs.push_back({c0, c1, c0.cost + c1.cost});
            }
        }

        keep_best(
            pairs,
            PAIR_TOP_K,
            [](const PairCandidate& a, const PairCandidate& b) {
                return a.cost < b.cost;
            }
        );

        std::vector<FrontTask> local_children;
        local_children.reserve(
            std::min<std::size_t>(
                PER_PARENT_TOP_K * 4,
                pairs.size() * outputs2.size()
            )
        );

        for (const PairCandidate& pair : pairs) {
            for (const DirectionCandidate& c2 : outputs2) {
                ++statistics.examined_routes;
                if (masks_intersect(pair.c0.chi_output, c2.chi_output)
                    || masks_intersect(pair.c1.chi_output, c2.chi_output)) {
                    ++statistics.pruned_conflicts;
                    continue;
                }

                FrontTask child;
                child.d0 = pair.c0.next_difference;
                child.d1 = pair.c1.next_difference;
                child.d2 = c2.next_difference;
                child.probability = child_probability;
                child.path_cost =
                    parent.path_cost + pair.cost + c2.cost;
                child.structural_score = route_structural_score(
                    child.probability,
                    child.path_cost
                );
                local_children.push_back(std::move(child));
            }
        }

        keep_best(
            local_children,
            PER_PARENT_TOP_K,
            [](const FrontTask& a, const FrontTask& b) {
                return a.structural_score > b.structural_score;
            }
        );

        all_children.insert(
            all_children.end(),
            local_children.begin(),
            local_children.end()
        );

        if ((parent_index + 1) % 8 == 0 || parent_index + 1 == parents.size()) {
            std::cout
                << "    expanded parents " << (parent_index + 1)
                << "/" << parents.size()
                << ", retained-before-merge=" << all_children.size()
                << "\n" << std::flush;
        }
    }

    return aggregate_and_trim_routes(all_children, FRONT_BEAM_WIDTH);
}

class PhaseMonitor {
public:
    PhaseMonitor(
        const std::string& phase,
        std::atomic<std::size_t>& completed,
        std::size_t total
    ) : phase_(phase), completed_(completed), total_(total) {
        worker_ = std::thread([this]() {
            std::unique_lock<std::mutex> lock(mutex_);
            while (!stopped_) {
                if (condition_.wait_for(
                        lock,
                        std::chrono::milliseconds(
                            static_cast<int>(FIXED_PROGRESS_EVERY * 1000.0)
                        ),
                        [this]() { return stopped_; }
                    )) {
                    break;
                }
                std::cout
                    << "  " << phase_ << " progress: "
                    << completed_.load(std::memory_order_relaxed)
                    << "/" << total_ << "\n" << std::flush;
            }
        });
    }

    ~PhaseMonitor() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        condition_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

private:
    std::string phase_;
    std::atomic<std::size_t>& completed_;
    std::size_t total_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool stopped_ = false;
    std::thread worker_;
};

static CorrVec evaluate_route_tail(
    const Mask& d0,
    const Mask& d1,
    const Mask& d2,
    int tail_rounds
) {
    return tail_correlations_third_order(d0, d1, d2, tail_rounds);
}


static void write_selected_routes(
    const std::vector<RankedRoute>& routes
) {
    std::ofstream out(FIXED_ROUTE_OUTPUT);
    if (!out) {
        throw std::runtime_error("cannot open selected-route output file");
    }
    out << std::setprecision(17);
    out << "rank probability log2_probability structural_score "
           "proxy_position proxy_correlation proxy_score D0 D1 D2\n";
    for (std::size_t i = 0; i < routes.size(); ++i) {
        const RankedRoute& route = routes[i];
        out << (i + 1) << " "
            << route.task.probability << " "
            << std::log2(route.task.probability) << " "
            << route.task.structural_score << " "
            << route.proxy_position << " "
            << route.proxy_correlation << " "
            << route.proxy_score << " "
            << fmt_support(route.task.d0, 257) << " "
            << fmt_support(route.task.d1, 257) << " "
            << fmt_support(route.task.d2, 257) << "\n";
    }
}

int main() {
    try {
        std::ios::sync_with_stdio(false);
#ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(FIXED_THREADS);
        omp_set_max_active_levels(1);
#endif
        init_tables();
        init_fast_chi_tables();
        TRUE_BLOCK2_ENGINE.self_test();

        const Mask input0 = parse_support(FIXED_INPUT_D0);
        const Mask input1 = parse_support(FIXED_INPUT_D1);
        const Mask input2 = parse_support(FIXED_INPUT_D2);
        const int tail_rounds = FIXED_ROUNDS - FIXED_FRONT_ROUNDS;
        if (FIXED_FRONT_ROUNDS < 0 || tail_rounds <= 0) {
            throw std::runtime_error(
                "FIXED_ROUNDS must be greater than FIXED_FRONT_ROUNDS"
            );
        }

        std::cout
            << "Subterranean-2.0 third-order route-focused hybrid DL estimation\n"
            << "  input d0                 : " << fmt_support(input0) << "\n"
            << "  input d1                 : " << fmt_support(input1) << "\n"
            << "  input d2                 : " << fmt_support(input2) << "\n"
            << "  total chi layers         : " << FIXED_ROUNDS << "\n"
            << "  exact front chi/L steps  : " << FIXED_FRONT_ROUNDS << "\n"
            << "  geometric tail layers    : " << tail_rounds << "\n"
            << "  true enlarged block size : 2 (256^2 coordinates)\n"
            << "  direction Top-K          : " << DIRECTION_TOP_K << "\n"
            << "  D0/D1 pair Top-K         : " << PAIR_TOP_K << "\n"
            << "  per-parent Top-K         : " << PER_PARENT_TOP_K << "\n"
            << "  front beam width         : " << FRONT_BEAM_WIDTH << "\n"
            << "  proxy candidate Top-K    : " << PROXY_CANDIDATE_TOP_K << "\n"
            << "  full-tail route Top-K    : " << FINAL_ROUTE_TOP_K << "\n"
            << "  proxy tail layers        : " << PROXY_TAIL_ROUNDS << "\n"
            << "  front log2 cutoff        : " << FRONT_LOG2_CUTOFF << "\n"
            << "  WARNING                  : route selection is heuristic; "
               "increase Top-K values and check convergence\n"
            << std::flush;

        const auto start_time = std::chrono::steady_clock::now();

        RouteStatistics statistics;
        std::unordered_map<Mask, std::vector<DirectionCandidate>, MaskHash>
            direction_cache;
        std::unordered_map<Mask, Mask, MaskHash> l_cache;
        direction_cache.reserve(4096);
        l_cache.reserve(65536);

        std::vector<FrontTask> beam;
        beam.push_back({input0, input1, input2, 1.0, 0.0, 0.0});

        for (int round = 0; round < FIXED_FRONT_ROUNDS; ++round) {
            const double mass_before = [&]() {
                double m = 0.0;
                for (const FrontTask& route : beam) m += route.probability;
                return m;
            }();

            std::cout
                << "\n  exact-front beam round " << (round + 1)
                << ": parents=" << beam.size()
                << ", represented_mass=" << std::setprecision(17)
                << mass_before << "\n" << std::flush;

            beam = expand_front_beam_one_round(
                beam,
                statistics,
                direction_cache,
                l_cache
            );

            double mass_after = 0.0;
            for (const FrontTask& route : beam) {
                mass_after += route.probability;
            }

            std::cout
                << "  round " << (round + 1)
                << " retained endpoints=" << beam.size()
                << ", represented_mass=" << std::setprecision(17)
                << mass_after << "\n" << std::flush;

            if (beam.empty()) {
                throw std::runtime_error(
                    "all front routes were removed; relax cutoff/beam controls"
                );
            }
        }

        if (FIXED_FRONT_ROUNDS == 0) {
            beam[0].structural_score = 0.0;
        }

        keep_best(
            beam,
            std::min<std::size_t>(PROXY_CANDIDATE_TOP_K, beam.size()),
            [](const FrontTask& a, const FrontTask& b) {
                return a.structural_score > b.structural_score;
            }
        );

        const int proxy_rounds = std::min(PROXY_TAIL_ROUNDS, tail_rounds);
        std::vector<RankedRoute> ranked(beam.size());
        std::atomic<std::size_t> proxy_done{0};
        {
            PhaseMonitor monitor("proxy", proxy_done, ranked.size());
#pragma omp parallel for schedule(dynamic, 1)
            for (std::size_t i = 0; i < beam.size(); ++i) {
                const CorrVec proxy = tail_correlations_third_order_bitwise(
                    beam[i].d0,
                    beam[i].d1,
                    beam[i].d2,
                    proxy_rounds
                );
                const auto best = max_abs_position(proxy);
                RankedRoute route;
                route.task = beam[i];
                route.proxy_position = best.first;
                route.proxy_correlation = best.second;
                route.proxy_score =
                    route.task.probability > 0.0 && best.second != 0.0
                    ? std::log2(route.task.probability)
                      + std::log2(std::fabs(best.second))
                    : -INFINITY;
                ranked[i] = std::move(route);
                proxy_done.fetch_add(1, std::memory_order_relaxed);
            }
        }

        keep_best(
            ranked,
            std::min<std::size_t>(FINAL_ROUTE_TOP_K, ranked.size()),
            [](const RankedRoute& a, const RankedRoute& b) {
                if (a.proxy_score != b.proxy_score) {
                    return a.proxy_score > b.proxy_score;
                }
                return a.task.structural_score > b.task.structural_score;
            }
        );

        write_selected_routes(ranked);

        CorrVec global_total{};
        CorrVec global_abs{};
        double selected_mass = 0.0;
        for (const RankedRoute& route : ranked) {
            selected_mass += route.task.probability;
        }

        std::atomic<std::size_t> full_done{0};
        {
            PhaseMonitor monitor("full tail", full_done, ranked.size());
            for (std::size_t i = 0; i < ranked.size(); ++i) {
                const RankedRoute& route = ranked[i];
                const CorrVec correlations = evaluate_route_tail(
                    route.task.d0,
                    route.task.d1,
                    route.task.d2,
                    tail_rounds
                );
                for (int position = 0; position < N; ++position) {
                    const double contribution =
                        route.task.probability * correlations[position];
                    global_total[position] += contribution;
                    global_abs[position] += std::fabs(contribution);
                }
                full_done.fetch_add(1, std::memory_order_relaxed);
            }
        }

        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time
        ).count();

        std::vector<int> order(N);
        for (int i = 0; i < N; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return std::fabs(global_total[a]) > std::fabs(global_total[b]);
        });

        double front_beam_mass = 0.0;
        for (const FrontTask& route : beam) front_beam_mass += route.probability;

        std::cout
            << "\nresult\n"
            << "  raw front Cartesian routes: "
            << std::scientific << std::setprecision(6)
            << statistics.raw_cartesian_routes << "\n"
            << "  actually examined routes  : "
            << statistics.examined_routes << "\n"
            << "  pruned conflicts          : "
            << statistics.pruned_conflicts << "\n"
            << "  probability subtrees cut  : "
            << statistics.pruned_probability_subtrees << "\n"
            << "  front beam mass           : "
            << std::setprecision(17) << front_beam_mass << "\n"
            << "  selected route mass       : "
            << selected_mass << "\n"
            << "  full tail evaluations     : "
            << ranked.size() << "\n"
            << "  elapsed seconds           : "
            << std::fixed << std::setprecision(3) << elapsed << "\n";

        std::cout << "\nTop 20 single-bit output masks\n";
        std::cout << "rank  position  correlation              log2(abs)       sum_abs\n";
        for (int rank = 0; rank < 20; ++rank) {
            const int position = order[rank];
            std::cout
                << std::setw(4) << (rank + 1)
                << "  " << std::setw(8) << position
                << "  " << std::scientific << std::setprecision(12)
                << global_total[position]
                << "  " << std::fixed << std::setprecision(9)
                << log2_abs(global_total[position])
                << "  " << std::scientific << std::setprecision(12)
                << global_abs[position]
                << std::defaultfloat << "\n";
        }

        std::cout << "\nPaper target [142]\n"
                  << "  correlation              : " << std::scientific
                  << std::setprecision(17) << global_total[142] << "\n"
                  << "  log2(abs)                : " << std::fixed
                  << std::setprecision(12) << log2_abs(global_total[142]) << "\n"
                  << "  expected                 : approximately -6.87\n"
                  << std::defaultfloat;

        std::ofstream json(FIXED_OUTPUT_JSON);
        if (!json) throw std::runtime_error("cannot open output JSON file");
        json << std::setprecision(17);
        json << "{\n";
        json << "  \"cipher\": \"Subterranean-2.0\",\n";
        json << "  \"order\": 3,\n";
        json << "  \"route_selection\": \"beam plus short-tail proxy\",\n";
        json << "  \"rounds\": " << FIXED_ROUNDS << ",\n";
        json << "  \"front_rounds\": " << FIXED_FRONT_ROUNDS << ",\n";
        json << "  \"tail_rounds\": " << tail_rounds << ",\n";
        json << "  \"front_beam_width\": " << FRONT_BEAM_WIDTH << ",\n";
        json << "  \"proxy_candidate_top_k\": "
             << PROXY_CANDIDATE_TOP_K << ",\n";
        json << "  \"final_route_top_k\": " << FINAL_ROUTE_TOP_K << ",\n";
        json << "  \"front_beam_mass\": " << front_beam_mass << ",\n";
        json << "  \"selected_route_mass\": " << selected_mass << ",\n";
        json << "  \"elapsed_seconds\": " << elapsed << ",\n";
        json << "  \"single_bit_results\": [\n";
        for (int position = 0; position < N; ++position) {
            json << "    {\"position\": " << position
                 << ", \"correlation\": " << global_total[position]
                 << ", \"log2_abs\": ";
            if (global_total[position] == 0.0) json << "null";
            else json << log2_abs(global_total[position]);
            json << ", \"sum_abs\": " << global_abs[position] << "}";
            if (position + 1 != N) json << ",";
            json << "\n";
        }
        json << "  ]\n";
        json << "}\n";

        std::cout
            << "\n  wrote JSON               : " << FIXED_OUTPUT_JSON << "\n"
            << "  wrote selected routes    : " << FIXED_ROUTE_OUTPUT << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
