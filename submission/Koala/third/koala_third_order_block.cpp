// koala_third_order_true_block2_strict.cpp
//
// Koala-p third-order hybrid differential-linear evaluation with a true
// BLOCK_SIZE=2 enlarged local unit.
//
// Construction:
//   - exact front: independently enumerate the three chi differential
//     distributions, keep only pairwise-disjoint chi outputs, then apply L;
//   - geometric tail: 8 cube vertices and 256 Walsh coordinates;
//   - final third-order HDL coordinate: 127 = 0b01111111;
//   - accumulate all 257 one-bit output masks simultaneously.
//
// The third-order chi pass is NOT implemented with a 3x256^4 LOCAL table
// (that table would require about 103 GiB).  Instead, the same transfer
// matrices are evaluated by an exact Walsh/rank-one reduction.  This is
// algebraically equivalent to the direct 256x256 matrix-chain formula.
//
// Default experiment:
//   input directions : D0={0}, D1={32}, D2={64}
//   total chi layers : 5
//   exact front      : 1 chi/L step
//   geometric tail  : 4 chi layers with three intervening L layers
//
// Change the FIXED_* constants near the end of the file for another test.
// For example, a 2+5 experiment uses FIXED_ROUNDS=7 and
// FIXED_FRONT_ROUNDS=2, but it can create a very large exact-front tree.
//
// Compile:
//   g++ -O3 -march=native -std=c++17 -fopenmp koala_third_order_true_block2_strict.cpp -o koala_3rd_block2 -lm
//
// Run:
//   ./koala_3rd_block2

#ifndef KOALA_THIRD_ORDER_SELF_TEST_ONLY
#define KOALA_THIRD_ORDER_SELF_TEST_ONLY 0
#endif

#ifndef KOALA_DENSE_BLOCK2
#define KOALA_DENSE_BLOCK2 1
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <immintrin.h>
#include <malloc.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static constexpr int N = 257;
static constexpr int W = 5;
static constexpr int ORDER = 3;
static constexpr int CUBE = 1 << ORDER;          // 8 cube vertices
static constexpr int DIM = 1 << CUBE;            // 256 Walsh coordinates
static constexpr int FINAL_COORD = 127;           // 0b01111111
static constexpr int INV121 = 17;                 // 121^{-1} mod 257

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

// Forward differential propagation through Koala-p L = theta o pi.
static Mask L_transform(const Mask& in) {
    Mask out;
    for (int p : active_bits(in)) {
        const int base = modN(static_cast<long long>(INV121) * p);
        xor_bit(out, base);
        xor_bit(out, base - 3);
        xor_bit(out, base - 10);
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

// Koala-p iota_j toggles state bit 0 unless j is 2, 5, or 6.
static inline bool koala_iota_is_active(int round_index) noexcept {
    return round_index != 2 && round_index != 5 && round_index != 6;
}

// True BLOCK_SIZE=2 enlarged-unit engine for third-order Koala-p.
// Requires the including translation unit to define:
//   N=257, DIM=256, FINAL_COORD=127, Mask, test_bit(), modN(), CorrVec.
#ifndef KOALA_THIRD_ORDER_TRUE_BLOCK2_ENGINE_HPP
#define KOALA_THIRD_ORDER_TRUE_BLOCK2_ENGINE_HPP

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
        int tail_rounds,
        bool include_iota,
        int first_chi_round_index
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
            if (include_iota) {
                apply_iota_to_factors(
                    factors,
                    first_chi_round_index + round + 1
                );
            }
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
                    modN(121LL * output_position),
                    modN(121LL * (output_position + 3)),
                    modN(121LL * (output_position + 10))
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

    void apply_iota_to_factors(
        Factors& correlations,
        int round_index
    ) const {
        if (!koala_iota_is_active(round_index)) return;
        // Position 0 is local position 0 of block 0.  Toggling the common
        // base value x multiplies a Walsh coordinate by (-1) when bit 7 of
        // that local 8-bit coordinate is active.
        Factor& block0 = correlations[0];
        for (int coordinate = 0;
             coordinate < static_cast<int>(block0.size());
             ++coordinate) {
            if ((coordinate & 0x80) != 0) {
                block0[coordinate] = -block0[coordinate];
            }
        }
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
                        modN(121LL * output_position),
                        modN(121LL * (output_position + 3)),
                        modN(121LL * (output_position + 10))
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

static void init_tables() {
    for (int i = 0; i < N; ++i) {
        // Pullback of one output mask bit through L:
        // L^T e_i = e_{121i} + e_{121(i+3)} + e_{121(i+10)}.
        L_ROWS[i] = {
            modN(121LL * i),
            modN(121LL * (i + 3)),
            modN(121LL * (i + 10))
        };
    }
}


static void verify_koala_block2_linear_model() {
    using BitState = std::array<uint8_t, N>;

    auto explicit_linear = [](const BitState& input) {
        BitState after_pi{};
        BitState output{};
        for (int i = 0; i < N; ++i) {
            after_pi[i] = input[modN(121LL * i)];
        }
        for (int i = 0; i < N; ++i) {
            output[i] =
                after_pi[i]
                ^ after_pi[(i + 3) % N]
                ^ after_pi[(i + 10) % N];
        }
        return output;
    };

    for (int input_position = 0; input_position < N; ++input_position) {
        BitState basis{};
        basis[input_position] = 1;
        const BitState explicit_output = explicit_linear(basis);

        Mask explicit_mask;
        for (int output_position = 0; output_position < N; ++output_position) {
            if (explicit_output[output_position]) {
                xor_bit(explicit_mask, output_position);
            }
        }

        Mask basis_mask;
        xor_bit(basis_mask, input_position);
        if (!(explicit_mask == L_transform(basis_mask))) {
            throw std::runtime_error(
                "Koala-p forward linear self-test failed at input bit "
                + std::to_string(input_position)
            );
        }
    }

    for (int output_position = 0; output_position < N; ++output_position) {
        const std::array<int, 3> expected = {
            modN(121LL * output_position),
            modN(121LL * (output_position + 3)),
            modN(121LL * (output_position + 10))
        };
        if (L_ROWS[output_position] != expected) {
            throw std::runtime_error(
                "Koala-p mask-pullback row self-test failed at output bit "
                + std::to_string(output_position)
            );
        }
    }

    // For every complete contiguous 2-bit target block, verify that the
    // source blocks of its two output positions are disjoint.  Therefore the
    // dense 65536-coordinate block factor re-factorizes exactly after Koala L.
    for (int block = 0; block < 128; ++block) {
        const int p0 = 2 * block;
        const int p1 = p0 + 1;
        std::array<int, 3> source_blocks0{};
        std::array<int, 3> source_blocks1{};
        for (int term = 0; term < 3; ++term) {
            source_blocks0[term] = L_ROWS[p0][term] / 2;
            source_blocks1[term] = L_ROWS[p1][term] / 2;
        }
        for (int a : source_blocks0) {
            for (int b : source_blocks1) {
                if (a == b) {
                    throw std::runtime_error(
                        "Koala-p block2 exact-refactorization test failed"
                    );
                }
            }
        }
    }

    std::cout
        << "  Koala forward L          : PASS\n"
        << "  Koala L^T rows           : PASS\n"
        << "  block2 L refactorization : PASS\n"
        << std::flush;
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
static Gamma pass_chi_third_order(const Gamma& gamma, bool only_final_coord) {
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

static void apply_koala_iota_to_gamma(
    Gamma& gamma,
    int round_index
) {
    if (!koala_iota_is_active(round_index)) return;
    for (int coordinate = 0; coordinate < DIM; ++coordinate) {
        if ((coordinate & 0x80) != 0) {
            gamma[0][coordinate] = -gamma[0][coordinate];
        }
    }
}

static CorrVec tail_correlations_third_order_bitwise(
    const Mask& d0,
    const Mask& d1,
    const Mask& d2,
    int tail_rounds,
    bool include_iota,
    int first_chi_round_index
) {
    Gamma gamma = init_gamma_third_order(d0, d1, d2);
    for (int r = 0; r < tail_rounds; ++r) {
        const bool last = (r == tail_rounds - 1);
        gamma = pass_chi_third_order(gamma, last);
        if (!last) {
            gamma = pass_linear_third_order(gamma);
            if (include_iota) {
                apply_koala_iota_to_gamma(
                    gamma,
                    first_chi_round_index + r + 1
                );
            }
        }
    }

    CorrVec out{};
    for (int i = 0; i < N; ++i) out[i] = gamma[i][FINAL_COORD];
    return out;
}

static CorrVec tail_correlations_third_order(
    const Mask& d0,
    const Mask& d1,
    const Mask& d2,
    int tail_rounds,
    bool include_iota,
    int first_chi_round_index
) {
    #if KOALA_DENSE_BLOCK2
    return TRUE_BLOCK2_ENGINE.tail_correlations(
        d0, d1, d2, tail_rounds, include_iota, first_chi_round_index
    );
#else
    // Exact fast path for contiguous BLOCK_SIZE=2: after every Koala-p L,
    // the two positions of each target block depend on disjoint source blocks.
    // Therefore the enlarged factor re-factorizes exactly, and all one-bit
    // output correlations equal the bit-wise rank-one implementation.
    return tail_correlations_third_order_bitwise(
        d0, d1, d2, tail_rounds, include_iota, first_chi_round_index
    );
#endif
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

// -------------------- fixed experiment parameters --------------------

static constexpr int FIXED_THREADS = 16;
static constexpr int FIXED_ROUNDS = 7;
static constexpr int FIXED_FRONT_ROUNDS = 2;
static const std::string FIXED_INPUT_D0 = "0";
static const std::string FIXED_INPUT_D1 = "11";
static const std::string FIXED_INPUT_D2 = "64";
// false reproduces the constant-free geometric model used by the original code.
static constexpr bool FIXED_INCLUDE_IOTA = false;
static constexpr int FIXED_FIRST_CHI_ROUND_INDEX = 0;
// Set to -1 to print only the global Top-20 outputs.
static constexpr int FIXED_TARGET_POSITION = -1;
static constexpr double FRONT_LOG2_CUTOFF = -100.0;
static constexpr double FIXED_PROGRESS_EVERY = 10.0;
static const std::string FIXED_OUTPUT_JSON =
    "koala_third_order_true_block2_strict_result.json";

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
    double probability;
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

static void flush_local_to_global(
    CorrVec& local_total,
    CorrVec& local_abs,
    double& local_mass,
    unsigned long long& local_count,
    CorrVec& global_total,
    CorrVec& global_abs,
    double& global_mass,
    unsigned long long& global_count
) {
    if (local_count == 0) return;
#pragma omp critical(streaming_accumulate)
    {
        for (int i = 0; i < N; ++i) {
            global_total[i] += local_total[i];
            global_abs[i] += local_abs[i];
        }
        global_mass += local_mass;
        global_count += local_count;
    }
    local_total.fill(0.0);
    local_abs.fill(0.0);
    local_mass = 0.0;
    local_count = 0;
}

static void dfs_exact_front(
    int depth,
    const Mask& d0,
    const Mask& d1,
    const Mask& d2,
    double probability,
    int front_rounds,
    int tail_rounds,
    std::unordered_map<Mask, std::vector<std::pair<Mask, double>>, MaskHash>& chi_cache,
    std::unordered_map<Mask, Mask, MaskHash>& l_cache,
    std::atomic<unsigned long long>& expanded_nodes,
    std::atomic<unsigned long long>& pruned_conflicts,
    std::atomic<unsigned long long>& pruned_probabilities,
    std::atomic<unsigned long long>& tail_evaluations,
    CorrVec& global_total,
    CorrVec& global_abs,
    double& global_mass,
    unsigned long long& global_count,
    CorrVec& local_total,
    CorrVec& local_abs,
    double& local_mass,
    unsigned long long& local_count
) {
    if (depth == front_rounds) {
        const CorrVec correlations =
            tail_correlations_third_order(
                d0, d1, d2, tail_rounds,
                FIXED_INCLUDE_IOTA,
                FIXED_FIRST_CHI_ROUND_INDEX + front_rounds
            );
        for (int i = 0; i < N; ++i) {
            const double contribution = probability * correlations[i];
            local_total[i] += contribution;
            local_abs[i] += std::fabs(contribution);
        }
        local_mass += probability;
        ++local_count;
        ++tail_evaluations;

        // Commit every completed endpoint so the monitor reports exact progress.
        flush_local_to_global(
            local_total, local_abs, local_mass, local_count,
            global_total, global_abs, global_mass, global_count
        );
        return;
    }

    const auto& chi0 = chi_cached_ref(d0, chi_cache);
    const auto& chi1 = chi_cached_ref(d1, chi_cache);
    const auto& chi2 = chi_cached_ref(d2, chi_cache);

    for (const auto& c0 : chi0) {
        const double p0 = probability * c0.second;
        if (!probability_survives(p0)) {
            ++pruned_probabilities;
            continue;
        }
        for (const auto& c1 : chi1) {
            const double p01 = p0 * c1.second;
            if (!probability_survives(p01)) {
                ++pruned_probabilities;
                continue;
            }
            for (const auto& c2 : chi2) {
                ++expanded_nodes;
                const double next_probability = p01 * c2.second;
                if (!probability_survives(next_probability)) {
                    ++pruned_probabilities;
                    continue;
                }

                // Match the supplied second-order rule:
                // reject pairwise intersections at chi output only.
                if (triple_pairwise_intersects(c0.first, c1.first, c2.first)) {
                    ++pruned_conflicts;
                    continue;
                }

                const Mask next0 = L_cached(c0.first, l_cache);
                const Mask next1 = L_cached(c1.first, l_cache);
                const Mask next2 = L_cached(c2.first, l_cache);

                dfs_exact_front(
                    depth + 1,
                    next0, next1, next2,
                    next_probability,
                    front_rounds, tail_rounds,
                    chi_cache, l_cache,
                    expanded_nodes, pruned_conflicts,
                    pruned_probabilities, tail_evaluations,
                    global_total, global_abs, global_mass, global_count,
                    local_total, local_abs, local_mass, local_count
                );
            }
        }
    }
}

int main() {
    try {
        std::ios::sync_with_stdio(false);
        _mm_setcsr(_mm_getcsr() | 0x8040); // FTZ + DAZ for tiny round-based terms
#ifdef _OPENMP
        omp_set_num_threads(FIXED_THREADS);
        omp_set_max_active_levels(1);
#endif
        init_tables();
        verify_koala_block2_linear_model();
        TRUE_BLOCK2_ENGINE.self_test();
#if KOALA_THIRD_ORDER_SELF_TEST_ONLY
        std::cout << "self-test-only mode finished\n";
        return 0;
#endif

        const Mask input0 = parse_support(FIXED_INPUT_D0);
        const Mask input1 = parse_support(FIXED_INPUT_D1);
        const Mask input2 = parse_support(FIXED_INPUT_D2);
        const int tail_rounds = FIXED_ROUNDS - FIXED_FRONT_ROUNDS;
        if (tail_rounds <= 0) {
            throw std::runtime_error(
                "FIXED_ROUNDS must be greater than FIXED_FRONT_ROUNDS"
            );
        }

        std::cout << "Koala-p third-order true BLOCK_SIZE=2 streaming hybrid DL estimation\n";
        std::cout << "  input d0                 : " << fmt_support(input0) << "\n";
        std::cout << "  input d1                 : " << fmt_support(input1) << "\n";
        std::cout << "  input d2                 : " << fmt_support(input2) << "\n";
        std::cout << "  total chi layers         : " << FIXED_ROUNDS << "\n";
        std::cout << "  exact front chi/L steps  : " << FIXED_FRONT_ROUNDS << "\n";
        std::cout << "  geometric tail layers    : " << tail_rounds << "\n";
        std::cout << "  workers                  : " << FIXED_THREADS << "\n";
        std::cout << "  geometric dimension      : " << DIM << "\n";
        std::cout << "  true enlarged block size : 2 (256^2 coordinates)\n";
        std::cout << "  block2 tail mode         : "
                  << (KOALA_DENSE_BLOCK2
                      ? "dense 65536-coordinate"
                      : "exact refactorization fast path")
                  << "\n"; 
        std::cout << "  include Koala iota       : "
                  << (FIXED_INCLUDE_IOTA ? "true" : "false") << "\n";
        std::cout << "  final HDL coordinate     : " << FINAL_COORD << "\n";
        std::cout << "  outputs accumulated      : all 257 single-bit masks\n";
        std::cout << "  conflict mode            : pairwise D0/D1/D2 chi-output overlap only\n";
        std::cout << "  front log2 cutoff        : " << FRONT_LOG2_CUTOFF << "\n";
        std::cout << "  progress interval        : " << FIXED_PROGRESS_EVERY << "s\n"
                  << std::flush;

        std::atomic<unsigned long long> expanded_nodes{0};
        std::atomic<unsigned long long> pruned_conflicts{0};
        std::atomic<unsigned long long> pruned_probabilities{0};
        std::atomic<unsigned long long> tail_evaluations{0};
        std::atomic<bool> monitor_stop{false};

        CorrVec global_total{};
        CorrVec global_abs{};
        double global_mass = 0.0;
        unsigned long long global_count = 0;

        const auto start_time = std::chrono::steady_clock::now();

        // Build one exact chi/L layer as the OpenMP task frontier.
        std::vector<FrontTask> tasks;
        if (FIXED_FRONT_ROUNDS == 0) {
            tasks.push_back({input0, input1, input2, 1.0});
        } else {
            std::unordered_map<Mask, std::vector<std::pair<Mask, double>>, MaskHash>
                seed_chi_cache;
            std::unordered_map<Mask, Mask, MaskHash> seed_l_cache;
            const auto& chi0 = chi_cached_ref(input0, seed_chi_cache);
            const auto& chi1 = chi_cached_ref(input1, seed_chi_cache);
            const auto& chi2 = chi_cached_ref(input2, seed_chi_cache);

            for (const auto& c0 : chi0) {
                const double p0 = c0.second;
                if (!probability_survives(p0)) {
                    ++pruned_probabilities;
                    continue;
                }
                for (const auto& c1 : chi1) {
                    const double p01 = p0 * c1.second;
                    if (!probability_survives(p01)) {
                        ++pruned_probabilities;
                        continue;
                    }
                    for (const auto& c2 : chi2) {
                        ++expanded_nodes;
                        const double probability = p01 * c2.second;
                        if (!probability_survives(probability)) {
                            ++pruned_probabilities;
                            continue;
                        }
                        if (triple_pairwise_intersects(
                                c0.first, c1.first, c2.first)) {
                            ++pruned_conflicts;
                            continue;
                        }
                        tasks.push_back({
                            L_cached(c0.first, seed_l_cache),
                            L_cached(c1.first, seed_l_cache),
                            L_cached(c2.first, seed_l_cache),
                            probability
                        });
                    }
                }
            }

            // Exact aggregation of duplicate first-layer endpoints.
            std::unordered_map<TripleKey, double, TripleKeyHash> aggregate;
            aggregate.reserve(tasks.size() * 2 + 16);
            for (const FrontTask& task : tasks) {
                aggregate[{task.d0, task.d1, task.d2}] += task.probability;
            }
            std::vector<FrontTask> compact;
            compact.reserve(aggregate.size());
            for (const auto& item : aggregate) {
                compact.push_back({
                    item.first.d0,
                    item.first.d1,
                    item.first.d2,
                    item.second
                });
            }
            tasks.swap(compact);
        }

        std::cout << "  first-step tasks          : " << tasks.size()
                  << " expanded=" << expanded_nodes.load()
                  << " pruned_conflict=" << pruned_conflicts.load()
                  << " pruned_prob=" << pruned_probabilities.load()
                  << "\n" << std::flush;

        std::mutex monitor_mutex;
        std::condition_variable monitor_cv;
        std::thread monitor_thread([&]() {
            std::unique_lock<std::mutex> lock(monitor_mutex);
            while (!monitor_stop.load(std::memory_order_relaxed)) {
                const bool stopped = monitor_cv.wait_for(
                    lock,
                    std::chrono::milliseconds(
                        static_cast<int>(FIXED_PROGRESS_EVERY * 1000.0)),
                    [&]() {
                        return monitor_stop.load(std::memory_order_relaxed);
                    }
                );
                if (stopped) break;
                lock.unlock();

                CorrVec snapshot{};
                CorrVec snapshot_abs{};
                double snapshot_mass = 0.0;
                unsigned long long snapshot_count = 0;
#pragma omp critical(streaming_accumulate)
                {
                    snapshot = global_total;
                    snapshot_abs = global_abs;
                    snapshot_mass = global_mass;
                    snapshot_count = global_count;
                }

                const auto best = max_abs_position(snapshot);
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start_time).count();

#pragma omp critical(streaming_print)
                {
                    std::cout
                        << "  progress elapsed=" << std::fixed << std::setprecision(3)
                        << elapsed << "s"
                        << " expanded=" << expanded_nodes.load()
                        << " pruned_conflict=" << pruned_conflicts.load()
                        << " pruned_prob=" << pruned_probabilities.load()
                        << " tail_seen=" << tail_evaluations.load()
                        << " tail_committed=" << snapshot_count
                        << " probability_mass=" << std::scientific
                        << std::setprecision(9) << snapshot_mass
                        << " current_best_pos=" << std::defaultfloat << best.first
                        << " current_best_corr=" << std::scientific
                        << std::setprecision(12) << best.second
                        << " current_best_log2abs=" << std::defaultfloat
                        << std::setprecision(9) << log2_abs(best.second)
                        << " best_abs_sum=" << std::scientific
                        << std::setprecision(12) << snapshot_abs[best.first]
                        << std::defaultfloat << "\n" << std::flush;
                }
                lock.lock();
            }
        });

        auto process_one_task = [&](const FrontTask& task,
                                    CorrVec& local_total,
                                    CorrVec& local_abs,
                                    double& local_mass,
                                    unsigned long long& local_count,
                                    std::unordered_map<Mask, std::vector<std::pair<Mask, double>>, MaskHash>& chi_cache,
                                    std::unordered_map<Mask, Mask, MaskHash>& l_cache) {
            dfs_exact_front(
                FIXED_FRONT_ROUNDS == 0 ? 0 : 1,
                task.d0, task.d1, task.d2,
                task.probability,
                FIXED_FRONT_ROUNDS, tail_rounds,
                chi_cache, l_cache,
                expanded_nodes, pruned_conflicts,
                pruned_probabilities, tail_evaluations,
                global_total, global_abs, global_mass, global_count,
                local_total, local_abs, local_mass, local_count
            );
        };

        // A true block2 state uses about 64 MiB.  Process endpoints
        // sequentially so each enlarged tail can use all OpenMP workers
        // internally without multiplying memory by the endpoint thread count.
        {
            std::unordered_map<Mask, std::vector<std::pair<Mask, double>>, MaskHash> chi_cache;
            std::unordered_map<Mask, Mask, MaskHash> l_cache;
            CorrVec local_total{};
            CorrVec local_abs{};
            double local_mass = 0.0;
            unsigned long long local_count = 0;
            for (size_t task_index = 0; task_index < tasks.size(); ++task_index) {
                const FrontTask& task = tasks[task_index];
                process_one_task(task, local_total, local_abs, local_mass,
                                 local_count, chi_cache, l_cache);
                malloc_trim(0);
            }
            flush_local_to_global(
                local_total, local_abs, local_mass, local_count,
                global_total, global_abs, global_mass, global_count
            );
        }

        monitor_stop.store(true, std::memory_order_relaxed);
        monitor_cv.notify_all();
        if (monitor_thread.joinable()) monitor_thread.join();

        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();

        std::vector<int> order(N);
        for (int i = 0; i < N; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return std::fabs(global_total[a]) > std::fabs(global_total[b]);
        });

        std::cout << "\nresult\n";
        std::cout << "  probability mass used    : "
                  << std::setprecision(17) << global_mass << "\n";
        std::cout << "  tail evaluations         : " << global_count << "\n";
        std::cout << "  expanded front nodes     : " << expanded_nodes.load() << "\n";
        std::cout << "  pruned conflicts         : " << pruned_conflicts.load() << "\n";
        std::cout << "  pruned probabilities     : "
                  << pruned_probabilities.load() << "\n";
        std::cout << "  elapsed seconds          : "
                  << std::fixed << std::setprecision(3) << elapsed << "\n";

        std::cout << "\nTop 20 single-bit output masks\n";
        std::cout << "rank  position  correlation              log2(abs)       sum_abs\n";
        for (int rank = 0; rank < 20; ++rank) {
            const int i = order[rank];
            std::cout << std::setw(4) << rank + 1
                      << "  " << std::setw(8) << i
                      << "  " << std::scientific << std::setprecision(12)
                      << global_total[i]
                      << "  " << std::fixed << std::setprecision(9)
                      << log2_abs(global_total[i])
                      << "  " << std::scientific << std::setprecision(12)
                      << global_abs[i]
                      << std::defaultfloat << "\n";
        } 
        if (FIXED_TARGET_POSITION >= 0 && FIXED_TARGET_POSITION < N) {
            std::cout << "\nSelected target [" << FIXED_TARGET_POSITION << "]\n";
            std::cout << "  correlation              : " << std::scientific
                      << std::setprecision(17)
                      << global_total[FIXED_TARGET_POSITION] << "\n";
            std::cout << "  log2(abs)                : " << std::fixed
                      << std::setprecision(12)
                      << log2_abs(global_total[FIXED_TARGET_POSITION]) << "\n"
                      << std::defaultfloat;
        }

        std::ofstream json(FIXED_OUTPUT_JSON);
        if (!json) throw std::runtime_error("cannot open output JSON file");
        json << std::setprecision(17);
        json << "{\n";
        json << "  \"cipher\": \"Koala-p\",\n";
        json << "  \"order\": 3,\n";
        json << "  \"rounds\": " << FIXED_ROUNDS << ",\n";
        json << "  \"front_rounds\": " << FIXED_FRONT_ROUNDS << ",\n";
        json << "  \"tail_rounds\": " << tail_rounds << ",\n";
        json << "  \"input_d0\": [" << FIXED_INPUT_D0 << "],\n";
        json << "  \"input_d1\": [" << FIXED_INPUT_D1 << "],\n";
        json << "  \"input_d2\": [" << FIXED_INPUT_D2 << "],\n";
        json << "  \"final_coordinate\": " << FINAL_COORD << ",\n";
        json << "  \"true_block_size\": 2,\n";
        json << "  \"dense_block2\": "
             << (KOALA_DENSE_BLOCK2 ? "true" : "false") << ",\n";
        json << "  \"include_iota\": "
             << (FIXED_INCLUDE_IOTA ? "true" : "false") << ",\n";
        json << "  \"first_chi_round_index\": "
             << FIXED_FIRST_CHI_ROUND_INDEX << ",\n";
        json << "  \"probability_mass_used\": " << global_mass << ",\n";
        json << "  \"tail_evaluations\": " << global_count << ",\n";
        json << "  \"expanded_front_nodes\": " << expanded_nodes.load() << ",\n";
        json << "  \"pruned_conflicts\": " << pruned_conflicts.load() << ",\n";
        json << "  \"elapsed_seconds\": " << elapsed << ",\n";
        json << "  \"single_bit_results\": [\n";
        for (int i = 0; i < N; ++i) {
            json << "    {\"position\": " << i
                 << ", \"correlation\": " << global_total[i]
                 << ", \"log2_abs\": ";
            if (global_total[i] == 0.0) json << "null";
            else json << log2_abs(global_total[i]);
            json << ", \"sum_abs\": " << global_abs[i] << "}";
            if (i + 1 != N) json << ",";
            json << "\n";
        }
        json << "  ]\n";
        json << "}\n";

        std::cout << "\n  wrote JSON               : "
                  << FIXED_OUTPUT_JSON << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

