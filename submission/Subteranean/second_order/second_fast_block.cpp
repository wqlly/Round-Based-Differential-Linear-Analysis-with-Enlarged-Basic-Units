// subterranean_second_order_fast_geometric.cpp
//
// Optimized second-order hybrid differential-linear evaluation for
// Subterranean-2.0.
//
// Scientific model:
//   - exact front: unchanged from the supplied second_sub.cpp;
//   - geometric tail: true BLOCK_SIZE=4 enlarged units, 16^4=65536 coordinates per full block;
//   - final HDL coordinate: 7 = 0b0111;
//   - all 257 one-bit output masks are accumulated simultaneously.
//
// Subterranean linear layer:
//   theta : y_i = x_i xor x_{i+3} xor x_{i+8}
//   pi    : z_i = y_{12i}
//   L     = pi o theta
//
// Hence
//   z_i = x_{12i} xor x_{12i+3} xor x_{12i+8},
//
// forward difference propagation is
//   L(e_p) =
//     e_{150p} xor e_{150(p-3)} xor e_{150(p-8)},
//
// and mask pullback is
//   L^T(e_i) = e_{12i} xor e_{12i+3} xor e_{12i+8}.
//
// Main optimization:
//   The supplied code explicitly builds 16x16 transfer matrices and multiplies
//   a cyclic matrix chain. This file evaluates the same chi transfer through
//   its exact Walsh/rank-one reduction:
//
//       sigma_i[v] = gamma_i[v] *
//           sum_a d_{i+2}[a] (-1)^{<P(v),a>}
//               gamma_{i+1}[P^{-1}(a & P(v))].
//
// Engineering fixes:
//   - monitor thread uses condition_variable and exits immediately;
//   - progress defaults to 10 seconds rather than 10000 seconds;
//   - one-task runs use internal tail parallelism;
//   - multiple front tasks use endpoint-level parallelism;
//   - thread-local totals are flushed in batches.
//
// Correctness verification:
//   Compile with SUBTERRANEAN_SECOND_ORDER_VERIFY_FAST=1 to retain the original
//   direct 16x16 matrix implementation and compare it against the fast
//   implementation on initial and post-linear gamma states.
//
// Default effective parameters are preserved from second_sub.cpp:
//   total chi layers = 7
//   exact front      = 1
//   geometric tail  = 6
//   D0=[0], D1=[32]
//
// Note: the original file name/comment says "2+5", but its actual constant
// FIXED_FRONT_ROUNDS is 1. Change it to 2 if a genuine 2+5 run is intended.
//
// Compile normal:
//   g++ -O3 -march=native -std=c++17 -fopenmp \
//       subterranean_second_order_fast_geometric.cpp \
//       -o subterranean_second_fast -lm
//
// Compile and run direct-vs-fast self-test:
//   g++ -O3 -march=native -std=c++17 -fopenmp \
//       -DSUBTERRANEAN_SECOND_ORDER_VERIFY_FAST=1 \
//       -DSUBTERRANEAN_SECOND_ORDER_SELF_TEST_ONLY=1 \
//       subterranean_second_order_fast_geometric.cpp \
//       -o subterranean_second_fast_selftest -lm
//   ./subterranean_second_fast_selftest
//

#ifndef SUBTERRANEAN_SECOND_ORDER_VERIFY_FAST
#define SUBTERRANEAN_SECOND_ORDER_VERIFY_FAST 0
#endif

#ifndef SUBTERRANEAN_SECOND_ORDER_SELF_TEST_ONLY
#define SUBTERRANEAN_SECOND_ORDER_SELF_TEST_ONLY 0
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <thread>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <limits>
#include <mutex>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif

static constexpr int N = 257;
static constexpr int W = 5; // 5*64 = 320 bits, only integer positions 0..256 used.
static constexpr uint64_t LAST_MASK = 1ULL; // position 256 is word 4 bit 0 if needed.

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
        return (size_t)h;
    }
};

using Dist = std::unordered_map<Mask, double, MaskHash>;
static std::vector<int> active_bits(const Mask& m);


// -------------------- pair-prefix pruning controls --------------------
// Keep only prefix routes whose accumulated probability is >= 2^FRONT_LOG2_CUTOFF.
// Set to -INFINITY to disable probability pruning.
static constexpr double FRONT_LOG2_CUTOFF = -30.0;
static constexpr bool FRONT_REQUIRE_DISJOINT_EACH_ROUND = true;
static constexpr bool FRONT_REQUIRE_DISJOINT_INITIAL = true;
// Block size used for block-structured output evaluation.
// To preserve the original opt3 numerical result, the tail remains the same
// bit-wise 16-dimensional second-order round-based propagation; the final
// output product is evaluated block by block, exactly equal to the original
// product when BLOCK_SIZE is changed to 4 or 8.
static constexpr int BLOCK_SIZE = 4; // legacy name; true enlarged-unit engine below is fixed to 4
static constexpr int ROUTE_CONFLICT_RADIUS = 0; // 0 = only exact overlap is treated as conflict; set to 1/2 for stricter route pruning
static constexpr bool FRONT_CHECK_SELF_CONFLICT = false; // chi-only mode: do not check D0/D1 self-conflict


static inline bool masks_intersect(const Mask& a, const Mask& b) noexcept {
    for (int i = 0; i < W; ++i) if ((a.w[i] & b.w[i]) != 0ULL) return true;
    return false;
}

static inline int cyclic_distance_int(int a, int b) noexcept {
    int d = std::abs(a - b) % N;
    return std::min(d, N - d);
}

static inline bool first_self_conflict(const Mask& a, int radius = ROUTE_CONFLICT_RADIUS) {
    auto A = active_bits(a);
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = i + 1; j < A.size(); ++j) {
            if (cyclic_distance_int(A[i], A[j]) <= radius) return true;
        }
    }
    return false;
}

static inline bool first_conflict_between(const Mask& a, const Mask& b, int radius = ROUTE_CONFLICT_RADIUS) {
    auto A = active_bits(a);
    auto B = active_bits(b);
    for (int x : A) {
        for (int y : B) {
            if (cyclic_distance_int(x, y) <= radius) return true;
        }
    }
    return false;
}

static inline bool route_conflict_rejected(const Mask& d0, const Mask& d1, bool check_self) {
    if (check_self && FRONT_CHECK_SELF_CONFLICT) {
        if (first_self_conflict(d0)) return true;
        if (first_self_conflict(d1)) return true;
    }
    return first_conflict_between(d0, d1);
}


struct PairMask {
    Mask d0;
    Mask d1;
    bool operator==(const PairMask& o) const noexcept { return d0 == o.d0 && d1 == o.d1; }
};

struct PairMaskHash {
    size_t operator()(const PairMask& p) const noexcept {
        MaskHash h;
        size_t a = h(p.d0);
        size_t b = h(p.d1);
        return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
    }
};

using PairDist = std::unordered_map<PairMask, double, PairMaskHash>;

static inline double front_prob_cutoff_value() {
    if (!std::isfinite(FRONT_LOG2_CUTOFF)) return 0.0;
    return std::exp2(FRONT_LOG2_CUTOFF);
}

static inline bool prob_survives(double p) {
    static const double cutoff = front_prob_cutoff_value();
    return p >= cutoff;
}


static inline int modN(long long x) {
    long long r = x % N;
    return (int)(r < 0 ? r + N : r);
}

static inline void xor_bit(Mask& m, int support_i) {
    support_i = modN(support_i);
    int pos = N - 1 - support_i;       // Python integer bit position from LSB.
    m.w[pos >> 6] ^= (1ULL << (pos & 63));
}

static inline int test_bit(const Mask& m, int support_i) {
    support_i = modN(support_i);
    int pos = N - 1 - support_i;
    return (int)((m.w[pos >> 6] >> (pos & 63)) & 1ULL);
}

static inline Mask set_support(const std::vector<int>& xs) {
    Mask m;
    for (int x : xs) {
        if (x < 0 || x >= N) throw std::runtime_error("support index outside [0,257)");
        xor_bit(m, x);
    }
    return m;
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
        uint64_t x = m.w[wi];
        if (x) return wi * 64 + 63 - __builtin_clzll(x);
    }
    return -1;
}

static inline int popcount_mask(const Mask& m) {
    return __builtin_popcountll(m.w[0]) + __builtin_popcountll(m.w[1]) +
           __builtin_popcountll(m.w[2]) + __builtin_popcountll(m.w[3]) +
           __builtin_popcountll(m.w[4]);
}

static std::vector<int> active_bits(const Mask& m) {
    std::vector<int> out;
    out.reserve(popcount_mask(m));
    for (int wi = 0; wi < W; ++wi) {
        uint64_t x = m.w[wi];
        while (x) {
            int b = __builtin_ctzll(x);
            int pos = wi * 64 + b;
            if (pos < N) out.push_back(N - 1 - pos);
            x &= x - 1;
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

static std::string fmt_support(const Mask& m, int max_len = 40) {
    auto v = active_bits(m);
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.size() && i < (size_t)max_len; ++i) {
        if (i) os << ",";
        os << v[i];
    }
    if (v.size() > (size_t)max_len) os << ",...(+" << (v.size() - max_len) << ")";
    os << "]";
    return os.str();
}

static Mask parse_support(std::string s) {
    for (char& c : s) if (c == '[' || c == ']') c = ' ';
    std::stringstream ss(s);
    std::vector<int> vals;
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        std::stringstream tt(tok);
        int x;
        if (tt >> x) vals.push_back(x);
    }
    return set_support(vals);
}

// 12^{-1} mod 257 = 150.
static constexpr int INV12 = 150;

static Mask L_transform(const Mask& in) {
    // Subterranean L = pi o theta:
    //   z_i = x_{12i} xor x_{12i+3} xor x_{12i+8}.
    //
    // Therefore input bit p contributes to output positions
    //   150*p,
    //   150*(p-3),
    //   150*(p-8)
    // modulo 257.
    Mask out;

    for (int p : active_bits(in)) {
        xor_bit(
            out,
            modN(
                static_cast<long long>(INV12)
                * p
            )
        );

        xor_bit(
            out,
            modN(
                static_cast<long long>(INV12)
                * (p - 3)
            )
        );

        xor_bit(
            out,
            modN(
                static_cast<long long>(INV12)
                * (p - 8)
            )
        );
    }

    return out;
}

static std::vector<Mask> gf2_independent_basis(const std::vector<Mask>& columns) {
    std::unordered_map<int, Mask> basis_by_lead;
    basis_by_lead.reserve(columns.size() * 2 + 1);
    for (const Mask& c : columns) {
        Mask x = c;
        while (!is_zero(x)) {
            int lead = highest_pos(x);
            auto it = basis_by_lead.find(lead);
            if (it == basis_by_lead.end()) {
                basis_by_lead.emplace(lead, x);
                break;
            }
            x = mask_xor(x, it->second);
        }
    }
    std::vector<Mask> basis;
    basis.reserve(basis_by_lead.size());
    for (auto& kv : basis_by_lead) basis.push_back(kv.second);
    return basis;
}

static std::vector<std::pair<Mask,double>> chi_diff_distribution(const Mask& delta) {
    Mask base;
    for (int i = 0; i < N; ++i) {
        int di = test_bit(delta, i);
        int d1 = test_bit(delta, i + 1);
        int d2 = test_bit(delta, i + 2);
        if (di ^ d2 ^ (d1 & d2)) xor_bit(base, i);
    }
    std::vector<Mask> cols;
    cols.reserve(2 * popcount_mask(delta));
    for (int j = 0; j < N; ++j) {
        Mask col;
        if (test_bit(delta, j - 1)) xor_bit(col, j - 2);
        if (test_bit(delta, j + 1)) xor_bit(col, j - 1);
        if (!is_zero(col)) cols.push_back(col);
    }
    auto basis = gf2_independent_basis(cols);
    const int r = (int)basis.size();
    if (r >= 63) throw std::runtime_error("chi basis too large to enumerate");
    const size_t nouts = 1ULL << r;
    std::vector<std::pair<Mask,double>> outs;
    outs.reserve(nouts);
    const double prob = std::ldexp(1.0, -r);
    outs.push_back({base, prob});
    // Match Python's iterative order: outs += [x ^ b for x in outs]
    for (const Mask& b : basis) {
        size_t old = outs.size();
        for (size_t k = 0; k < old; ++k) outs.push_back({mask_xor(outs[k].first, b), prob});
    }
    return outs;
}


static inline const std::vector<std::pair<Mask,double>>& chi_cached_ref(
    const Mask& d,
    std::unordered_map<Mask, std::vector<std::pair<Mask,double>>, MaskHash>& cache
) {
    auto it = cache.find(d);
    if (it != cache.end()) return it->second;
    auto res = chi_diff_distribution(d);
    auto em = cache.emplace(d, std::move(res));
    return em.first->second;
}

static inline Mask L_cached(
    const Mask& d,
    std::unordered_map<Mask, Mask, MaskHash>& cache
) {
    auto it = cache.find(d);
    if (it != cache.end()) return it->second;
    Mask out = L_transform(d);
    cache.emplace(d, out);
    return out;
}

static Dist rho_diff_distribution_once_parallel(const Dist& dist) {
    std::vector<std::pair<Mask,double>> items(dist.begin(), dist.end());
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    std::vector<Dist> locals(nthreads);
    for (auto& mp : locals) {
        // The exact number is data-dependent. Reserving aggressively avoids rehashing.
        mp.reserve(std::max<size_t>(4096, items.size() * 32 / std::max(1, nthreads)));
    }

#pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        Dist& local = locals[tid];
        std::unordered_map<Mask, Mask, MaskHash> local_L_cache;
        local_L_cache.reserve(65536);

#pragma omp for schedule(dynamic,1)
        for (size_t idx = 0; idx < items.size(); ++idx) {
            const Mask& d = items[idx].first;
            const double p = items[idx].second;
            auto chi = chi_diff_distribution(d);
            for (const auto& cq : chi) {
                auto itL = local_L_cache.find(cq.first);
                Mask Ld;
                if (itL == local_L_cache.end()) {
                    Ld = L_transform(cq.first);
                    local_L_cache.emplace(cq.first, Ld);
                } else {
                    Ld = itL->second;
                }
                local[Ld] += p * cq.second;
            }
        }
    }

    Dist out;
    size_t reserve_hint = 0;
    for (auto& mp : locals) reserve_hint += mp.size();
    out.reserve(reserve_hint * 2 + 16);
    for (auto& mp : locals) {
        for (const auto& kv : mp) out[kv.first] += kv.second;
    }
    return out;
}

static Dist exact_front_distribution(const Mask& input, int front_rounds) {
    Dist dist;
    dist.reserve(16);
    dist[input] = 1.0;
    for (int r = 0; r < front_rounds; ++r) {
        dist = rho_diff_distribution_once_parallel(dist);
    }
    return dist;
}


static PairDist rho_pair_distribution_once_pruned_parallel(const PairDist& dist, int round_index) {
    std::vector<std::pair<PairMask,double>> items(dist.begin(), dist.end());
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    std::vector<PairDist> locals(nthreads);
    for (auto& mp : locals) {
        mp.reserve(std::max<size_t>(4096, items.size() * 64 / std::max(1, nthreads)));
    }

#pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        PairDist& local = locals[tid];
        std::unordered_map<Mask, Mask, MaskHash> local_L_cache;
        std::unordered_map<Mask, std::vector<std::pair<Mask,double>>, MaskHash> local_chi_cache;
        local_L_cache.reserve(65536);
        local_chi_cache.reserve(8192);

#pragma omp for schedule(dynamic,1)
        for (size_t idx = 0; idx < items.size(); ++idx) {
            const PairMask cur = items[idx].first;
            const double p = items[idx].second;
            if (FRONT_REQUIRE_DISJOINT_EACH_ROUND && route_conflict_rejected(cur.d0, cur.d1, true)) continue;

            const auto& chi0 = chi_cached_ref(cur.d0, local_chi_cache);
            const auto& chi1 = chi_cached_ref(cur.d1, local_chi_cache);

            for (const auto& c0 : chi0) {
                const double p0 = p * c0.second;
                // Since c1.second <= 1, if p0 is already below cutoff then p0*c1.second cannot survive.
                if (!prob_survives(p0)) continue;
                const Mask L0 = L_cached(c0.first, local_L_cache);

                for (const auto& c1 : chi1) {
                    const double np = p0 * c1.second;
                    if (!prob_survives(np)) continue;

                    const Mask L1 = L_cached(c1.first, local_L_cache);
                    if (FRONT_REQUIRE_DISJOINT_EACH_ROUND && route_conflict_rejected(L0, L1, true)) continue;
                    PairMask nxt{L0, L1};
                    local[nxt] += np;
                }
            }
        }
    }

    PairDist out;
    size_t reserve_hint = 0;
    for (auto& mp : locals) reserve_hint += mp.size();
    out.reserve(reserve_hint * 2 + 16);
    for (auto& mp : locals) {
        for (const auto& kv : mp) out[kv.first] += kv.second;
    }
    return out;
}

static PairDist exact_front_pair_distribution_pruned(const Mask& input0, const Mask& input1, int front_rounds) {
    PairDist dist;
    dist.reserve(16);
    if (!(FRONT_REQUIRE_DISJOINT_INITIAL && route_conflict_rejected(input0, input1, true))) {
        PairMask init{input0, input1};
        dist[init] = 1.0;
    }
    for (int r = 0; r < front_rounds; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        dist = rho_pair_distribution_once_pruned_parallel(dist, r);
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        double mass = 0.0;
        for (const auto& kv : dist) mass += kv.second;
        std::cout << "  pruned front pair round " << (r + 1)
                  << ": endpoints=" << dist.size()
                  << " mass=" << std::setprecision(17) << mass
                  << " time=" << std::fixed << std::setprecision(3) << sec << "s\n" << std::flush;
        if (dist.empty()) break;
    }
    return dist;
}

static std::string summarize_pair_distribution(const PairDist& dist) {
    std::map<int, int> wt;
    std::map<std::string, int> probcls;
    double mass = 0.0;
    for (const auto& kv : dist) {
        mass += kv.second;
        wt[popcount_mask(kv.first.d0) + popcount_mask(kv.first.d1)]++;
        std::ostringstream key;
        key << std::fixed << std::setprecision(12) << std::log2(kv.second);
        probcls[key.str()]++;
    }
    std::ostringstream os;
    os << "{'num': " << dist.size()
       << ", 'mass': " << std::setprecision(17) << mass
       << ", 'prob_log2_classes': {";
    bool first = true;
    for (const auto& kv : probcls) {
        if (!first) os << ", "; first = false;
        os << kv.first << ": " << kv.second;
    }
    os << "}, 'total_weight_classes': {";
    first = true;
    for (const auto& kv : wt) {
        if (!first) os << ", "; first = false;
        os << kv.first << ": " << kv.second;
    }
    os << "}}";
    return os.str();
}

// -------------------- optimized second-order geometric tail --------------------

static constexpr int SECOND_DIM = 16;
static constexpr int SECOND_FINAL_COORD = 7;

using Row16 = std::array<double, SECOND_DIM>;
using Gamma = std::array<Row16, N>;
using CorrVec = std::array<double, N>;
using DTable16 = std::array<Row16, N>;

static std::array<std::array<int, 3>, N> L_ROWS;

static void init_tables() {
    for (int i = 0; i < N; ++i) {
        // Subterranean mask pullback:
        //   L^T e_i =
        //     e_{12i} + e_{12i+3} + e_{12i+8}.
        L_ROWS[i] = {
            modN(12LL * i),
            modN(12LL * i + 3),
            modN(12LL * i + 8)
        };
    }
}

static void fwt_inplace(Row16& row) {
    for (int half = 1; half < SECOND_DIM; half <<= 1) {
        for (int start = 0; start < SECOND_DIM; start += 2 * half) {
            for (int offset = 0; offset < half; ++offset) {
                const double left = row[start + offset];
                const double right = row[start + offset + half];

                row[start + offset] = left + right;
                row[start + offset + half] = left - right;
            }
        }
    }
}

// P changes the first/MSB derivative-representation coordinate into the
// parity of all four coordinates and leaves the remaining three unchanged.
static inline int p_map4(int coordinate) noexcept {
    const int low = coordinate & 0x7;
    const int high =
        __builtin_popcount(static_cast<unsigned>(coordinate)) & 1;

    return (high << 3) | low;
}

static inline int p_inverse4(int coordinate) noexcept {
    const int low = coordinate & 0x7;
    const int parity_coordinate = (coordinate >> 3) & 1;

    const int original_high =
        parity_coordinate
        ^ (
            __builtin_popcount(static_cast<unsigned>(low))
            & 1
        );

    return (original_high << 3) | low;
}

// P^T in the ordinary Walsh dot product.
// For b=(b0,b1,b2,b3):
//   P^T(b)=(b0,b0+b1,b0+b2,b0+b3).
static inline int p_transpose4(int coordinate) noexcept {
    return ((coordinate >> 3) & 1)
        ? (coordinate ^ 0x7)
        : coordinate;
}

static Gamma init_gamma_second_order(
    const Mask& d0,
    const Mask& d1
) {
    Gamma gamma{};

    for (int i = 0; i < N; ++i) {
        const int a = test_bit(d0, i);
        const int b = test_bit(d1, i);
        const int ab = a ^ b;

        Row16& row = gamma[i];
        row.fill(0.0);

        row[(0 << 3) | (a << 2) | (b << 1) | ab] = 1.0;
        row[(1 << 3) | (a << 2) | (b << 1) | ab] = 1.0;

        fwt_inplace(row);

        for (double& value : row) {
            value *= 0.5;
        }
    }

    return gamma;
}

// Exact Walsh/rank-one reduction of the same second-order chi transfer that
// the old direct 16x16 matrix-chain code computes.
static Gamma pass_chi_second_order_fast(
    const Gamma& gamma,
    bool only_final_coordinate = false
) {
    DTable16 dtable{};

    // One 16-point Walsh transform per state position.
#pragma omp parallel for schedule(static)
    for (int position = 0; position < N; ++position) {
        Row16 transformed = gamma[position];
        fwt_inplace(transformed);

        for (int coordinate = 0;
             coordinate < SECOND_DIM;
             ++coordinate) {
            dtable[position][coordinate] =
                transformed[p_transpose4(coordinate)]
                / static_cast<double>(SECOND_DIM);
        }
    }

    Gamma output{};

    for (int position = 0; position < N; ++position) {
        output[position][0] = 1.0;
    }

    const int coordinate_begin =
        only_final_coordinate
            ? SECOND_FINAL_COORD
            : 1;

    const int coordinate_end =
        only_final_coordinate
            ? SECOND_FINAL_COORD + 1
            : SECOND_DIM;

#pragma omp parallel for schedule(dynamic, 1)
    for (int coordinate = coordinate_begin;
         coordinate < coordinate_end;
         ++coordinate) {
        const int transformed_output =
            p_map4(coordinate);

        for (int position = 0;
             position < N;
             ++position) {
            const int next1 =
                (position + 1) % N;
            const int next2 =
                (position + 2) % N;

            double contraction = 0.0;

            for (int boundary = 0;
                 boundary < SECOND_DIM;
                 ++boundary) {
                const double sign =
                    (
                        __builtin_popcount(
                            static_cast<unsigned>(
                                transformed_output
                                & boundary
                            )
                        ) & 1
                    )
                    ? -1.0
                    : 1.0;

                const int source_coordinate =
                    p_inverse4(
                        boundary
                        & transformed_output
                    );

                contraction +=
                    dtable[next2][boundary]
                    * sign
                    * gamma[next1][source_coordinate];
            }

            output[position][coordinate] =
                gamma[position][coordinate]
                * contraction;
        }
    }

    return output;
}

static Gamma pass_linear_second_order(
    const Gamma& gamma
) {
    Gamma output{};

#pragma omp parallel for schedule(static)
    for (int position = 0;
         position < N;
         ++position) {
        output[position][0] = 1.0;

        const int source0 =
            L_ROWS[position][0];
        const int source1 =
            L_ROWS[position][1];
        const int source2 =
            L_ROWS[position][2];

        for (int coordinate = 1;
             coordinate < SECOND_DIM;
             ++coordinate) {
            output[position][coordinate] =
                gamma[source0][coordinate]
                * gamma[source1][coordinate]
                * gamma[source2][coordinate];
        }
    }

    return output;
}

static Gamma round_based_tail_second_order(
    const Mask& d0,
    const Mask& d1,
    int tail_rounds
) {
    Gamma gamma =
        init_gamma_second_order(d0, d1);

    for (int round = 0;
         round < tail_rounds;
         ++round) {
        const bool last =
            round == tail_rounds - 1;

        gamma =
            pass_chi_second_order_fast(
                gamma,
                last
            );

        if (!last) {
            gamma =
                pass_linear_second_order(gamma);
        }
    }

    return gamma;
}

static CorrVec tail_correlations_second_order_bitwise(
    const Mask& d0,
    const Mask& d1,
    int tail_rounds
) {
    const Gamma gamma =
        round_based_tail_second_order(
            d0,
            d1,
            tail_rounds
        );

    CorrVec output{};

    for (int position = 0;
         position < N;
         ++position) {
        output[position] =
            gamma[position][SECOND_FINAL_COORD];
    }

    return output;
}

static std::pair<int, double> max_abs_position(
    const CorrVec& values
) {
    int best_position = 0;
    double best_value = values[0];
    double best_absolute =
        std::fabs(best_value);

    for (int position = 1;
         position < N;
         ++position) {
        const double current_absolute =
            std::fabs(values[position]);

        if (current_absolute > best_absolute) {
            best_position = position;
            best_value = values[position];
            best_absolute = current_absolute;
        }
    }

    return {
        best_position,
        best_value
    };
}

static double log2_abs(double value) {
    return value == 0.0
        ? -INFINITY
        : std::log2(std::fabs(value));
}

static double log2_abs_safe(double value) {
    return log2_abs(value);
}

// ---------------------------------------------------------------------------
// Optional direct 16x16 reference implementation.
// It is excluded from normal builds.
// ---------------------------------------------------------------------------

#if SUBTERRANEAN_SECOND_ORDER_VERIFY_FAST

static double REF_MT[4][2][2];
static double REF_LOCAL[3][SECOND_DIM][SECOND_DIM][SECOND_DIM][SECOND_DIM];

struct RefMat16 {
    double m[SECOND_DIM][SECOND_DIM];
};

static void init_reference_tables() {
    const double M00[2][2] = {
        {0.5, 0.5},
        {0.5, 0.5}
    };
    const double M01[2][2] = {
        {0.5, -0.5},
        {0.5, -0.5}
    };
    const double M10[2][2] = {
        {0.5, 0.5},
        {-0.5, 0.5}
    };
    const double M11[2][2] = {
        {0.5, -0.5},
        {-0.5, -0.5}
    };

    std::memcpy(REF_MT[0], M00, sizeof(M00));
    std::memcpy(REF_MT[1], M01, sizeof(M01));
    std::memcpy(REF_MT[2], M10, sizeof(M10));
    std::memcpy(REF_MT[3], M11, sizeof(M11));

    for (int relation = 0;
         relation < 3;
         ++relation) {
        for (int output_coordinate = 0;
             output_coordinate < SECOND_DIM;
             ++output_coordinate) {
            int output_bits[4] = {
                (output_coordinate >> 3) & 1,
                (output_coordinate >> 2) & 1,
                (output_coordinate >> 1) & 1,
                output_coordinate & 1
            };

            const int output_parity =
                output_bits[0]
                ^ output_bits[1]
                ^ output_bits[2]
                ^ output_bits[3];

            for (int input_coordinate = 0;
                 input_coordinate < SECOND_DIM;
                 ++input_coordinate) {
                int input_bits[4] = {
                    (input_coordinate >> 3) & 1,
                    (input_coordinate >> 2) & 1,
                    (input_coordinate >> 1) & 1,
                    input_coordinate & 1
                };

                const int input_parity =
                    input_bits[0]
                    ^ input_bits[1]
                    ^ input_bits[2]
                    ^ input_bits[3];

                int matrix_index[4];

                const int a0 =
                    relation == 1
                        ? output_parity
                        : 0;

                const int b0 =
                    input_parity
                    ^ (
                        relation == 0
                            ? output_parity
                            : 0
                    );

                matrix_index[0] =
                    (a0 << 1) | b0;

                for (int component = 1;
                     component < 4;
                     ++component) {
                    const int a =
                        relation == 1
                            ? output_bits[component]
                            : 0;

                    const int b =
                        input_bits[component]
                        ^ (
                            relation == 0
                                ? output_bits[component]
                                : 0
                        );

                    matrix_index[component] =
                        (a << 1) | b;
                }

                for (int row = 0;
                     row < SECOND_DIM;
                     ++row) {
                    for (int column = 0;
                         column < SECOND_DIM;
                         ++column) {
                        double value = 1.0;

                        for (int component = 0;
                             component < 4;
                             ++component) {
                            const int row_bit =
                                (
                                    row
                                    >> (3 - component)
                                ) & 1;

                            const int column_bit =
                                (
                                    column
                                    >> (3 - component)
                                ) & 1;

                            value *= REF_MT[
                                matrix_index[component]
                            ][row_bit][column_bit];
                        }

                        REF_LOCAL[
                            relation
                        ][output_coordinate]
                         [input_coordinate]
                         [row]
                         [column] = value;
                    }
                }
            }
        }
    }
}

static inline void reference_matmul16(
    const double left[SECOND_DIM][SECOND_DIM],
    const double right[SECOND_DIM][SECOND_DIM],
    double output[SECOND_DIM][SECOND_DIM]
) {
    for (int row = 0;
         row < SECOND_DIM;
         ++row) {
        for (int column = 0;
             column < SECOND_DIM;
             ++column) {
            double sum = 0.0;

            for (int middle = 0;
                 middle < SECOND_DIM;
                 ++middle) {
                sum +=
                    left[row][middle]
                    * right[middle][column];
            }

            output[row][column] = sum;
        }
    }
}

static inline void reference_identity16(
    double matrix[SECOND_DIM][SECOND_DIM]
) {
    std::memset(
        matrix,
        0,
        sizeof(double)
            * SECOND_DIM
            * SECOND_DIM
    );

    for (int diagonal = 0;
         diagonal < SECOND_DIM;
         ++diagonal) {
        matrix[diagonal][diagonal] = 1.0;
    }
}

static inline void reference_build_matrix(
    const Gamma& gamma,
    int relation,
    int position,
    int output_coordinate,
    double matrix[SECOND_DIM][SECOND_DIM]
) {
    const Row16& input_row =
        gamma[position];

    for (int row = 0;
         row < SECOND_DIM;
         ++row) {
        for (int column = 0;
             column < SECOND_DIM;
             ++column) {
            double sum = 0.0;

            for (int input_coordinate = 0;
                 input_coordinate < SECOND_DIM;
                 ++input_coordinate) {
                sum +=
                    input_row[input_coordinate]
                    * REF_LOCAL[
                        relation
                    ][output_coordinate]
                     [input_coordinate]
                     [row]
                     [column];
            }

            matrix[row][column] = sum;
        }
    }
}

static Gamma pass_chi_second_order_reference(
    const Gamma& gamma,
    bool only_final_coordinate = false
) {
    static thread_local std::vector<RefMat16>
        default_matrices;
    static thread_local std::vector<RefMat16>
        relation0_matrices;
    static thread_local std::vector<RefMat16>
        relation1_matrices;
    static thread_local std::vector<RefMat16>
        prefixes;
    static thread_local std::vector<RefMat16>
        suffixes;

    if (default_matrices.empty()) {
        default_matrices.resize(N);
        relation0_matrices.resize(N);
        relation1_matrices.resize(N);
        prefixes.resize(N + 1);
        suffixes.resize(N + 1);
    }

    RefMat16 middle_product;
    RefMat16 temporary0;
    RefMat16 temporary1;

    Gamma output{};

    for (int position = 0;
         position < N;
         ++position) {
        output[position][0] = 1.0;
    }

    const int coordinate_begin =
        only_final_coordinate
            ? SECOND_FINAL_COORD
            : 1;

    const int coordinate_end =
        only_final_coordinate
            ? SECOND_FINAL_COORD + 1
            : SECOND_DIM;

    for (int output_coordinate = coordinate_begin;
         output_coordinate < coordinate_end;
         ++output_coordinate) {
        for (int position = 0;
             position < N;
             ++position) {
            reference_build_matrix(
                gamma,
                2,
                position,
                output_coordinate,
                default_matrices[position].m
            );

            reference_build_matrix(
                gamma,
                0,
                position,
                output_coordinate,
                relation0_matrices[position].m
            );

            reference_build_matrix(
                gamma,
                1,
                position,
                output_coordinate,
                relation1_matrices[position].m
            );
        }

        reference_identity16(prefixes[0].m);

        for (int k = 1; k <= N; ++k) {
            reference_matmul16(
                default_matrices[k - 1].m,
                prefixes[k - 1].m,
                prefixes[k].m
            );
        }

        reference_identity16(suffixes[N].m);

        for (int k = N - 1;
             k >= 0;
             --k) {
            reference_matmul16(
                suffixes[k + 1].m,
                default_matrices[k].m,
                suffixes[k].m
            );
        }

        reference_identity16(middle_product.m);

        for (int k = 1;
             k <= N - 2;
             ++k) {
            reference_matmul16(
                default_matrices[k].m,
                middle_product.m,
                temporary0.m
            );

            middle_product = temporary0;
        }

        for (int output_position = 0;
             output_position < N;
             ++output_position) {
            double trace = 0.0;

            if (output_position <= N - 2) {
                reference_matmul16(
                    relation1_matrices[
                        output_position + 1
                    ].m,
                    relation0_matrices[
                        output_position
                    ].m,
                    temporary0.m
                );

                reference_matmul16(
                    prefixes[output_position].m,
                    suffixes[output_position + 2].m,
                    temporary1.m
                );

                for (int row = 0;
                     row < SECOND_DIM;
                     ++row) {
                    for (int column = 0;
                         column < SECOND_DIM;
                         ++column) {
                        trace +=
                            temporary0.m[row][column]
                            * temporary1.m[column][row];
                    }
                }
            } else {
                reference_matmul16(
                    relation1_matrices[0].m,
                    relation0_matrices[N - 1].m,
                    temporary0.m
                );

                for (int row = 0;
                     row < SECOND_DIM;
                     ++row) {
                    for (int column = 0;
                         column < SECOND_DIM;
                         ++column) {
                        trace +=
                            temporary0.m[row][column]
                            * middle_product.m[column][row];
                    }
                }
            }

            output[output_position][
                output_coordinate
            ] = trace;
        }
    }

    return output;
}

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

    for (int position = 0;
         position < N;
         ++position) {
        const int coordinate_begin =
            only_final_coordinate
                ? SECOND_FINAL_COORD
                : 0;

        const int coordinate_end =
            only_final_coordinate
                ? SECOND_FINAL_COORD + 1
                : SECOND_DIM;

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
    // Check forward basis images:
    // L(e_p) = {150p, 150(p-3), 150(p-8)}.
    for (int p = 0; p < N; ++p) {
        Mask basis;
        xor_bit(basis, p);

        const Mask formula =
            L_transform(basis);

        Mask explicit_image;

        // Explicitly evaluate z_i =
        // x_{12i} + x_{12i+3} + x_{12i+8}
        // on the basis vector e_p.
        for (int i = 0; i < N; ++i) {
            const int value =
                (p == modN(12LL * i))
                ^ (p == modN(12LL * i + 3))
                ^ (p == modN(12LL * i + 8));

            if (value != 0) {
                xor_bit(explicit_image, i);
            }
        }

        if (!(formula == explicit_image)) {
            throw std::runtime_error(
                "Subterranean forward linear self-test failed at bit "
                + std::to_string(p)
            );
        }
    }

    // Check every transpose row against the same explicit linear map.
    for (int output_position = 0;
         output_position < N;
         ++output_position) {
        for (int input_position = 0;
             input_position < N;
             ++input_position) {
            const bool explicit_coefficient =
                input_position
                    == modN(12LL * output_position)
                || input_position
                    == modN(12LL * output_position + 3)
                || input_position
                    == modN(12LL * output_position + 8);

            const bool row_coefficient =
                input_position
                    == L_ROWS[output_position][0]
                || input_position
                    == L_ROWS[output_position][1]
                || input_position
                    == L_ROWS[output_position][2];

            if (explicit_coefficient != row_coefficient) {
                throw std::runtime_error(
                    "Subterranean L^T row self-test failed"
                );
            }
        }
    }

    std::cout
        << "  forward L and L^T rows: PASS\n"
        << std::flush;
}

static void run_fast_chi_self_tests() {
    std::cout
        << "Subterranean-2.0 second-order fast chi self-tests\n"
        << std::flush;

    verify_subterranean_linear_layer();
    init_reference_tables();

    const std::array<std::pair<std::string, std::string>, 3>
        direction_pairs = {{
            {"0", "32"},
            {"0", "11"},
            {"0,7", "32,93"}
        }};

    for (const auto& pair : direction_pairs) {
        const Mask d0 =
            parse_support(pair.first);
        const Mask d1 =
            parse_support(pair.second);

        const Gamma initial =
            init_gamma_second_order(d0, d1);

        const Gamma direct_first =
            pass_chi_second_order_reference(
                initial,
                false
            );

        const Gamma fast_first =
            pass_chi_second_order_fast(
                initial,
                false
            );

        int worst_position = 0;
        int worst_coordinate = 0;

        const double first_difference =
            maximum_gamma_difference(
                direct_first,
                fast_first,
                false,
                worst_position,
                worst_coordinate
            );

        std::cout
            << "  initial gamma d0=["
            << pair.first
            << "] d1=["
            << pair.second
            << "] max_diff="
            << std::scientific
            << std::setprecision(6)
            << first_difference
            << " at ("
            << worst_position
            << ","
            << worst_coordinate
            << ")\n";

        if (first_difference > 2e-11) {
            throw std::runtime_error(
                "fast chi mismatch on initial gamma"
            );
        }

        // Compare again after a linear layer.  This gamma is generally no
        // longer the trivial initial sign-vector family.
        const Gamma post_linear =
            pass_linear_second_order(
                fast_first
            );

        const Gamma direct_second =
            pass_chi_second_order_reference(
                post_linear,
                false
            );

        const Gamma fast_second =
            pass_chi_second_order_fast(
                post_linear,
                false
            );

        const double second_difference =
            maximum_gamma_difference(
                direct_second,
                fast_second,
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

        if (second_difference > 5e-11) {
            throw std::runtime_error(
                "fast chi mismatch on post-linear gamma"
            );
        }

        const Gamma direct_final =
            pass_chi_second_order_reference(
                post_linear,
                true
            );

        const Gamma fast_final =
            pass_chi_second_order_fast(
                post_linear,
                true
            );

        const double final_difference =
            maximum_gamma_difference(
                direct_final,
                fast_final,
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

        if (final_difference > 5e-11) {
            throw std::runtime_error(
                "fast final-coordinate chi mismatch"
            );
        }
    }

    std::cout
        << "  direct 16x16 matrix vs fast rank-one: PASS\n"
        << std::flush;
}

#endif // SUBTERRANEAN_SECOND_ORDER_VERIFY_FAST


// ===========================================================================
// TRUE BLOCK-SIZE-4 ENLARGED-UNIT SECOND-ORDER TAIL
//
// Each physical bit carries a 4-bit 2nd-order state
//   (x, D0 x, D1 x, D_{D0+D1} x),
// hence one 4-bit physical block has 4*4=16 binary coordinates and a full
// local correlation factor of dimension 2^16 = 16^4 = 65536.
//
// The implementation below is deliberately different from the old
// BLOCK_SIZE parameter that merely grouped scalar products.  Here:
//   * a block factor stores all 65536 joint coordinates;
//   * chi is applied exactly to the target block together with the two right
//     boundary positions, under the current block-product approximation;
//   * only after chi or L has been evaluated are the 65 block factors
//     re-factorized.
//
// The probability-domain implementation is equivalent to correlation-matrix
// propagation because a correlation vector is the Walsh transform of the
// (possibly signed) local mass vector.  It is especially useful here because
// the deterministic chi map on four cube vertices can be evaluated directly.
// ===========================================================================

static constexpr int TRUE_BLOCK_SIZE = 4;
static constexpr int BLOCK_ALPHABET = SECOND_DIM; // 16 states per physical bit.

struct EnlargedBlockSpec {
    int start = 0;
    int length = 0;
    int dimension = 0; // 16^length
};

using DenseFactor = std::vector<double>;
using EnlargedFactors = std::vector<DenseFactor>;

static inline int pow16_int(int exponent) {
    int out = 1;
    for (int i = 0; i < exponent; ++i) out *= 16;
    return out;
}

static void fwt_dense_inplace(DenseFactor& values) {
    const size_t n = values.size();
    if (n == 0 || (n & (n - 1)) != 0) {
        throw std::runtime_error("FWT length must be a nonzero power of two");
    }
    for (size_t half = 1; half < n; half <<= 1) {
        for (size_t start = 0; start < n; start += 2 * half) {
            for (size_t offset = 0; offset < half; ++offset) {
                const double left = values[start + offset];
                const double right = values[start + offset + half];
                values[start + offset] = left + right;
                values[start + offset + half] = left - right;
            }
        }
    }
}

static inline int parity_u32(unsigned value) noexcept {
    return __builtin_popcount(value) & 1;
}

// Convert the 4-bit derivative representation
//   r=(x,d0,d1,dsum)
// to the four actual cube vertices
//   (x, x+d0, x+d1, x+dsum).
static inline std::array<int, 4> rep4_to_vertices(int r) noexcept {
    const int x = (r >> 3) & 1;
    const int d0 = (r >> 2) & 1;
    const int d1 = (r >> 1) & 1;
    const int ds = r & 1;
    return {x, x ^ d0, x ^ d1, x ^ ds};
}

static inline int vertices_to_rep4(const std::array<int, 4>& v) noexcept {
    return (v[0] << 3)
        | ((v[0] ^ v[1]) << 2)
        | ((v[0] ^ v[2]) << 1)
        | (v[0] ^ v[3]);
}

static inline int chi_rep4_direct(int left, int middle, int right) noexcept {
    const auto a = rep4_to_vertices(left);
    const auto b = rep4_to_vertices(middle);
    const auto c = rep4_to_vertices(right);
    std::array<int, 4> y{};
    for (int vertex = 0; vertex < 4; ++vertex) {
        y[vertex] = a[vertex] ^ ((b[vertex] ^ 1) & c[vertex]);
    }
    return vertices_to_rep4(y);
}

class TrueBlock4SecondOrder {
public:
    explicit TrueBlock4SecondOrder(int requested_block_size = TRUE_BLOCK_SIZE)
        : block_size_(requested_block_size) {
        if (block_size_ <= 0 || block_size_ > TRUE_BLOCK_SIZE) {
            throw std::runtime_error("supported block sizes are 1..4");
        }
        build_layout();
        build_chi_rep_table();
        build_linear_basis_contributions();
        if (block_size_ == 4) build_chi_map4();
    }

    int block_size() const noexcept { return block_size_; }
    int num_blocks() const noexcept { return static_cast<int>(blocks_.size()); }
    const std::vector<EnlargedBlockSpec>& blocks() const noexcept { return blocks_; }

    EnlargedFactors initial_factors(const Mask& d0, const Mask& d1) const {
        EnlargedFactors factors(blocks_.size());
#pragma omp parallel for schedule(static)
        for (int block = 0; block < static_cast<int>(blocks_.size()); ++block) {
            const auto& spec = blocks_[block];
            DenseFactor factor(spec.dimension, 1.0);
            std::array<Row16, TRUE_BLOCK_SIZE> local_rows{};

            for (int local = 0; local < spec.length; ++local) {
                const int position = spec.start + local;
                const int a = test_bit(d0, position);
                const int b = test_bit(d1, position);
                const int ds = a ^ b;
                Row16 row{};
                row.fill(0.0);
                row[(0 << 3) | (a << 2) | (b << 1) | ds] = 1.0;
                row[(1 << 3) | (a << 2) | (b << 1) | ds] = 1.0;
                fwt_inplace(row);
                for (double& x : row) x *= 0.5;
                local_rows[local] = row;
            }

            for (int index = 0; index < spec.dimension; ++index) {
                int x = index;
                double value = 1.0;
                for (int local = 0; local < spec.length; ++local) {
                    value *= local_rows[local][x & 0xF];
                    x >>= 4;
                }
                factor[index] = value;
            }
            factor[0] = 1.0;
            factors[block] = std::move(factor);
        }
        return factors;
    }

    CorrVec tail_correlations(const Mask& d0, const Mask& d1, int tail_rounds) const {
        EnlargedFactors factors = initial_factors(d0, d1);
        if (tail_rounds == 0) return read_single_bit_masks(factors);

        for (int round = 0; round < tail_rounds; ++round) {
            const bool last = (round + 1 == tail_rounds);
            if (last) {
                return pass_chi_final_single_bits(factors);
            }
            factors = pass_chi_full(factors);
            factors = pass_linear_full(factors);
        }
        throw std::runtime_error("unreachable tail loop");
    }

    EnlargedFactors pass_chi_full(const EnlargedFactors& factors) const {
        check_factor_shapes(factors);
        const EnlargedFactors pmfs = factors_to_pmfs(factors);
        EnlargedFactors output(blocks_.size());

#pragma omp parallel for schedule(dynamic, 1)
        for (int block = 0; block < static_cast<int>(blocks_.size()); ++block) {
            const auto& spec = blocks_[block];
            DenseFactor context = right_context_distribution(pmfs, block);
            DenseFactor out_pmf(spec.dimension, 0.0);

            std::vector<std::pair<int, double>> nonzero_context;
            nonzero_context.reserve(256);
            for (int c = 0; c < 256; ++c) {
                if (context[c] != 0.0) nonzero_context.push_back({c, context[c]});
            }

            const DenseFactor& input_pmf = pmfs[block];
            for (int input = 0; input < spec.dimension; ++input) {
                const double px = input_pmf[input];
                if (px == 0.0) continue;
                for (const auto& cp : nonzero_context) {
                    const int output_index = chi_block_map(spec.length, input, cp.first);
                    out_pmf[output_index] += px * cp.second;
                }
            }

            fwt_dense_inplace(out_pmf);
            out_pmf[0] = 1.0;
            output[block] = std::move(out_pmf);
        }
        return output;
    }

    EnlargedFactors pass_linear_full(const EnlargedFactors& factors) const {
        check_factor_shapes(factors);
        EnlargedFactors output(blocks_.size());

#pragma omp parallel for schedule(dynamic, 1)
        for (int target = 0; target < static_cast<int>(blocks_.size()); ++target) {
            const auto& spec = blocks_[target];
            DenseFactor out(spec.dimension, 1.0);
            for (int coordinate = 1; coordinate < spec.dimension; ++coordinate) {
                out[coordinate] = linear_coordinate_value(factors, target, coordinate);
            }
            out[0] = 1.0;
            output[target] = std::move(out);
        }
        return output;
    }

    // Used by self-tests: convert a block-size-1 factorization to the old Gamma.
    Gamma as_bitwise_gamma(const EnlargedFactors& factors) const {
        if (block_size_ != 1 || static_cast<int>(blocks_.size()) != N) {
            throw std::runtime_error("as_bitwise_gamma requires block size 1");
        }
        Gamma gamma{};
        for (int i = 0; i < N; ++i) {
            if (factors[i].size() != SECOND_DIM) {
                throw std::runtime_error("bad block-size-1 factor");
            }
            for (int c = 0; c < SECOND_DIM; ++c) gamma[i][c] = factors[i][c];
        }
        return gamma;
    }

    // Exact local kernel used independently by self-tests.
    DenseFactor chi_local_from_pmfs(
        int length,
        const DenseFactor& target_pmf,
        const DenseFactor& context_pmf
    ) const {
        const int dimension = pow16_int(length);
        if (static_cast<int>(target_pmf.size()) != dimension || context_pmf.size() != 256) {
            throw std::runtime_error("bad local chi PMF shape");
        }
        DenseFactor out_pmf(dimension, 0.0);
        for (int x = 0; x < dimension; ++x) {
            if (target_pmf[x] == 0.0) continue;
            for (int c = 0; c < 256; ++c) {
                if (context_pmf[c] == 0.0) continue;
                out_pmf[chi_block_map(length, x, c)] += target_pmf[x] * context_pmf[c];
            }
        }
        fwt_dense_inplace(out_pmf);
        return out_pmf;
    }

    double linear_coordinate_value(
        const EnlargedFactors& factors,
        int target_block,
        int coordinate
    ) const {
        std::array<uint16_t, N> source_coordinates{};
        std::array<uint8_t, N> touched{};
        std::array<int, N> touched_list{};
        int touched_count = 0;

        unsigned bits = static_cast<unsigned>(coordinate);
        while (bits != 0U) {
            const int bit = __builtin_ctz(bits);
            bits &= bits - 1U;
            const auto& contribution = linear_basis_[target_block][bit];
            for (int term = 0; term < 3; ++term) {
                const int source_block = contribution[term].first;
                const uint16_t mask = contribution[term].second;
                if (!touched[source_block]) {
                    touched[source_block] = 1;
                    touched_list[touched_count++] = source_block;
                }
                source_coordinates[source_block] ^= mask;
            }
        }

        double value = 1.0;
        for (int j = 0; j < touched_count; ++j) {
            const int source_block = touched_list[j];
            value *= factors[source_block][source_coordinates[source_block]];
            if (value == 0.0) break;
        }
        return value;
    }

private:
    int block_size_ = TRUE_BLOCK_SIZE;
    std::vector<EnlargedBlockSpec> blocks_;
    std::array<int, N> block_of_position_{};
    std::array<int, N> local_of_position_{};
    std::array<std::array<std::array<uint8_t, 16>, 16>, 16> chi_rep_{};
    std::vector<uint16_t> chi_map4_;

    using LinearTerm = std::pair<int, uint16_t>;
    using LinearBasisContribution = std::array<LinearTerm, 3>;
    std::vector<std::vector<LinearBasisContribution>> linear_basis_;

    void build_layout() {
        int start = 0;
        while (start < N) {
            const int length = std::min(block_size_, N - start);
            const int block = static_cast<int>(blocks_.size());
            blocks_.push_back({start, length, pow16_int(length)});
            for (int local = 0; local < length; ++local) {
                block_of_position_[start + local] = block;
                local_of_position_[start + local] = local;
            }
            start += length;
        }
        if (block_size_ == 4 && blocks_.size() != 65) {
            throw std::runtime_error("block-size-4 layout must have 65 blocks");
        }
    }

    void build_chi_rep_table() {
        for (int a = 0; a < 16; ++a) {
            for (int b = 0; b < 16; ++b) {
                for (int c = 0; c < 16; ++c) {
                    chi_rep_[a][b][c] = static_cast<uint8_t>(chi_rep4_direct(a, b, c));
                }
            }
        }
    }

    void build_chi_map4() {
        chi_map4_.resize(static_cast<size_t>(1u << 16) * 256u);
#pragma omp parallel for schedule(static)
        for (int input = 0; input < (1 << 16); ++input) {
            const int x0 = input & 0xF;
            const int x1 = (input >> 4) & 0xF;
            const int x2 = (input >> 8) & 0xF;
            const int x3 = (input >> 12) & 0xF;
            const size_t base = static_cast<size_t>(input) * 256u;
            for (int context = 0; context < 256; ++context) {
                const int x4 = context & 0xF;
                const int x5 = (context >> 4) & 0xF;
                const int y0 = chi_rep_[x0][x1][x2];
                const int y1 = chi_rep_[x1][x2][x3];
                const int y2 = chi_rep_[x2][x3][x4];
                const int y3 = chi_rep_[x3][x4][x5];
                chi_map4_[base + context] = static_cast<uint16_t>(
                    y0 | (y1 << 4) | (y2 << 8) | (y3 << 12)
                );
            }
        }
    }

    int chi_block_map(int length, int input, int context) const noexcept {
        if (length == 4 && !chi_map4_.empty()) {
            return chi_map4_[static_cast<size_t>(input) * 256u + context];
        }
        std::array<int, 6> x{};
        for (int local = 0; local < length; ++local) {
            x[local] = (input >> (4 * local)) & 0xF;
        }
        x[length] = context & 0xF;
        x[length + 1] = (context >> 4) & 0xF;
        int output = 0;
        for (int local = 0; local < length; ++local) {
            output |= static_cast<int>(chi_rep_[x[local]][x[local + 1]][x[local + 2]])
                << (4 * local);
        }
        return output;
    }

    void build_linear_basis_contributions() {
        linear_basis_.resize(blocks_.size());
        for (int block = 0; block < static_cast<int>(blocks_.size()); ++block) {
            const auto& spec = blocks_[block];
            linear_basis_[block].resize(4 * spec.length);
            for (int local = 0; local < spec.length; ++local) {
                const int output_position = spec.start + local;
                for (int component_bit = 0; component_bit < 4; ++component_bit) {
                    LinearBasisContribution contribution{};
                    for (int term = 0; term < 3; ++term) {
                        const int source_position = L_ROWS[output_position][term];
                        const int source_block = block_of_position_[source_position];
                        const int source_local = local_of_position_[source_position];
                        const uint16_t source_mask = static_cast<uint16_t>(
                            1u << (4 * source_local + component_bit)
                        );
                        contribution[term] = {source_block, source_mask};
                    }
                    linear_basis_[block][4 * local + component_bit] = contribution;
                }
            }
        }
    }

    void check_factor_shapes(const EnlargedFactors& factors) const {
        if (factors.size() != blocks_.size()) {
            throw std::runtime_error("wrong number of enlarged factors");
        }
        for (size_t b = 0; b < blocks_.size(); ++b) {
            if (static_cast<int>(factors[b].size()) != blocks_[b].dimension) {
                throw std::runtime_error("wrong enlarged factor dimension");
            }
        }
    }

    EnlargedFactors factors_to_pmfs(const EnlargedFactors& factors) const {
        EnlargedFactors pmfs(blocks_.size());
#pragma omp parallel for schedule(static)
        for (int block = 0; block < static_cast<int>(blocks_.size()); ++block) {
            DenseFactor p = factors[block];
            fwt_dense_inplace(p);
            const double scale = 1.0 / static_cast<double>(p.size());
            for (double& x : p) x *= scale;
            pmfs[block] = std::move(p);
        }
        return pmfs;
    }

    DenseFactor marginal(
        const DenseFactor& pmf,
        int block_length,
        const std::vector<int>& selected_positions
    ) const {
        const int output_dimension = pow16_int(static_cast<int>(selected_positions.size()));
        DenseFactor out(output_dimension, 0.0);
        for (int state = 0; state < static_cast<int>(pmf.size()); ++state) {
            const double p = pmf[state];
            if (p == 0.0) continue;
            int key = 0;
            for (int j = 0; j < static_cast<int>(selected_positions.size()); ++j) {
                const int nibble = (state >> (4 * selected_positions[j])) & 0xF;
                key |= nibble << (4 * j);
            }
            out[key] += p;
        }
        return out;
    }

    DenseFactor right_context_distribution(
        const EnlargedFactors& pmfs,
        int target_block
    ) const {
        const auto& target = blocks_[target_block];
        const int p1 = (target.start + target.length) % N;
        const int p2 = (p1 + 1) % N;
        const int b1 = block_of_position_[p1];
        const int b2 = block_of_position_[p2];
        const int l1 = local_of_position_[p1];
        const int l2 = local_of_position_[p2];

        if (b1 == b2) {
            return marginal(pmfs[b1], blocks_[b1].length, {l1, l2});
        }

        const DenseFactor m1 = marginal(pmfs[b1], blocks_[b1].length, {l1});
        const DenseFactor m2 = marginal(pmfs[b2], blocks_[b2].length, {l2});
        DenseFactor out(256, 0.0);
        for (int a = 0; a < 16; ++a) {
            for (int b = 0; b < 16; ++b) {
                out[a | (b << 4)] = m1[a] * m2[b];
            }
        }
        return out;
    }

    CorrVec read_single_bit_masks(const EnlargedFactors& factors) const {
        CorrVec output{};
        for (int position = 0; position < N; ++position) {
            const int block = block_of_position_[position];
            const int local = local_of_position_[position];
            output[position] = factors[block][SECOND_FINAL_COORD << (4 * local)];
        }
        return output;
    }

    DenseFactor three_position_joint(
        const EnlargedFactors& pmfs,
        int p0,
        int p1,
        int p2
    ) const {
        const int b0 = block_of_position_[p0];
        const int b1 = block_of_position_[p1];
        const int b2 = block_of_position_[p2];
        const int l0 = local_of_position_[p0];
        const int l1 = local_of_position_[p1];
        const int l2 = local_of_position_[p2];
        DenseFactor out(4096, 0.0);

        if (b0 == b1 && b1 == b2) {
            return marginal(pmfs[b0], blocks_[b0].length, {l0, l1, l2});
        }
        if (b0 == b1) {
            const DenseFactor pair = marginal(pmfs[b0], blocks_[b0].length, {l0, l1});
            const DenseFactor single = marginal(pmfs[b2], blocks_[b2].length, {l2});
            for (int ab = 0; ab < 256; ++ab) {
                for (int c = 0; c < 16; ++c) out[ab | (c << 8)] = pair[ab] * single[c];
            }
            return out;
        }
        if (b1 == b2) {
            const DenseFactor single = marginal(pmfs[b0], blocks_[b0].length, {l0});
            const DenseFactor pair = marginal(pmfs[b1], blocks_[b1].length, {l1, l2});
            for (int a = 0; a < 16; ++a) {
                for (int bc = 0; bc < 256; ++bc) out[a | (bc << 4)] = single[a] * pair[bc];
            }
            return out;
        }

        const DenseFactor m0 = marginal(pmfs[b0], blocks_[b0].length, {l0});
        const DenseFactor m1 = marginal(pmfs[b1], blocks_[b1].length, {l1});
        const DenseFactor m2 = marginal(pmfs[b2], blocks_[b2].length, {l2});
        for (int a = 0; a < 16; ++a) {
            for (int b = 0; b < 16; ++b) {
                for (int c = 0; c < 16; ++c) {
                    out[a | (b << 4) | (c << 8)] = m0[a] * m1[b] * m2[c];
                }
            }
        }
        return out;
    }

    CorrVec pass_chi_final_single_bits(const EnlargedFactors& factors) const {
        const EnlargedFactors pmfs = factors_to_pmfs(factors);
        CorrVec output{};
#pragma omp parallel for schedule(dynamic, 1)
        for (int position = 0; position < N; ++position) {
            const int p1 = (position + 1) % N;
            const int p2 = (position + 2) % N;
            const DenseFactor joint = three_position_joint(pmfs, position, p1, p2);
            double correlation = 0.0;
            for (int state = 0; state < 4096; ++state) {
                const double p = joint[state];
                if (p == 0.0) continue;
                const int a = state & 0xF;
                const int b = (state >> 4) & 0xF;
                const int c = (state >> 8) & 0xF;
                const int y = chi_rep_[a][b][c];
                const double sign = parity_u32(static_cast<unsigned>(SECOND_FINAL_COORD & y)) ? -1.0 : 1.0;
                correlation += p * sign;
            }
            output[position] = correlation;
        }
        return output;
    }
};

static TrueBlock4SecondOrder& true_block4_engine() {
    static TrueBlock4SecondOrder engine(TRUE_BLOCK_SIZE);
    return engine;
}

static CorrVec tail_correlations_second_order(
    const Mask& d0,
    const Mask& d1,
    int tail_rounds
) {
    return true_block4_engine().tail_correlations(d0, d1, tail_rounds);
}

static double max_gamma_difference_basic(
    const Gamma& left,
    const Gamma& right,
    int& worst_position,
    int& worst_coordinate
) {
    double maximum = 0.0;
    worst_position = 0;
    worst_coordinate = 0;
    for (int position = 0; position < N; ++position) {
        for (int coordinate = 0; coordinate < SECOND_DIM; ++coordinate) {
            const double difference = std::fabs(
                left[position][coordinate] - right[position][coordinate]
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

static double max_corrvec_difference(const CorrVec& a, const CorrVec& b, int& where) {
    double maximum = 0.0;
    where = 0;
    for (int i = 0; i < N; ++i) {
        const double d = std::fabs(a[i] - b[i]);
        if (d > maximum) {
            maximum = d;
            where = i;
        }
    }
    return maximum;
}

static void run_true_block4_self_tests() {
    std::cout << "Running true BLOCK_SIZE=4 enlarged-unit self-tests...\n";

    // 1. Derivative representation and direct four-vertex chi consistency.
    for (int a = 0; a < 16; ++a) {
        for (int b = 0; b < 16; ++b) {
            for (int c = 0; c < 16; ++c) {
                const int y = chi_rep4_direct(a, b, c);
                if (y < 0 || y >= 16) throw std::runtime_error("chi_rep4 range failure");
            }
        }
    }

    // 2. FWT involution test at the real block dimension 65536.
    DenseFactor probe(1u << 16);
    for (size_t i = 0; i < probe.size(); ++i) {
        probe[i] = static_cast<double>((i * 17u + 13u) % 101u) / 101.0 - 0.5;
    }
    const DenseFactor original = probe;
    fwt_dense_inplace(probe);
    fwt_dense_inplace(probe);
    const double inverse_scale = 1.0 / static_cast<double>(probe.size());
    double fwt_error = 0.0;
    for (size_t i = 0; i < probe.size(); ++i) {
        probe[i] *= inverse_scale;
        fwt_error = std::max(fwt_error, std::fabs(probe[i] - original[i]));
    }
    if (fwt_error > 1e-11) throw std::runtime_error("65536-point FWT round-trip failed");

    // 3. BLOCK_SIZE=1 must exactly reproduce the old bitwise model.
    TrueBlock4SecondOrder engine1(1);
    const Mask d0 = set_support({0});
    const Mask d1 = set_support({32});
    const EnlargedFactors init1 = engine1.initial_factors(d0, d1);
    const Gamma old_init = init_gamma_second_order(d0, d1);
    const Gamma new_init = engine1.as_bitwise_gamma(init1);
    int wp = 0, wc = 0;
    const double init_error = max_gamma_difference_basic(old_init, new_init, wp, wc);
    if (init_error > 1e-12) throw std::runtime_error("BLOCK_SIZE=1 initialization mismatch");

    const EnlargedFactors new_chi_factors = engine1.pass_chi_full(init1);
    const Gamma new_chi = engine1.as_bitwise_gamma(new_chi_factors);
    const Gamma old_chi = pass_chi_second_order_fast(old_init, false);
    const double chi_error = max_gamma_difference_basic(old_chi, new_chi, wp, wc);
    if (chi_error > 2e-12) {
        throw std::runtime_error("BLOCK_SIZE=1 chi mismatch at position/coordinate "
            + std::to_string(wp) + "/" + std::to_string(wc));
    }

    const EnlargedFactors new_linear_factors = engine1.pass_linear_full(new_chi_factors);
    const Gamma new_linear = engine1.as_bitwise_gamma(new_linear_factors);
    const Gamma old_linear = pass_linear_second_order(old_chi);
    const double linear_error = max_gamma_difference_basic(old_linear, new_linear, wp, wc);
    if (linear_error > 2e-12) throw std::runtime_error("BLOCK_SIZE=1 linear mismatch");

    const CorrVec old_tail2 = tail_correlations_second_order_bitwise(d0, d1, 2);
    const CorrVec new_tail2 = engine1.tail_correlations(d0, d1, 2);
    int where = 0;
    const double tail_error = max_corrvec_difference(old_tail2, new_tail2, where);
    if (tail_error > 3e-12) {
        throw std::runtime_error("BLOCK_SIZE=1 end-to-end mismatch at output " + std::to_string(where));
    }

    // 4. Independent BLOCK_SIZE=4 local-kernel validation.
    TrueBlock4SecondOrder engine4(4);
    DenseFactor target_pmf(1u << 16, 0.0);
    DenseFactor context_pmf(256, 0.0);
    double target_sum = 0.0;
    double context_sum = 0.0;
    for (int k = 0; k < 32; ++k) {
        const int state = (k * 1973 + 17) & 0xFFFF;
        const double weight = static_cast<double>(k + 1);
        target_pmf[state] += weight;
        target_sum += weight;
    }
    for (int k = 0; k < 16; ++k) {
        const int state = (k * 37 + 9) & 0xFF;
        const double weight = static_cast<double>(2 * k + 1);
        context_pmf[state] += weight;
        context_sum += weight;
    }
    for (double& x : target_pmf) x /= target_sum;
    for (double& x : context_pmf) x /= context_sum;
    const DenseFactor local_corr = engine4.chi_local_from_pmfs(4, target_pmf, context_pmf);

    double local_factorization_defect = 0.0;
    for (int index = 0; index < (1 << 16); ++index) {
        int x = index;
        double product = 1.0;
        for (int local = 0; local < 4; ++local) {
            const int coordinate = x & 0xF;
            x >>= 4;
            product *= local_corr[coordinate << (4 * local)];
        }
        local_factorization_defect = std::max(
            local_factorization_defect,
            std::fabs(local_corr[index] - product)
        );
    }
    if (local_factorization_defect < 1e-8) {
        throw std::runtime_error(
            "BLOCK_SIZE=4 did not retain the synthetic within-block correlation"
        );
    }

    for (int sample = 0; sample < 64; ++sample) {
        const int coordinate = (sample * 1009 + 123) & 0xFFFF;
        double direct = 0.0;
        for (int x = 0; x < (1 << 16); ++x) {
            if (target_pmf[x] == 0.0) continue;
            for (int c = 0; c < 256; ++c) {
                if (context_pmf[c] == 0.0) continue;
                // Recompute the four output nibbles independently of the map table.
                const int x0 = x & 0xF;
                const int x1 = (x >> 4) & 0xF;
                const int x2 = (x >> 8) & 0xF;
                const int x3 = (x >> 12) & 0xF;
                const int x4 = c & 0xF;
                const int x5 = (c >> 4) & 0xF;
                const int y = chi_rep4_direct(x0, x1, x2)
                    | (chi_rep4_direct(x1, x2, x3) << 4)
                    | (chi_rep4_direct(x2, x3, x4) << 8)
                    | (chi_rep4_direct(x3, x4, x5) << 12);
                const double sign = parity_u32(static_cast<unsigned>(coordinate & y)) ? -1.0 : 1.0;
                direct += target_pmf[x] * context_pmf[c] * sign;
            }
        }
        if (std::fabs(direct - local_corr[coordinate]) > 2e-12) {
            throw std::runtime_error("BLOCK_SIZE=4 local chi direct-definition mismatch");
        }
    }

    // 5. A two-layer tail is cheap from a deterministic input and must be
    // finite.  It should generally differ from b=1, proving that b=4 is not
    // merely a regrouping of scalar products.
    const CorrVec block4_tail2 = engine4.tail_correlations(d0, d1, 2);
    double block_difference = 0.0;
    for (int i = 0; i < N; ++i) {
        if (!std::isfinite(block4_tail2[i])) throw std::runtime_error("non-finite BLOCK_SIZE=4 result");
        block_difference = std::max(block_difference, std::fabs(block4_tail2[i] - old_tail2[i]));
    }

    std::cout << std::scientific << std::setprecision(3)
              << "  FWT round-trip max error        : " << fwt_error << "\n"
              << "  b=1 initialization max error   : " << init_error << "\n"
              << "  b=1 chi max error              : " << chi_error << "\n"
              << "  b=1 linear max error           : " << linear_error << "\n"
              << "  b=1 two-layer tail max error   : " << tail_error << "\n"
              << "  synthetic block correlation defect: " << local_factorization_defect << "\n"
              << "  b=4 vs b=1 two-layer max diff  : " << block_difference << "\n"
              << "True BLOCK_SIZE=4 enlarged-unit self-tests: PASS\n"
              << std::defaultfloat << std::flush;
}

static std::string summarize_distribution(const Dist& dist) {
    std::map<int, int> wt;
    std::map<std::string, int> probcls;
    double mass = 0.0;
    for (const auto& kv : dist) {
        mass += kv.second;
        wt[popcount_mask(kv.first)]++;
        std::ostringstream key;
        key << std::fixed << std::setprecision(12) << std::log2(kv.second);
        probcls[key.str()]++;
    }
    std::ostringstream os;
    os << "{'num': " << dist.size()
       << ", 'mass': " << std::setprecision(17) << mass
       << ", 'prob_log2_classes': {";
    bool first = true;
    for (const auto& kv : probcls) {
        if (!first) os << ", "; first = false;
        os << kv.first << ": " << kv.second;
    }
    os << "}, 'weight_classes': {";
    first = true;
    for (const auto& kv : wt) {
        if (!first) os << ", "; first = false;
        os << kv.first << ": " << kv.second;
    }
    os << "}}";
    return os.str();
}


// -------------------- fixed parameters --------------------
static constexpr int FIXED_THREADS = 16;
static constexpr int FIXED_ROUNDS = 7;
static constexpr int FIXED_FRONT_ROUNDS = 2;
static const std::string FIXED_INPUT_D0 = "0";
static const std::string FIXED_INPUT_D1 = "32";
static constexpr double FIXED_PROGRESS_EVERY = 1000.0; // seconds between progress output
static const std::string FIXED_OUTPUT_JSON = "subterranean_second_order_fast_result.json";
// Keep this as 1 so every contribution is committed to global totals.
// Progress printing is controlled by FIXED_PROGRESS_EVERY, not by this counter.
static constexpr unsigned long long LOCAL_FLUSH_TAILS = 64;

//static_assert(BLOCK_SIZE == 4 || BLOCK_SIZE == 8, "BLOCK_SIZE must be 4 or 8");

struct FrontTask {
    Mask d0;
    Mask d1;
    double prob;
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

static void dfs_stream_front(
    int depth,
    const Mask& d0,
    const Mask& d1,
    double prob,
    int front_rounds,
    int tail_rounds,
    std::unordered_map<Mask, std::vector<std::pair<Mask,double>>, MaskHash>& chi_cache,
    std::unordered_map<Mask, Mask, MaskHash>& l_cache,
    std::atomic<unsigned long long>& expanded_nodes,
    std::atomic<unsigned long long>& pruned_conflict,
    std::atomic<unsigned long long>& pruned_prob,
    std::atomic<unsigned long long>& tail_evals,
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
        const CorrVec corr = tail_correlations_second_order(d0, d1, tail_rounds);
        for (int i = 0; i < N; ++i) {
            const double contribution = prob * corr[i];
            local_total[i] += contribution;
            local_abs[i] += std::fabs(contribution);
        }
        local_mass += prob;
        ++local_count;
        ++tail_evals;

        if (local_count >= LOCAL_FLUSH_TAILS) {
            flush_local_to_global(
                local_total, local_abs, local_mass, local_count,
                global_total, global_abs, global_mass, global_count
            );
        }
        return;
    }

    const auto& chi0 = chi_cached_ref(d0, chi_cache);
    const auto& chi1 = chi_cached_ref(d1, chi_cache);

    for (const auto& c0 : chi0) {
        const double p0 = prob * c0.second;
        if (!prob_survives(p0)) {
            ++pruned_prob;
            continue;
        }

        for (const auto& c1 : chi1) {
            ++expanded_nodes;
            const double np = p0 * c1.second;
            if (!prob_survives(np)) {
                ++pruned_prob;
                continue;
            }

            // Required pruning rule:
            // reject only when the two chi-output supports intersect.
            if (FRONT_REQUIRE_DISJOINT_EACH_ROUND &&
                first_conflict_between(c0.first, c1.first)) {
                ++pruned_conflict;
                continue;
            }

            const Mask L0 = L_cached(c0.first, l_cache);
            const Mask L1 = L_cached(c1.first, l_cache);

            dfs_stream_front(
                depth + 1, L0, L1, np,
                front_rounds, tail_rounds,
                chi_cache, l_cache,
                expanded_nodes, pruned_conflict, pruned_prob, tail_evals,
                global_total, global_abs, global_mass, global_count,
                local_total, local_abs, local_mass, local_count
            );
        }
    }
}

int main() {
    try {
        std::ios::sync_with_stdio(false);
#ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_max_active_levels(1);
        omp_set_num_threads(FIXED_THREADS);
#endif
        init_tables();

#if SUBTERRANEAN_SECOND_ORDER_VERIFY_FAST
        run_fast_chi_self_tests();
#endif
        run_true_block4_self_tests();

#if SUBTERRANEAN_SECOND_ORDER_SELF_TEST_ONLY
        std::cout << "self-test-only mode finished\n";
        return 0;
#endif

        const Mask d0 = parse_support(FIXED_INPUT_D0);
        const Mask d1 = parse_support(FIXED_INPUT_D1);
        const int tail_rounds = FIXED_ROUNDS - FIXED_FRONT_ROUNDS;
        if (tail_rounds < 0) {
            throw std::runtime_error("FIXED_ROUNDS must be >= FIXED_FRONT_ROUNDS");
        }

        std::cout << "Subterranean-2.0 second-order streaming hybrid DL estimation\n";
        std::cout << "  input d0                 : " << fmt_support(d0) << "\n";
        std::cout << "  input d1                 : " << fmt_support(d1) << "\n";
        std::cout << "  total chi layers         : " << FIXED_ROUNDS << "\n";
        std::cout << "  exact front chi/L steps  : " << FIXED_FRONT_ROUNDS << "\n";
        std::cout << "  geometric tail layers    : " << tail_rounds << "\n";
        std::cout << "  workers                  : " << FIXED_THREADS << "\n";
        std::cout << "  final HDL coordinate     : 7\n";
        std::cout << "  chi implementation       : true block-size-4 enlarged unit (65536 coordinates/block)\n";
        std::cout << "  outputs accumulated      : all 257 single-bit masks\n";
        std::cout << "  conflict mode            : reject intersecting D0/D1 chi outputs only\n";
        std::cout << "  progress interval        : " << FIXED_PROGRESS_EVERY << "s\n";
        std::cout << "  front log2 cutoff        : " << FRONT_LOG2_CUTOFF << "\n"
                  << std::flush;

        std::atomic<unsigned long long> expanded_nodes{0};
        std::atomic<unsigned long long> pruned_conflict{0};
        std::atomic<unsigned long long> pruned_prob{0};
        std::atomic<unsigned long long> tail_evals{0};
        std::atomic<bool> monitor_stop{false};

        CorrVec global_total{};
        CorrVec global_abs{};
        double global_mass = 0.0;
        unsigned long long global_count = 0;

        const auto start_time = std::chrono::steady_clock::now();

        // Build the small first-front-step task frontier.
        std::vector<FrontTask> tasks;
        if (FIXED_FRONT_ROUNDS == 0) {
            tasks.push_back({d0, d1, 1.0});
        } else {
            std::unordered_map<Mask, std::vector<std::pair<Mask,double>>, MaskHash> seed_chi_cache;
            std::unordered_map<Mask, Mask, MaskHash> seed_l_cache;

            const auto& chi0 = chi_cached_ref(d0, seed_chi_cache);
            const auto& chi1 = chi_cached_ref(d1, seed_chi_cache);

            for (const auto& c0 : chi0) {
                const double p0 = c0.second;
                if (!prob_survives(p0)) {
                    ++pruned_prob;
                    continue;
                }

                for (const auto& c1 : chi1) {
                    ++expanded_nodes;
                    const double np = p0 * c1.second;
                    if (!prob_survives(np)) {
                        ++pruned_prob;
                        continue;
                    }

                    if (FRONT_REQUIRE_DISJOINT_EACH_ROUND &&
                        first_conflict_between(c0.first, c1.first)) {
                        ++pruned_conflict;
                        continue;
                    }

                    const Mask L0 = L_cached(c0.first, seed_l_cache);
                    const Mask L1 = L_cached(c1.first, seed_l_cache);
                    tasks.push_back({L0, L1, np});
                }
            }
        }

        std::cout << "  first-step tasks           : " << tasks.size()
                  << " expanded=" << expanded_nodes.load()
                  << " pruned_conflict=" << pruned_conflict.load()
                  << " pruned_prob=" << pruned_prob.load()
                  << "\n" << std::flush;

        std::mutex monitor_mutex;
        std::condition_variable monitor_cv;

        std::thread monitor_thread([&]() {
            std::unique_lock<std::mutex> lock(monitor_mutex);

            while (!monitor_stop.load(std::memory_order_relaxed)) {
                const bool stopped =
                    monitor_cv.wait_for(
                        lock,
                        std::chrono::milliseconds(
                            static_cast<int>(
                                FIXED_PROGRESS_EVERY
                                * 1000.0
                            )
                        ),
                        [&]() {
                            return monitor_stop.load(
                                std::memory_order_relaxed
                            );
                        }
                    );

                if (stopped) {
                    break;
                }

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

                const auto best =
                    max_abs_position(snapshot);

                const double elapsed =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now()
                        - start_time
                    ).count();

#pragma omp critical(streaming_print)
                {
                    std::cout
                        << "  progress"
                        << " elapsed="
                        << std::fixed
                        << std::setprecision(3)
                        << elapsed
                        << "s"
                        << " expanded="
                        << expanded_nodes.load()
                        << " pruned_conflict="
                        << pruned_conflict.load()
                        << " pruned_prob="
                        << pruned_prob.load()
                        << " tail_seen="
                        << tail_evals.load()
                        << " tail_committed="
                        << snapshot_count
                        << " probability_mass="
                        << std::scientific
                        << std::setprecision(9)
                        << snapshot_mass
                        << " current_best_pos="
                        << std::defaultfloat
                        << best.first
                        << " current_best_corr="
                        << std::scientific
                        << std::setprecision(12)
                        << best.second
                        << " current_best_log2abs="
                        << std::defaultfloat
                        << std::setprecision(9)
                        << log2_abs_safe(best.second)
                        << " best_abs_sum="
                        << std::scientific
                        << std::setprecision(12)
                        << snapshot_abs[best.first]
                        << std::defaultfloat
                        << "\n"
                        << std::flush;
                }

                lock.lock();
            }
        });

        auto process_one_task = [&](
            const FrontTask& task,
            CorrVec& local_total,
            CorrVec& local_abs,
            double& local_mass,
            unsigned long long& local_count,
            std::unordered_map<
                Mask,
                std::vector<std::pair<Mask, double>>,
                MaskHash
            >& chi_cache,
            std::unordered_map<
                Mask,
                Mask,
                MaskHash
            >& l_cache
        ) {
            dfs_stream_front(
                FIXED_FRONT_ROUNDS == 0 ? 0 : 1,
                task.d0,
                task.d1,
                task.prob,
                FIXED_FRONT_ROUNDS,
                tail_rounds,
                chi_cache,
                l_cache,
                expanded_nodes,
                pruned_conflict,
                pruned_prob,
                tail_evals,
                global_total,
                global_abs,
                global_mass,
                global_count,
                local_total,
                local_abs,
                local_mass,
                local_count
            );
        };

        if (tasks.size() == 1) {
            // No outer OpenMP region: the fast chi/linear passes can use all
            // workers through their own parallel-for loops.
            std::unordered_map<
                Mask,
                std::vector<std::pair<Mask, double>>,
                MaskHash
            > chi_cache;

            std::unordered_map<
                Mask,
                Mask,
                MaskHash
            > l_cache;

            CorrVec local_total{};
            CorrVec local_abs{};
            double local_mass = 0.0;
            unsigned long long local_count = 0;

            process_one_task(
                tasks[0],
                local_total,
                local_abs,
                local_mass,
                local_count,
                chi_cache,
                l_cache
            );

            flush_local_to_global(
                local_total,
                local_abs,
                local_mass,
                local_count,
                global_total,
                global_abs,
                global_mass,
                global_count
            );
        } else {
            // Multiple endpoints: parallelize over endpoints.  Since nested
            // OpenMP is disabled, each tail's internal loops serialize and
            // threads work on different endpoints.
#pragma omp parallel
            {
                std::unordered_map<
                    Mask,
                    std::vector<std::pair<Mask, double>>,
                    MaskHash
                > chi_cache;

                std::unordered_map<
                    Mask,
                    Mask,
                    MaskHash
                > l_cache;

                chi_cache.reserve(16384);
                l_cache.reserve(65536);

                CorrVec local_total{};
                CorrVec local_abs{};
                double local_mass = 0.0;
                unsigned long long local_count = 0;

#pragma omp for schedule(dynamic, 1)
                for (size_t task_index = 0;
                     task_index < tasks.size();
                     ++task_index) {
                    process_one_task(
                        tasks[task_index],
                        local_total,
                        local_abs,
                        local_mass,
                        local_count,
                        chi_cache,
                        l_cache
                    );
                }

                flush_local_to_global(
                    local_total,
                    local_abs,
                    local_mass,
                    local_count,
                    global_total,
                    global_abs,
                    global_mass,
                    global_count
                );
            }
        }

        monitor_stop.store(true, std::memory_order_relaxed);
        monitor_cv.notify_all();

        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }

        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time
        ).count();

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
        std::cout << "  pruned conflicts         : " << pruned_conflict.load() << "\n";
        std::cout << "  pruned probabilities     : " << pruned_prob.load() << "\n";
        std::cout << "  elapsed seconds          : "
                  << std::setprecision(3) << elapsed << "\n";

        std::cout << "\nTop 20 single-bit output masks\n";
        std::cout << "rank  position  correlation              log2(abs)       sum_abs\n";
        for (int rank = 0; rank < 20; ++rank) {
            const int i = order[rank];
            std::cout << std::setw(4) << rank + 1
                      << "  " << std::setw(8) << i
                      << "  " << std::scientific << std::setprecision(12)
                      << global_total[i]
                      << "  " << std::fixed << std::setprecision(9)
                      << log2_abs_safe(global_total[i])
                      << "  " << std::scientific << std::setprecision(12)
                      << global_abs[i]
                      << std::defaultfloat << "\n";
        }

        std::cout << "\nFull single-bit result table\n";
        std::cout << "position correlation log2_abs sum_abs\n";
        for (int i = 0; i < N; ++i) {
            std::cout << i << " "
                      << std::scientific << std::setprecision(17)
                      << global_total[i] << " "
                      << std::defaultfloat << std::setprecision(12)
                      << log2_abs_safe(global_total[i]) << " "
                      << std::scientific << std::setprecision(17)
                      << global_abs[i]
                      << std::defaultfloat << "\n";
        }

        std::ofstream json(FIXED_OUTPUT_JSON);
        json << std::setprecision(17);
        json << "{\n";
        json << "  \"cipher\": \"Subterranean-2.0\",\n";
        json << "  \"order\": 2,\n";
        json << "  \"chi_implementation\": \"fast Walsh/rank-one\",\n";
        json << "  \"rounds\": " << FIXED_ROUNDS << ",\n";
        json << "  \"front_rounds\": " << FIXED_FRONT_ROUNDS << ",\n";
        json << "  \"tail_rounds\": " << tail_rounds << ",\n";
        json << "  \"input_d0\": [" << FIXED_INPUT_D0 << "],\n";
        json << "  \"input_d1\": [" << FIXED_INPUT_D1 << "],\n";
        json << "  \"probability_mass_used\": " << global_mass << ",\n";
        json << "  \"tail_evaluations\": " << global_count << ",\n";
        json << "  \"expanded_front_nodes\": " << expanded_nodes.load() << ",\n";
        json << "  \"pruned_conflicts\": " << pruned_conflict.load() << ",\n";
        json << "  \"elapsed_seconds\": " << elapsed << ",\n";
        json << "  \"single_bit_results\": [\n";
        for (int i = 0; i < N; ++i) {
            json << "    {\"position\": " << i
                 << ", \"correlation\": " << global_total[i]
                 << ", \"log2_abs\": " << log2_abs_safe(global_total[i])
                 << ", \"sum_abs\": " << global_abs[i] << "}";
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

