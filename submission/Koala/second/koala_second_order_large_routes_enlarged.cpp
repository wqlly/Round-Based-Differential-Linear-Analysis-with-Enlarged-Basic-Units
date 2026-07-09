// koala_second_order_large_routes.cpp
//
// Optimized second-order hybrid differential-linear evaluation for
// Koala-p.
//
// Scientific model:
//   - exact front: unchanged from the supplied second_sub.cpp;
//   - geometric tail: genuine block-wise enlarged-unit second-order round-based model;
//   - final HDL coordinate: 7 = 0b0111;
//   - all 257 one-bit output masks are accumulated simultaneously.
//
// Koala-p linear layer:
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
//   In addition to the fast geometric tail, this version avoids evaluating
//   every exact-front route.  It rejects whole chi-product subtrees by a
//   rigorous probability upper bound, keeps a bounded structural beam, merges
//   duplicate endpoints, scores them with a short geometric-tail proxy, and
//   runs the full tail only for the best candidates.
//
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
//   Compile with KOALA_SECOND_ORDER_VERIFY_FAST=1 to retain the original
//   direct 16x16 matrix implementation and compare it against the fast
//   implementation on initial and post-linear gamma states.
//
// Default effective parameters are preserved from second_sub.cpp:
//   total chi layers = 7
//   exact front      = 1
//   geometric tail  = 6
//   D0=[0], D1=[11]
//
// Note: the original file name/comment says "2+5", but its actual constant
// FIXED_FRONT_ROUNDS is 1. Change it to 2 if a genuine 2+5 run is intended.
//
// Compile normal:
//   g++ -O3 -march=native -std=c++17 -fopenmp \
//       koala_second_order_large_routes.cpp \
//       -o koala_second_large_routes -lm
//
// Compile and run direct-vs-fast self-test:
//   g++ -O3 -march=native -std=c++17 -fopenmp \
//       -DKOALA_SECOND_ORDER_VERIFY_FAST=1 \
//       -DKOALA_SECOND_ORDER_SELF_TEST_ONLY=1 \
//       koala_second_order_large_routes.cpp \
//       -o koala_second_large_routes_selftest -lm
//   ./koala_second_large_routes_selftest
//

#ifndef KOALA_SECOND_ORDER_VERIFY_FAST
#define KOALA_SECOND_ORDER_VERIFY_FAST 0
#endif

#ifndef KOALA_SECOND_ORDER_SELF_TEST_ONLY
#define KOALA_SECOND_ORDER_SELF_TEST_ONLY 0
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
#include <queue>
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
// The geometric tail below uses genuine enlarged chi units.
// ENLARGED_BLOCK_SIZE is defined in the tail implementation and defaults to 2.
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

// 121^{-1} mod 257 = 17.
static constexpr int INV121 = 17;

static Mask L_transform(const Mask& in) {
    // Koala-p L = theta o pi:
    //   z_i = x_{121i} xor x_{121(i+3)} xor x_{121(i+10)}.
    // Therefore input bit p contributes to output positions
    //   17*p, 17*p-3 and 17*p-10 modulo 257.
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


// Rank of the affine output space of the chi differential.  This computes
// only the rank and does not enumerate the 2^rank outputs.  It is used to
// reject an entire subtree before allocating a huge chi distribution.
static int chi_diff_rank(const Mask& delta) {
    std::vector<Mask> cols;
    cols.reserve(2 * popcount_mask(delta));
    for (int j = 0; j < N; ++j) {
        Mask col;
        if (test_bit(delta, j - 1)) xor_bit(col, j - 2);
        if (test_bit(delta, j + 1)) xor_bit(col, j - 1);
        if (!is_zero(col)) cols.push_back(col);
    }
    return static_cast<int>(gf2_independent_basis(cols).size());
}

static inline int chi_rank_cached(
    const Mask& d,
    std::unordered_map<Mask, int, MaskHash>& cache
) {
    auto it = cache.find(d);
    if (it != cache.end()) return it->second;
    const int rank = chi_diff_rank(d);
    cache.emplace(d, rank);
    return rank;
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

                    if (
                        FRONT_REQUIRE_DISJOINT_EACH_ROUND
                        && first_conflict_between(c0.first, c1.first)
                    ) {
                        continue;
                    }

                    const Mask L1 = L_cached(c1.first, local_L_cache);
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


// -------------------- genuine enlarged-unit second-order geometric tail --------------------
//
// The old programs stored one 16-coordinate vector per bit.  Merely grouping
// the final product did not enlarge the computational unit.  This replacement
// stores one joint vector of size 16^m for every block of m consecutive bits.
//
// For one chi propagation and one output-block coordinate V_B:
//   1. each input block B_k is contracted into a 16x16 transfer matrix
//          T_k(V_B) = sum_{U_k} gamma_k[U_k]
//                       prod_{r in B_k} M_r(V_B,U_k);
//   2. the cyclic product trace tr(T_{q-1} ... T_0) is evaluated;
//   3. all coordinates of the selected output block are retained together.
//
// Thus the independence approximation is applied between enlarged blocks,
// not between individual bits.  ENLARGED_BLOCK_SIZE=1 reduces to the original
// bit-wise round-based model; the practical enlarged setting is 2.
//
// A block of four bits would require 16^4=65536 coordinates per block and is
// intentionally rejected here because these programs evaluate the tail once
// per exact-front endpoint.  The implementation can be extended mechanically,
// but the resulting route search is not computationally practical.

static constexpr int SECOND_DIM = 16;
static constexpr int SECOND_FINAL_COORD = 7;
static constexpr int ENLARGED_BLOCK_SIZE = 2;
static_assert(
    ENLARGED_BLOCK_SIZE == 1 || ENLARGED_BLOCK_SIZE == 2,
    "This verified implementation supports block size 1 or 2."
);

using CorrVec = std::array<double, N>;
using Row16 = std::array<double, SECOND_DIM>;

static std::array<std::array<int, 3>, N> L_ROWS;

struct Mat16 {
    std::array<double, SECOND_DIM * SECOND_DIM> a{};
};

struct FloatKernel {
    int length = 0;
    int dimension = 0;

    // target[(v * dimension + u) * 256 + entry]:
    // block product when the output coordinate is localized in this block.
    std::vector<float> target;

    // next[(last_v * dimension + u) * 256 + entry]:
    // block product for the block immediately following the target block.
    std::vector<float> next;
};

struct BlockInfo {
    int start = 0;
    int length = 0;
    int dimension = 0;
};

struct EnlargedGamma {
    std::vector<std::vector<double>> value;
};

static std::vector<BlockInfo> E_BLOCKS;
static std::array<int, N> E_POS_BLOCK{};
static std::array<int, N> E_POS_OFFSET{};

static double E_MT[4][2][2];
static std::vector<float> E_LOCAL_BIT; // [v_prev][v_cur][u][entry]
static FloatKernel E_KERNEL1;
static FloatKernel E_KERNEL2;

static inline int parity4(int x) noexcept {
    return __builtin_popcount(static_cast<unsigned>(x)) & 1;
}

static inline int nibble_at(int packed, int offset) noexcept {
    return (packed >> (4 * offset)) & 0xF;
}

static inline void mat_identity(Mat16& m) {
    m.a.fill(0.0);
    for (int i = 0; i < SECOND_DIM; ++i) {
        m.a[i * SECOND_DIM + i] = 1.0;
    }
}

static inline void matmul16(
    const Mat16& left,
    const Mat16& right,
    Mat16& output
) {
    Mat16 temp;
    for (int i = 0; i < SECOND_DIM; ++i) {
        for (int j = 0; j < SECOND_DIM; ++j) {
            double sum = 0.0;
            for (int k = 0; k < SECOND_DIM; ++k) {
                sum +=
                    left.a[i * SECOND_DIM + k]
                    * right.a[k * SECOND_DIM + j];
            }
            temp.a[i * SECOND_DIM + j] = sum;
        }
    }
    output = temp;
}

static inline double trace_product(
    const Mat16& left,
    const Mat16& right
) {
    // tr(left * right)
    double sum = 0.0;
    for (int i = 0; i < SECOND_DIM; ++i) {
        for (int k = 0; k < SECOND_DIM; ++k) {
            sum +=
                left.a[i * SECOND_DIM + k]
                * right.a[k * SECOND_DIM + i];
        }
    }
    return sum;
}

static inline const float* local_bit_ptr(
    int v_previous,
    int v_current,
    int u_current
) {
    const size_t index =
        (
            (
                static_cast<size_t>(v_previous) * SECOND_DIM
                + static_cast<size_t>(v_current)
            ) * SECOND_DIM
            + static_cast<size_t>(u_current)
        ) * SECOND_DIM * SECOND_DIM;

    return E_LOCAL_BIT.data() + index;
}

static void construct_local_bit_matrix(
    int v_previous,
    int v_current,
    int u_current,
    float* output
) {
    int index[4];

    const int pv_previous = parity4(v_previous);
    const int pv_current = parity4(v_current);
    const int pu_current = parity4(u_current);

    index[0] =
        (pv_previous << 1)
        | (pu_current ^ pv_current);

    for (int component = 1; component < 4; ++component) {
        const int shift = 3 - component;
        const int a = (v_previous >> shift) & 1;
        const int b =
            ((u_current >> shift) & 1)
            ^ ((v_current >> shift) & 1);

        index[component] = (a << 1) | b;
    }

    for (int row = 0; row < SECOND_DIM; ++row) {
        for (int column = 0; column < SECOND_DIM; ++column) {
            double value = 1.0;

            for (int component = 0; component < 4; ++component) {
                const int state_row =
                    (row >> (3 - component)) & 1;
                const int state_column =
                    (column >> (3 - component)) & 1;

                value *=
                    E_MT[index[component]]
                        [state_row]
                        [state_column];
            }

            output[row * SECOND_DIM + column] =
                static_cast<float>(value);
        }
    }
}

static void multiply_float_matrices(
    const float* left,
    const float* right,
    float* output
) {
    for (int i = 0; i < SECOND_DIM; ++i) {
        for (int j = 0; j < SECOND_DIM; ++j) {
            double sum = 0.0;
            for (int k = 0; k < SECOND_DIM; ++k) {
                sum +=
                    static_cast<double>(
                        left[i * SECOND_DIM + k]
                    )
                    * static_cast<double>(
                        right[k * SECOND_DIM + j]
                    );
            }
            output[i * SECOND_DIM + j] =
                static_cast<float>(sum);
        }
    }
}

static FloatKernel make_kernel(int length) {
    FloatKernel kernel;
    kernel.length = length;
    kernel.dimension = 1 << (4 * length);

    const size_t entries = SECOND_DIM * SECOND_DIM;

    kernel.target.resize(
        static_cast<size_t>(kernel.dimension)
        * kernel.dimension
        * entries
    );

    kernel.next.resize(
        static_cast<size_t>(SECOND_DIM)
        * kernel.dimension
        * entries
    );

    std::array<float, SECOND_DIM * SECOND_DIM> product{};

    // Output coordinate localized in this block.
    for (int v_packed = 0;
         v_packed < kernel.dimension;
         ++v_packed) {
        for (int u_packed = 0;
             u_packed < kernel.dimension;
             ++u_packed) {
            const float* coefficient = nullptr;

            if (length == 1) {
                coefficient =
                    local_bit_ptr(
                        0,
                        nibble_at(v_packed, 0),
                        nibble_at(u_packed, 0)
                    );

                std::copy(
                    coefficient,
                    coefficient + entries,
                    product.begin()
                );
            } else {
                const int v0 = nibble_at(v_packed, 0);
                const int v1 = nibble_at(v_packed, 1);
                const int u0 = nibble_at(u_packed, 0);
                const int u1 = nibble_at(u_packed, 1);

                const float* first =
                    local_bit_ptr(0, v0, u0);
                const float* second =
                    local_bit_ptr(v0, v1, u1);

                // State positions increase from left to right, while the
                // cyclic transfer product is M_end ... M_start.
                multiply_float_matrices(
                    second,
                    first,
                    product.data()
                );
            }

            const size_t base =
                (
                    static_cast<size_t>(v_packed)
                    * kernel.dimension
                    + static_cast<size_t>(u_packed)
                ) * entries;

            std::copy(
                product.begin(),
                product.end(),
                kernel.target.begin() + base
            );
        }
    }

    // Block immediately after the localized output block.  Only the previous
    // output coordinate at the first bit can be nonzero.
    for (int previous_v = 0;
         previous_v < SECOND_DIM;
         ++previous_v) {
        for (int u_packed = 0;
             u_packed < kernel.dimension;
             ++u_packed) {
            if (length == 1) {
                const float* coefficient =
                    local_bit_ptr(
                        previous_v,
                        0,
                        nibble_at(u_packed, 0)
                    );

                std::copy(
                    coefficient,
                    coefficient + entries,
                    product.begin()
                );
            } else {
                const int u0 = nibble_at(u_packed, 0);
                const int u1 = nibble_at(u_packed, 1);

                const float* first =
                    local_bit_ptr(previous_v, 0, u0);
                const float* second =
                    local_bit_ptr(0, 0, u1);

                multiply_float_matrices(
                    second,
                    first,
                    product.data()
                );
            }

            const size_t base =
                (
                    static_cast<size_t>(previous_v)
                    * kernel.dimension
                    + static_cast<size_t>(u_packed)
                ) * entries;

            std::copy(
                product.begin(),
                product.end(),
                kernel.next.begin() + base
            );
        }
    }

    return kernel;
}

static inline const FloatKernel& kernel_for_length(int length) {
    return length == 1 ? E_KERNEL1 : E_KERNEL2;
}

static void init_tables() {
    for (int i = 0; i < N; ++i) {
        // Koala-p mask pullback:
        // L^T e_i =
        //   e_{121i} + e_{121(i+3)} + e_{121(i+10)}.
        L_ROWS[i] = {
            modN(121LL * i),
            modN(121LL * (i + 3)),
            modN(121LL * (i + 10))
        };
    }

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

    std::memcpy(E_MT[0], M00, sizeof(M00));
    std::memcpy(E_MT[1], M01, sizeof(M01));
    std::memcpy(E_MT[2], M10, sizeof(M10));
    std::memcpy(E_MT[3], M11, sizeof(M11));

    E_BLOCKS.clear();
    for (int start = 0; start < N; start += ENLARGED_BLOCK_SIZE) {
        const int length =
            std::min(ENLARGED_BLOCK_SIZE, N - start);

        const int block_index =
            static_cast<int>(E_BLOCKS.size());

        E_BLOCKS.push_back(
            BlockInfo{
                start,
                length,
                1 << (4 * length)
            }
        );

        for (int offset = 0; offset < length; ++offset) {
            E_POS_BLOCK[start + offset] = block_index;
            E_POS_OFFSET[start + offset] = offset;
        }
    }

    E_LOCAL_BIT.resize(
        static_cast<size_t>(SECOND_DIM)
        * SECOND_DIM
        * SECOND_DIM
        * SECOND_DIM
        * SECOND_DIM
    );

    for (int v_previous = 0;
         v_previous < SECOND_DIM;
         ++v_previous) {
        for (int v_current = 0;
             v_current < SECOND_DIM;
             ++v_current) {
            for (int u_current = 0;
                 u_current < SECOND_DIM;
                 ++u_current) {
                float* destination =
                    const_cast<float*>(
                        local_bit_ptr(
                            v_previous,
                            v_current,
                            u_current
                        )
                    );

                construct_local_bit_matrix(
                    v_previous,
                    v_current,
                    u_current,
                    destination
                );
            }
        }
    }

    E_KERNEL1 = make_kernel(1);

    if constexpr (ENLARGED_BLOCK_SIZE == 2) {
        E_KERNEL2 = make_kernel(2);
    }
}

static void fwt_inplace(Row16& row) {
    for (int half = 1; half < SECOND_DIM; half <<= 1) {
        for (int start = 0;
             start < SECOND_DIM;
             start += 2 * half) {
            for (int offset = 0;
                 offset < half;
                 ++offset) {
                const double left =
                    row[start + offset];
                const double right =
                    row[start + offset + half];

                row[start + offset] = left + right;
                row[start + offset + half] = left - right;
            }
        }
    }
}

static std::array<Row16, N> initial_bit_rows(
    const Mask& d0,
    const Mask& d1
) {
    std::array<Row16, N> rows{};

    for (int position = 0;
         position < N;
         ++position) {
        const int a = test_bit(d0, position);
        const int b = test_bit(d1, position);
        const int ab = a ^ b;

        Row16& row = rows[position];
        row.fill(0.0);

        row[(0 << 3) | (a << 2) | (b << 1) | ab] = 1.0;
        row[(1 << 3) | (a << 2) | (b << 1) | ab] = 1.0;

        fwt_inplace(row);

        for (double& value : row) {
            value *= 0.5;
        }
    }

    return rows;
}

static EnlargedGamma init_gamma_second_order(
    const Mask& d0,
    const Mask& d1
) {
    const auto bit_rows =
        initial_bit_rows(d0, d1);

    EnlargedGamma gamma;
    gamma.value.resize(E_BLOCKS.size());

#pragma omp parallel for schedule(static)
    for (int block_index = 0;
         block_index < static_cast<int>(E_BLOCKS.size());
         ++block_index) {
        const BlockInfo& block =
            E_BLOCKS[block_index];

        std::vector<double>& local =
            gamma.value[block_index];

        local.resize(block.dimension);

        for (int packed = 0;
             packed < block.dimension;
             ++packed) {
            double value = 1.0;

            for (int offset = 0;
                 offset < block.length;
                 ++offset) {
                value *=
                    bit_rows[block.start + offset]
                        [nibble_at(packed, offset)];
            }

            local[packed] = value;
        }

        // The zero correlation coordinate is exact.
        local[0] = 1.0;
    }

    return gamma;
}

static Mat16 weighted_kernel_matrix(
    const std::vector<double>& gamma_block,
    const FloatKernel& kernel,
    bool target_mode,
    int selector
) {
    Mat16 result;

    const size_t entries =
        SECOND_DIM * SECOND_DIM;

    const std::vector<float>& coefficients =
        target_mode
            ? kernel.target
            : kernel.next;

    for (int u = 0;
         u < kernel.dimension;
         ++u) {
        const double weight =
            gamma_block[u];

        if (weight == 0.0) {
            continue;
        }

        const size_t base =
            (
                static_cast<size_t>(selector)
                * kernel.dimension
                + static_cast<size_t>(u)
            ) * entries;

        const float* source =
            coefficients.data() + base;

#pragma omp simd
        for (int entry = 0;
             entry < SECOND_DIM * SECOND_DIM;
             ++entry) {
            result.a[entry] +=
                weight
                * static_cast<double>(source[entry]);
        }
    }

    return result;
}

struct WeightedBlockMatrices {
    std::vector<std::vector<Mat16>> target;
    std::vector<std::array<Mat16, SECOND_DIM>> next;
    std::vector<Mat16> prefix;
    std::vector<Mat16> suffix;
    Mat16 wrap_middle;
};

static WeightedBlockMatrices precompute_weighted_block_matrices(
    const EnlargedGamma& gamma
) {
    const int block_count =
        static_cast<int>(E_BLOCKS.size());

    WeightedBlockMatrices weighted;
    weighted.target.resize(block_count);
    weighted.next.resize(block_count);

#pragma omp parallel for schedule(dynamic, 1)
    for (int block_index = 0;
         block_index < block_count;
         ++block_index) {
        const BlockInfo& block =
            E_BLOCKS[block_index];

        const FloatKernel& kernel =
            kernel_for_length(block.length);

        weighted.target[block_index].resize(
            block.dimension
        );

        for (int v = 0;
             v < block.dimension;
             ++v) {
            weighted.target[block_index][v] =
                weighted_kernel_matrix(
                    gamma.value[block_index],
                    kernel,
                    true,
                    v
                );
        }

        for (int previous_v = 0;
             previous_v < SECOND_DIM;
             ++previous_v) {
            weighted.next[block_index][previous_v] =
                weighted_kernel_matrix(
                    gamma.value[block_index],
                    kernel,
                    false,
                    previous_v
                );
        }
    }

    weighted.prefix.resize(block_count + 1);
    weighted.suffix.resize(block_count + 1);

    mat_identity(weighted.prefix[0]);

    for (int k = 1;
         k <= block_count;
         ++k) {
        matmul16(
            weighted.target[k - 1][0],
            weighted.prefix[k - 1],
            weighted.prefix[k]
        );
    }

    mat_identity(weighted.suffix[block_count]);

    for (int k = block_count - 1;
         k >= 0;
         --k) {
        matmul16(
            weighted.suffix[k + 1],
            weighted.target[k][0],
            weighted.suffix[k]
        );
    }

    // D_{q-2} ... D_1, used by the wrap-around replacement
    // (target block q-1, following block 0).
    mat_identity(weighted.wrap_middle);

    for (int k = 1;
         k <= block_count - 2;
         ++k) {
        Mat16 temporary;
        matmul16(
            weighted.target[k][0],
            weighted.wrap_middle,
            temporary
        );
        weighted.wrap_middle = temporary;
    }

    return weighted;
}

static double enlarged_chi_coordinate(
    const WeightedBlockMatrices& weighted,
    int target_block,
    int packed_output
) {
    if (packed_output == 0) {
        return 1.0;
    }

    const int block_count =
        static_cast<int>(E_BLOCKS.size());

    const int following_block =
        (target_block + 1) % block_count;

    const BlockInfo& target_info =
        E_BLOCKS[target_block];

    const int last_output_coordinate =
        nibble_at(
            packed_output,
            target_info.length - 1
        );

    const Mat16& target_matrix =
        weighted.target[target_block][packed_output];

    const Mat16& following_matrix =
        weighted.next[following_block]
                     [last_output_coordinate];

    Mat16 pair_product;

    // Pair in cyclic order: following * target.
    matmul16(
        following_matrix,
        target_matrix,
        pair_product
    );

    if (target_block < block_count - 1) {
        Mat16 environment;

        // prefix[target] * suffix[target+2] is a cyclic rotation of
        // the remaining default block product.
        matmul16(
            weighted.prefix[target_block],
            weighted.suffix[target_block + 2],
            environment
        );

        return
            trace_product(
                pair_product,
                environment
            );
    }

    // Wrap case:
    // T_{q-1} D_{q-2} ... D_1 T_0.
    // Cyclically rotate to (T_0 T_{q-1})(D_{q-2} ... D_1).
    return
        trace_product(
            pair_product,
            weighted.wrap_middle
        );
}

static EnlargedGamma pass_chi_second_order_enlarged(
    const EnlargedGamma& gamma
) {
    const WeightedBlockMatrices weighted =
        precompute_weighted_block_matrices(gamma);

    EnlargedGamma output;
    output.value.resize(E_BLOCKS.size());

#pragma omp parallel for schedule(dynamic, 1)
    for (int block_index = 0;
         block_index < static_cast<int>(E_BLOCKS.size());
         ++block_index) {
        const int dimension =
            E_BLOCKS[block_index].dimension;

        std::vector<double>& local =
            output.value[block_index];

        local.resize(dimension);
        local[0] = 1.0;

        for (int packed = 1;
             packed < dimension;
             ++packed) {
            local[packed] =
                enlarged_chi_coordinate(
                    weighted,
                    block_index,
                    packed
                );
        }
    }

    return output;
}

static EnlargedGamma pass_linear_second_order_enlarged(
    const EnlargedGamma& gamma
) {
    EnlargedGamma output;
    output.value.resize(E_BLOCKS.size());

#pragma omp parallel for schedule(dynamic, 1)
    for (int output_block = 0;
         output_block < static_cast<int>(E_BLOCKS.size());
         ++output_block) {
        const BlockInfo& block =
            E_BLOCKS[output_block];

        std::vector<double>& local =
            output.value[output_block];

        local.resize(block.dimension);
        local[0] = 1.0;

        for (int packed_output = 1;
             packed_output < block.dimension;
             ++packed_output) {
            // At most 2 output positions, each touching 3 input positions.
            std::array<int, 6> touched_block{};
            std::array<int, 6> touched_coordinate{};
            int touched_count = 0;

            for (int offset = 0;
                 offset < block.length;
                 ++offset) {
                const int coordinate =
                    nibble_at(packed_output, offset);

                if (coordinate == 0) {
                    continue;
                }

                const int output_position =
                    block.start + offset;

                for (int branch = 0;
                     branch < 3;
                     ++branch) {
                    const int input_position =
                        L_ROWS[output_position][branch];

                    const int input_block =
                        E_POS_BLOCK[input_position];

                    const int input_offset =
                        E_POS_OFFSET[input_position];

                    int slot = -1;

                    for (int k = 0;
                         k < touched_count;
                         ++k) {
                        if (touched_block[k] == input_block) {
                            slot = k;
                            break;
                        }
                    }

                    if (slot < 0) {
                        slot = touched_count++;
                        touched_block[slot] = input_block;
                        touched_coordinate[slot] = 0;
                    }

                    touched_coordinate[slot] ^=
                        coordinate << (4 * input_offset);
                }
            }

            double value = 1.0;

            for (int k = 0;
                 k < touched_count;
                 ++k) {
                value *=
                    gamma.value[touched_block[k]]
                               [touched_coordinate[k]];
            }

            local[packed_output] = value;
        }
    }

    return output;
}

static CorrVec extract_one_bit_coordinates(
    const EnlargedGamma& gamma
) {
    CorrVec output{};

    for (int position = 0;
         position < N;
         ++position) {
        const int block =
            E_POS_BLOCK[position];
        const int offset =
            E_POS_OFFSET[position];

        const int packed =
            SECOND_FINAL_COORD
            << (4 * offset);

        output[position] =
            gamma.value[block][packed];
    }

    return output;
}

static CorrVec final_chi_correlations_second_order(
    const EnlargedGamma& gamma
) {
    const WeightedBlockMatrices weighted =
        precompute_weighted_block_matrices(gamma);

    CorrVec output{};

#pragma omp parallel for schedule(static)
    for (int position = 0;
         position < N;
         ++position) {
        const int block =
            E_POS_BLOCK[position];
        const int offset =
            E_POS_OFFSET[position];

        const int packed =
            SECOND_FINAL_COORD
            << (4 * offset);

        output[position] =
            enlarged_chi_coordinate(
                weighted,
                block,
                packed
            );
    }

    return output;
}

static CorrVec tail_correlations_second_order(
    const Mask& d0,
    const Mask& d1,
    int tail_rounds
) {
    EnlargedGamma gamma =
        init_gamma_second_order(d0, d1);

    if (tail_rounds == 0) {
        return extract_one_bit_coordinates(gamma);
    }

    for (int round = 0;
         round + 1 < tail_rounds;
         ++round) {
        gamma =
            pass_chi_second_order_enlarged(gamma);

        gamma =
            pass_linear_second_order_enlarged(gamma);
    }

    return
        final_chi_correlations_second_order(gamma);
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

// -------------------- enlarged-unit correctness checks --------------------

static Mat16 direct_bit_matrix(
    const Row16& gamma_row,
    int v_previous,
    int v_current
) {
    Mat16 result;

    for (int u = 0;
         u < SECOND_DIM;
         ++u) {
        const double weight =
            gamma_row[u];

        const float* coefficient =
            local_bit_ptr(
                v_previous,
                v_current,
                u
            );

        for (int entry = 0;
             entry < SECOND_DIM * SECOND_DIM;
             ++entry) {
            result.a[entry] +=
                weight
                * static_cast<double>(coefficient[entry]);
        }
    }

    return result;
}

static double direct_initial_chi_coordinate(
    const Mask& d0,
    const Mask& d1,
    int target_block,
    int packed_output
) {
    const auto rows =
        initial_bit_rows(d0, d1);

    Mat16 product;
    mat_identity(product);

    for (int position = 0;
         position < N;
         ++position) {
        int v_current = 0;
        int v_previous = 0;

        if (E_POS_BLOCK[position] == target_block) {
            v_current =
                nibble_at(
                    packed_output,
                    E_POS_OFFSET[position]
                );
        }

        const int previous_position =
            (position + N - 1) % N;

        if (E_POS_BLOCK[previous_position] == target_block) {
            v_previous =
                nibble_at(
                    packed_output,
                    E_POS_OFFSET[previous_position]
                );
        }

        const Mat16 local =
            direct_bit_matrix(
                rows[position],
                v_previous,
                v_current
            );

        Mat16 temporary;
        matmul16(local, product, temporary);
        product = temporary;
    }

    double trace = 0.0;
    for (int i = 0; i < SECOND_DIM; ++i) {
        trace += product.a[i * SECOND_DIM + i];
    }

    return trace;
}

static void verify_koala_linear_layer() {
    for (int p = 0; p < N; ++p) {
        Mask basis;
        xor_bit(basis, p);

        const Mask formula =
            L_transform(basis);

        Mask explicit_image;

        for (int i = 0; i < N; ++i) {
            const int value =
                (p == modN(121LL * i))
                ^ (p == modN(121LL * (i + 3)))
                ^ (p == modN(121LL * (i + 10)));

            if (value != 0) {
                xor_bit(explicit_image, i);
            }
        }

        if (!(formula == explicit_image)) {
            throw std::runtime_error(
                "Koala-p forward linear self-test failed at bit "
                + std::to_string(p)
            );
        }
    }

    for (int output_position = 0;
         output_position < N;
         ++output_position) {
        for (int input_position = 0;
             input_position < N;
             ++input_position) {
            const bool explicit_coefficient =
                input_position
                    == modN(121LL * output_position)
                || input_position
                    == modN(121LL * (output_position + 3))
                || input_position
                    == modN(121LL * (output_position + 10));

            const bool row_coefficient =
                input_position
                    == L_ROWS[output_position][0]
                || input_position
                    == L_ROWS[output_position][1]
                || input_position
                    == L_ROWS[output_position][2];

            if (explicit_coefficient != row_coefficient) {
                throw std::runtime_error(
                    "Koala-p L^T row self-test failed"
                );
            }
        }
    }
}

static void run_fast_chi_self_tests() {
    std::cout
        << "Koala-p genuine enlarged-unit second-order self-tests\n"
        << "  block size: " << ENLARGED_BLOCK_SIZE << "\n"
        << std::flush;

    verify_koala_linear_layer();

    const std::array<std::pair<std::string, std::string>, 2>
        direction_pairs = {{
            {"0", "11"},
            {"0,7", "11,93"}
        }};

    for (const auto& pair : direction_pairs) {
        const Mask d0 =
            parse_support(pair.first);
        const Mask d1 =
            parse_support(pair.second);

        const EnlargedGamma gamma =
            init_gamma_second_order(d0, d1);

        const WeightedBlockMatrices weighted =
            precompute_weighted_block_matrices(gamma);

        const int test_block =
            std::min<int>(
                5,
                static_cast<int>(E_BLOCKS.size()) - 1
            );

        std::vector<int> coordinates;

        coordinates.push_back(SECOND_FINAL_COORD);

        if (E_BLOCKS[test_block].length == 2) {
            coordinates.push_back(
                SECOND_FINAL_COORD
                | (3 << 4)
            );
            coordinates.push_back(
                5
                | (SECOND_FINAL_COORD << 4)
            );
        }

        for (int packed : coordinates) {
            const double enlarged =
                enlarged_chi_coordinate(
                    weighted,
                    test_block,
                    packed
                );

            const double direct =
                direct_initial_chi_coordinate(
                    d0,
                    d1,
                    test_block,
                    packed
                );

            const double error =
                std::fabs(enlarged - direct);

            if (error > 2e-11) {
                throw std::runtime_error(
                    "enlarged chi self-test mismatch: error="
                    + std::to_string(error)
                );
            }
        }
    }

    std::cout
        << "  forward L and L^T rows       : PASS\n"
        << "  grouped chi block contraction: PASS\n"
        << "  exact first-chi coordinates  : PASS\n"
        << std::flush;
}


// -------------------- fixed parameters --------------------
static constexpr int FIXED_THREADS = 16;
static constexpr int FIXED_ROUNDS =7;
static constexpr int FIXED_FRONT_ROUNDS = 2;
static const std::string FIXED_INPUT_D0 = "0";
static const std::string FIXED_INPUT_D1 = "11";
static constexpr double FIXED_PROGRESS_EVERY = 10.0; // seconds between progress output
static const std::string FIXED_OUTPUT_JSON = "koala_second_order_large_routes_result.json";
static const std::string FIXED_ROUTE_OUTPUT = "koala_second_order_selected_routes.txt";
// Keep this as 1 so every contribution is committed to global totals.
// Progress printing is controlled by FIXED_PROGRESS_EVERY, not by this counter.
static constexpr unsigned long long LOCAL_FLUSH_TAILS = 64;


// -------------------- large-contribution route selection --------------------
// The full tail is NOT evaluated for every exact-front route.  Selection has
// three stages:
//   1. rigorous probability upper bound: |p*C_tail| <= p;
//   2. cheap structural prefilter: prefer high p and lower endpoint activity;
//   3. short geometric-tail proxy, then retain only the best routes.
//
// These controls intentionally make the computation approximate.  Increase
// the two Top-K values and/or decrease FRONT_LOG2_CUTOFF to approach the full
// result.  Setting ROUTE_PREFILTER_TOP_K and ROUTE_FINAL_TOP_K very large
// effectively disables the beam restriction.
static constexpr std::size_t ROUTE_PREFILTER_TOP_K = 2000000;
static constexpr std::size_t ROUTE_FINAL_TOP_K = 200000;
static constexpr int ROUTE_PROXY_ROUNDS = 4;
static constexpr double ROUTE_KEEP_WITHIN_BITS = 20.0;
static constexpr double ROUTE_WEIGHT_PENALTY = 0.25;


struct FrontTask {
    Mask d0;
    Mask d1;
    double prob;
};


struct RouteCandidate {
    Mask d0;
    Mask d1;
    double prob = 0.0;
    double pre_score_log2 = -INFINITY;
    int total_weight = 0;
};

// priority_queue top() is the currently worst retained candidate.
struct RouteCandidateMinCompare {
    bool operator()(const RouteCandidate& a, const RouteCandidate& b) const noexcept {
        if (a.pre_score_log2 != b.pre_score_log2)
            return a.pre_score_log2 > b.pre_score_log2;
        if (a.prob != b.prob) return a.prob > b.prob;
        return a.total_weight < b.total_weight;
    }
};

using RouteHeap = std::priority_queue<
    RouteCandidate,
    std::vector<RouteCandidate>,
    RouteCandidateMinCompare
>;

static inline bool route_candidate_better(
    const RouteCandidate& a,
    const RouteCandidate& b
) noexcept {
    if (a.pre_score_log2 != b.pre_score_log2)
        return a.pre_score_log2 > b.pre_score_log2;
    if (a.prob != b.prob) return a.prob > b.prob;
    return a.total_weight < b.total_weight;
}

static inline void keep_route_candidate(RouteHeap& heap, RouteCandidate candidate) {
    if (heap.size() < ROUTE_PREFILTER_TOP_K) {
        heap.push(std::move(candidate));
        return;
    }
    if (route_candidate_better(candidate, heap.top())) {
        heap.pop();
        heap.push(std::move(candidate));
    }
}

static inline unsigned long long saturating_pow2_count(int exponent) noexcept {
    if (exponent < 0) return 0;
    if (exponent >= 64) return std::numeric_limits<unsigned long long>::max();
    return 1ULL << exponent;
}

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


static void collect_large_front_routes(
    int depth,
    const Mask& d0,
    const Mask& d1,
    double prob,
    int front_rounds,
    std::unordered_map<Mask, std::vector<std::pair<Mask,double>>, MaskHash>& chi_cache,
    std::unordered_map<Mask, Mask, MaskHash>& l_cache,
    std::unordered_map<Mask, int, MaskHash>& rank_cache,
    std::atomic<unsigned long long>& expanded_nodes,
    std::atomic<unsigned long long>& pruned_conflict,
    std::atomic<unsigned long long>& pruned_prob,
    RouteHeap& heap
) {
    if (depth == front_rounds) {
        const int weight = popcount_mask(d0) + popcount_mask(d1);
        const double score = log2_abs_safe(prob)
                           - ROUTE_WEIGHT_PENALTY * static_cast<double>(weight);
        keep_route_candidate(heap, RouteCandidate{d0, d1, prob, score, weight});
        return;
    }

    // Every output of chi_diff_distribution(delta) has probability 2^-rank.
    // Therefore every child pair below this node has the same probability.
    // Check that probability before constructing either output vector.
    const int rank0 = chi_rank_cached(d0, rank_cache);
    const int rank1 = chi_rank_cached(d1, rank_cache);
    const double child_prob = std::ldexp(prob, -(rank0 + rank1));
    const unsigned long long child_count = saturating_pow2_count(rank0 + rank1);

    if (!prob_survives(child_prob)) {
        pruned_prob.fetch_add(child_count, std::memory_order_relaxed);
        return;
    }

    const auto& chi0 = chi_cached_ref(d0, chi_cache);
    const auto& chi1 = chi_cached_ref(d1, chi_cache);

    for (const auto& c0 : chi0) {
        for (const auto& c1 : chi1) {
            expanded_nodes.fetch_add(1, std::memory_order_relaxed);

            if (FRONT_REQUIRE_DISJOINT_EACH_ROUND &&
                first_conflict_between(c0.first, c1.first)) {
                pruned_conflict.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            const Mask L0 = L_cached(c0.first, l_cache);
            const Mask L1 = L_cached(c1.first, l_cache);

            collect_large_front_routes(
                depth + 1,
                L0,
                L1,
                child_prob,
                front_rounds,
                chi_cache,
                l_cache,
                rank_cache,
                expanded_nodes,
                pruned_conflict,
                pruned_prob,
                heap
            );
        }
    }
}

struct ProxyScoredTask {
    FrontTask task;
    double proxy_abs_contribution = 0.0;
    int proxy_best_position = 0;
    double pre_score_log2 = -INFINITY;
    int total_weight = 0;
};

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

#if KOALA_SECOND_ORDER_VERIFY_FAST
        run_fast_chi_self_tests();
#endif

#if KOALA_SECOND_ORDER_SELF_TEST_ONLY
        std::cout << "self-test-only mode finished\n";
        return 0;
#endif

        const Mask d0 = parse_support(FIXED_INPUT_D0);
        const Mask d1 = parse_support(FIXED_INPUT_D1);
        const int tail_rounds = FIXED_ROUNDS - FIXED_FRONT_ROUNDS;
        if (tail_rounds < 0) {
            throw std::runtime_error("FIXED_ROUNDS must be >= FIXED_FRONT_ROUNDS");
        }

        std::cout << "Koala-p second-order large-route hybrid DL estimation\n";
        std::cout << "  input d0                 : " << fmt_support(d0) << "\n";
        std::cout << "  input d1                 : " << fmt_support(d1) << "\n";
        std::cout << "  total chi layers         : " << FIXED_ROUNDS << "\n";
        std::cout << "  exact front chi/L steps  : " << FIXED_FRONT_ROUNDS << "\n";
        std::cout << "  geometric tail layers    : " << tail_rounds << "\n";
        std::cout << "  workers                  : " << FIXED_THREADS << "\n";
        std::cout << "  final HDL coordinate     : 7\n";
        std::cout << "  enlarged block size      : " << ENLARGED_BLOCK_SIZE << "\n";
        std::cout << "  chi implementation       : genuine block-wise enlarged unit\n";
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

        // Stage 1: traverse the remaining exact-front steps, but retain only a
        // bounded beam of promising leaves in each worker.  The rank test in
        // collect_large_front_routes skips whole chi-product subtrees.
        std::vector<RouteHeap> local_heaps(FIXED_THREADS);

#pragma omp parallel
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            auto& heap = local_heaps[tid];

            std::unordered_map<
                Mask,
                std::vector<std::pair<Mask,double>>,
                MaskHash
            > chi_cache;
            std::unordered_map<Mask, Mask, MaskHash> l_cache;
            std::unordered_map<Mask, int, MaskHash> rank_cache;
            chi_cache.reserve(4096);
            l_cache.reserve(32768);
            rank_cache.reserve(4096);

#pragma omp for schedule(dynamic, 1)
            for (std::size_t task_index = 0;
                 task_index < tasks.size();
                 ++task_index) {
                collect_large_front_routes(
                    FIXED_FRONT_ROUNDS == 0 ? 0 : 1,
                    tasks[task_index].d0,
                    tasks[task_index].d1,
                    tasks[task_index].prob,
                    FIXED_FRONT_ROUNDS,
                    chi_cache,
                    l_cache,
                    rank_cache,
                    expanded_nodes,
                    pruned_conflict,
                    pruned_prob,
                    heap
                );
            }
        }

        std::vector<RouteCandidate> prefiltered;
        for (auto& heap : local_heaps) {
            while (!heap.empty()) {
                prefiltered.push_back(heap.top());
                heap.pop();
            }
        }
        std::sort(prefiltered.begin(), prefiltered.end(), route_candidate_better);
        if (prefiltered.size() > ROUTE_PREFILTER_TOP_K)
            prefiltered.resize(ROUTE_PREFILTER_TOP_K);

        // Merge identical endpoints among the retained routes.  The original
        // streaming DFS evaluated them repeatedly.
        PairDist merged_endpoints;
        merged_endpoints.reserve(prefiltered.size() * 2 + 16);
        std::unordered_map<PairMask, std::pair<double,int>, PairMaskHash> best_meta;
        best_meta.reserve(prefiltered.size() * 2 + 16);
        for (const RouteCandidate& candidate : prefiltered) {
            PairMask key{candidate.d0, candidate.d1};
            merged_endpoints[key] += candidate.prob;
            auto it = best_meta.find(key);
            if (it == best_meta.end() || candidate.pre_score_log2 > it->second.first)
                best_meta[key] = {candidate.pre_score_log2, candidate.total_weight};
        }

        std::vector<ProxyScoredTask> scored;
        scored.reserve(merged_endpoints.size());
        for (const auto& kv : merged_endpoints) {
            const auto meta = best_meta.find(kv.first)->second;
            scored.push_back({
                FrontTask{kv.first.d0, kv.first.d1, kv.second},
                0.0,
                0,
                meta.first,
                meta.second
            });
        }

        const int proxy_rounds = std::min(ROUTE_PROXY_ROUNDS, tail_rounds);

#pragma omp parallel for schedule(dynamic, 1)
        for (std::size_t i = 0; i < scored.size(); ++i) {
            const CorrVec proxy_corr = tail_correlations_second_order(
                scored[i].task.d0,
                scored[i].task.d1,
                proxy_rounds
            );
            const auto best = max_abs_position(proxy_corr);
            scored[i].proxy_best_position = best.first;
            scored[i].proxy_abs_contribution =
                scored[i].task.prob * std::fabs(best.second);
        }

        std::sort(scored.begin(), scored.end(), [](const ProxyScoredTask& a,
                                                   const ProxyScoredTask& b) {
            if (a.proxy_abs_contribution != b.proxy_abs_contribution)
                return a.proxy_abs_contribution > b.proxy_abs_contribution;
            if (a.task.prob != b.task.prob) return a.task.prob > b.task.prob;
            return a.total_weight < b.total_weight;
        });

        std::vector<FrontTask> selected_tasks;
        if (!scored.empty()) {
            const double best_proxy = scored.front().proxy_abs_contribution;
            const double relative_cutoff = best_proxy > 0.0
                ? std::ldexp(best_proxy, -static_cast<int>(ROUTE_KEEP_WITHIN_BITS))
                : 0.0;

            for (const ProxyScoredTask& candidate : scored) {
                if (selected_tasks.size() >= ROUTE_FINAL_TOP_K) break;
                if (best_proxy > 0.0 &&
                    candidate.proxy_abs_contribution < relative_cutoff) break;
                selected_tasks.push_back(candidate.task);
            }
        }

        double prefiltered_mass = 0.0;
        for (const auto& c : prefiltered) prefiltered_mass += c.prob;
        double merged_mass = 0.0;
        for (const auto& kv : merged_endpoints) merged_mass += kv.second;
        double selected_mass = 0.0;
        for (const auto& task : selected_tasks) selected_mass += task.prob;

        std::cout << "\nroute selection\n";
        std::cout << "  probability cutoff       : 2^(" << FRONT_LOG2_CUTOFF << ")\n";
        std::cout << "  structural prefilter K   : " << ROUTE_PREFILTER_TOP_K << "\n";
        std::cout << "  prefiltered routes       : " << prefiltered.size() << "\n";
        std::cout << "  unique retained endpoints: " << merged_endpoints.size() << "\n";
        std::cout << "  proxy rounds             : " << proxy_rounds << "\n";
        std::cout << "  final route K            : " << ROUTE_FINAL_TOP_K << "\n";
        std::cout << "  selected full tails      : " << selected_tasks.size() << "\n";
        std::cout << "  prefiltered route mass   : " << std::setprecision(17) << prefiltered_mass << "\n";
        std::cout << "  merged endpoint mass     : " << merged_mass << "\n";
        std::cout << "  selected endpoint mass   : " << selected_mass << "\n";

        std::cout << "\nTop proxy-scored routes\n";
        std::cout << "rank  p_log2   weight  proxy_pos  proxy_contribution_log2\n";
        for (std::size_t i = 0; i < std::min<std::size_t>(20, scored.size()); ++i) {
            std::cout << std::setw(4) << (i + 1)
                      << "  " << std::setw(7) << std::fixed << std::setprecision(2)
                      << log2_abs_safe(scored[i].task.prob)
                      << "  " << std::setw(6) << scored[i].total_weight
                      << "  " << std::setw(9) << scored[i].proxy_best_position
                      << "  " << std::setw(23) << std::setprecision(9)
                      << log2_abs_safe(scored[i].proxy_abs_contribution)
                      << "\n";
        }
        std::cout << std::defaultfloat << std::flush;

        std::ofstream route_out(FIXED_ROUTE_OUTPUT);
        route_out << std::setprecision(17);
        route_out << "rank prob log2_prob total_weight proxy_best_position "
                     "proxy_abs_contribution log2_proxy_abs d0 d1\n";
        for (std::size_t i = 0; i < scored.size() && i < selected_tasks.size(); ++i) {
            route_out << (i + 1) << " "
                      << scored[i].task.prob << " "
                      << log2_abs_safe(scored[i].task.prob) << " "
                      << scored[i].total_weight << " "
                      << scored[i].proxy_best_position << " "
                      << scored[i].proxy_abs_contribution << " "
                      << log2_abs_safe(scored[i].proxy_abs_contribution) << " "
                      << fmt_support(scored[i].task.d0, N) << " "
                      << fmt_support(scored[i].task.d1, N) << "\n";
        }

        std::mutex monitor_mutex;
        std::condition_variable monitor_cv;

        std::thread monitor_thread([&]() {
            std::unique_lock<std::mutex> lock(monitor_mutex);
            while (!monitor_stop.load(std::memory_order_relaxed)) {
                const bool stopped = monitor_cv.wait_for(
                    lock,
                    std::chrono::milliseconds(
                        static_cast<int>(FIXED_PROGRESS_EVERY * 1000.0)
                    ),
                    [&]() { return monitor_stop.load(std::memory_order_relaxed); }
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
                    std::chrono::steady_clock::now() - start_time
                ).count();
#pragma omp critical(streaming_print)
                {
                    std::cout << "  progress elapsed=" << std::fixed
                              << std::setprecision(3) << elapsed << "s"
                              << " selected_done=" << snapshot_count
                              << "/" << selected_tasks.size()
                              << " probability_mass=" << std::scientific
                              << std::setprecision(9) << snapshot_mass
                              << " current_best_pos=" << std::defaultfloat
                              << best.first
                              << " current_best_log2abs="
                              << std::setprecision(9) << log2_abs_safe(best.second)
                              << "\n" << std::flush;
                }
                lock.lock();
            }
        });

#pragma omp parallel
        {
            CorrVec local_total{};
            CorrVec local_abs{};
            double local_mass = 0.0;
            unsigned long long local_count = 0;

#pragma omp for schedule(dynamic, 1)
            for (std::size_t task_index = 0;
                 task_index < selected_tasks.size();
                 ++task_index) {
                const FrontTask& task = selected_tasks[task_index];
                const CorrVec corr = tail_correlations_second_order(
                    task.d0,
                    task.d1,
                    tail_rounds
                );
                for (int i = 0; i < N; ++i) {
                    const double contribution = task.prob * corr[i];
                    local_total[i] += contribution;
                    local_abs[i] += std::fabs(contribution);
                }
                local_mass += task.prob;
                ++local_count;
                tail_evals.fetch_add(1, std::memory_order_relaxed);

                if (local_count >= LOCAL_FLUSH_TAILS) {
                    flush_local_to_global(
                        local_total, local_abs, local_mass, local_count,
                        global_total, global_abs, global_mass, global_count
                    );
                }
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
        json << "  \"cipher\": \"Koala-p\",\n";
        json << "  \"order\": 2,\n";
        json << "  \"chi_implementation\": \"fast Walsh/rank-one\",\n";
        json << "  \"rounds\": " << FIXED_ROUNDS << ",\n";
        json << "  \"front_rounds\": " << FIXED_FRONT_ROUNDS << ",\n";
        json << "  \"route_probability_log2_cutoff\": " << FRONT_LOG2_CUTOFF << ",\n";
        json << "  \"route_prefilter_top_k\": " << ROUTE_PREFILTER_TOP_K << ",\n";
        json << "  \"route_proxy_rounds\": " << ROUTE_PROXY_ROUNDS << ",\n";
        json << "  \"route_final_top_k\": " << ROUTE_FINAL_TOP_K << ",\n";
        json << "  \"tail_rounds\": " << tail_rounds << ",\n";
        json << "  \"enlarged_block_size\": " << ENLARGED_BLOCK_SIZE << ",\n";
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
        std::cout << "  wrote selected routes    : "
                  << FIXED_ROUTE_OUTPUT << "\n";

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

