// koala_p_verify3.cpp
//
// Koala-p third-order differential-linear Monte-Carlo verification.
// Converted from the supplied Subterranean-2.0 verify3.cpp.
//
// The tested tail contains ROUND chi layers and therefore has the form
//
//     chi -> (pi -> theta -> [iota]) -> chi -> ... -> chi.
//
// This is the same chi-to-chi boundary convention as the supplied
// Subterranean verification program.  It is not the convention ¡°ROUND full
// rounds starting before pi¡±.
//
// Koala-p:
//     pi    : y_i = x_{121 i}
//     theta : z_i = y_i xor y_{i+3} xor y_{i+10}
//     iota  : z_0 ^= 1 when j not in {2,5,6}
//     chi   : s_i = z_i xor ((z_{i+1} xor 1) & z_{i+2})
//
// By default INCLUDE_IOTA is false, matching the supplied Subterranean code
// and the constant-free geometric model. Set it to true to insert Koala-p
// constants between consecutive chi layers.
//
// Compile:
//   g++ -O3 -march=native -std=c++17 -fopenmp koala_p_verify3.cpp -o koala_verify3
//
// Run:
//   ./koala_verify3

#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

using State = std::array<uint8_t, 257>;

namespace {

constexpr int N = 257;

// ===============================
// Experiment parameters
// ===============================
constexpr int ROUND = 5;                  // number of chi layers
constexpr int LOCATION = 191;             // one-bit output mask
constexpr uint64_t TEST_NUM = (1ULL << 26); // 4,194,304 samples per diff2

constexpr int DIFF0_BIT = 0;
constexpr int DIFF1_BIT = 11;

// The supplied program evaluates only diff2=222.  Use [0,N) for a full scan.
constexpr int DIFF2_BEGIN = 64;
constexpr int DIFF2_END = 65;             // exclusive

// false: match the supplied constant-free geometric-tail verification.
// true : insert Koala-p iota constants between consecutive chi layers.
constexpr bool INCLUDE_IOTA = false;

// Zero-based round index of the first chi layer.  It matters only when
// INCLUDE_IOTA=true.  The transition after chi layer r uses the constant of
// the next Koala-p round, FIRST_CHI_ROUND_INDEX + r + 1.
constexpr int FIRST_CHI_ROUND_INDEX = 0;

constexpr uint64_t RNG_BASE_SEED = 123456789ULL;

static_assert(ROUND >= 1, "ROUND must be positive");
static_assert(0 <= LOCATION && LOCATION < N, "LOCATION outside [0,256]");
static_assert(0 <= DIFF0_BIT && DIFF0_BIT < N, "DIFF0_BIT outside [0,256]");
static_assert(0 <= DIFF1_BIT && DIFF1_BIT < N, "DIFF1_BIT outside [0,256]");
static_assert(0 <= DIFF2_BEGIN && DIFF2_BEGIN <= DIFF2_END && DIFF2_END <= N,
              "invalid diff2 range");

inline int thread_id() noexcept {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

// Each OpenMP thread owns an independent deterministic generator.
inline uint64_t rand64() {
    static thread_local std::mt19937_64 gen(
        RNG_BASE_SEED + 0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(thread_id() + 1)
    );
    return gen();
}

void random_state(State& s) {
    int i = 0;
    while (i < N) {
        const uint64_t r = rand64();
        for (int b = 0; b < 64 && i < N; ++b, ++i) {
            s[i] = static_cast<uint8_t>((r >> b) & 1ULL);
        }
    }
}

// chi: s_i <- s_i xor ((s_{i+1} xor 1) & s_{i+2})
void chi(const State& in, State& out) {
    for (int i = 0; i < N; ++i) {
        out[i] = static_cast<uint8_t>(
            in[i] ^ ((in[(i + 1) % N] ^ 1U) & in[(i + 2) % N])
        );
    }
}

// Koala-p linear layer L = theta o pi.
//
// pi    : p_i = in_{121 i}
// theta : out_i = p_i xor p_{i+3} xor p_{i+10}
void koala_linear(const State& in, State& out) {
    State p{};

    for (int i = 0; i < N; ++i) {
        p[i] = in[(121 * i) % N];
    }

    for (int i = 0; i < N; ++i) {
        out[i] = static_cast<uint8_t>(
            p[i] ^ p[(i + 3) % N] ^ p[(i + 10) % N]
        );
    }
}

// Koala-p iota_j: toggle bit 0 when j is not 2, 5, or 6.
void koala_iota(State& s, int round_index) {
    if (round_index != 2 && round_index != 5 && round_index != 6) {
        s[0] ^= 1U;
    }
}

// One chi layer, followed by the Koala-p transition to the next chi layer.
void one_tail_chi_layer(State& s, int chi_layer_index, bool has_next_chi) {
    State after_chi{};
    chi(s, after_chi);

    if (!has_next_chi) {
        s = after_chi;
        return;
    }

    State after_linear{};
    koala_linear(after_chi, after_linear);

    if constexpr (INCLUDE_IOTA) {
        const int next_round_index = FIRST_CHI_ROUND_INDEX + chi_layer_index + 1;
        koala_iota(after_linear, next_round_index);
    }

    s = after_linear;
}

void tail_function(State& s) {
    for (int r = 0; r < ROUND; ++r) {
        one_tail_chi_layer(s, r, r + 1 < ROUND);
    }
}

// Estimate
// E_x[(-1)^{D_{d0}D_{d1}D_{d2} F(x)[LOCATION]}].
double compute_correlation(int diff2_bit) {
    long long counter = 0;

#pragma omp parallel for reduction(+:counter) schedule(static)
    for (uint64_t test = 0; test < TEST_NUM; ++test) {
        State x000{}, x100{}, x010{}, x001{};
        State x110{}, x101{}, x011{}, x111{};

        random_state(x000);
        x100 = x000;
        x010 = x000;
        x001 = x000;
        x110 = x000;
        x101 = x000;
        x011 = x000;
        x111 = x000;

        x100[DIFF0_BIT] ^= 1U;
        x010[DIFF1_BIT] ^= 1U;
        x001[diff2_bit] ^= 1U;

        x110[DIFF0_BIT] ^= 1U;
        x110[DIFF1_BIT] ^= 1U;

        x101[DIFF0_BIT] ^= 1U;
        x101[diff2_bit] ^= 1U;

        x011[DIFF1_BIT] ^= 1U;
        x011[diff2_bit] ^= 1U;

        x111[DIFF0_BIT] ^= 1U;
        x111[DIFF1_BIT] ^= 1U;
        x111[diff2_bit] ^= 1U;

        tail_function(x000);
        tail_function(x100);
        tail_function(x010);
        tail_function(x001);
        tail_function(x110);
        tail_function(x101);
        tail_function(x011);
        tail_function(x111);

        const uint8_t derivative = static_cast<uint8_t>(
            x000[LOCATION] ^ x100[LOCATION] ^ x010[LOCATION] ^ x001[LOCATION] ^
            x110[LOCATION] ^ x101[LOCATION] ^ x011[LOCATION] ^ x111[LOCATION]
        );

        counter += (derivative == 0U) ? 1LL : -1LL;
    }

    return static_cast<double>(counter) / static_cast<double>(TEST_NUM);
}

} // namespace

int main() {
#ifdef _OPENMP
    omp_set_max_active_levels(1);
    omp_set_dynamic(0);
#endif

    std::cout << std::scientific << std::setprecision(8);
    std::cout << "============================================================\n";
    std::cout << "Koala-p third-order DL Monte-Carlo verification\n";
    std::cout << "chi-layer boundary: chi -> pi -> theta -> [iota] -> chi\n";
    std::cout << "ROUND = " << ROUND
              << ", LOCATION = " << LOCATION
              << ", TESTS = " << TEST_NUM << "\n";
    std::cout << "D0 = [" << DIFF0_BIT << "]"
              << ", D1 = [" << DIFF1_BIT << "]"
              << ", D2 range = [" << DIFF2_BEGIN << "," << DIFF2_END << ")\n";
    std::cout << "iota constants = " << (INCLUDE_IOTA ? "enabled" : "omitted") << "\n";
#ifdef _OPENMP
    std::cout << "Threads = " << omp_get_max_threads() << "\n";
#else
    std::cout << "Threads = 1 (OpenMP disabled)\n";
#endif
    std::cout << "============================================================\n";
    std::cout << "diff2\tcorrelation\t\tlog2(|corr|)\tstderr\tz-score\n";
    std::cout << "============================================================\n";

    for (int diff2 = DIFF2_BEGIN; diff2 < DIFF2_END; ++diff2) {
        const double corr = compute_correlation(diff2);
        const double abs_corr = std::fabs(corr);
        const double logv = (abs_corr > 0.0)
            ? std::log2(abs_corr)
            : -std::numeric_limits<double>::infinity();

        // For a +/-1 sample mean, Var(mean) = (1-corr^2)/TEST_NUM.
        const double variance = std::max(0.0, 1.0 - corr * corr);
        const double stderr = std::sqrt(variance / static_cast<double>(TEST_NUM));
        const double zscore = (stderr > 0.0) ? corr / stderr : 0.0;

        std::cout << diff2 << '\t'
                  << corr << '\t'
                  << logv << '\t'
                  << stderr << '\t'
                  << zscore << '\n';
    }

    std::cout << "============================================================\n";
    std::cout << "For a full D2 scan, set DIFF2_BEGIN=0 and DIFF2_END=N.\n";
    return 0;
}

