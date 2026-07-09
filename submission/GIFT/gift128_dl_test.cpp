#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <random>
using State = std::array<std::uint8_t, 128>;

// GIFT S-box, Table 3 of the GIFT specification.
static constexpr std::array<std::uint8_t, 16> SBOX = {
    0x1, 0xA, 0x4, 0xC,
    0x6, 0xF, 0x3, 0x9,
    0x2, 0xD, 0xB, 0x7,
    0x5, 0x0, 0x8, 0xE
};

// GIFT-128 forward bit permutation P128:
//     output[P128[i]] = input[i].
// Bit numbering follows the paper: b0 is the least-significant bit.
static constexpr std::array<int, 128> P128 = {
      0,  33,  66,  99,  96,   1,  34,  67,
     64,  97,   2,  35,  32,  65,  98,   3,
      4,  37,  70, 103, 100,   5,  38,  71,
     68, 101,   6,  39,  36,  69, 102,   7,
      8,  41,  74, 107, 104,   9,  42,  75,
     72, 105,  10,  43,  40,  73, 106,  11,
     12,  45,  78, 111, 108,  13,  46,  79,
     76, 109,  14,  47,  44,  77, 110,  15,
     16,  49,  82, 115, 112,  17,  50,  83,
     80, 113,  18,  51,  48,  81, 114,  19,
     20,  53,  86, 119, 116,  21,  54,  87,
     84, 117,  22,  55,  52,  85, 118,  23,
     24,  57,  90, 123, 120,  25,  58,  91,
     88, 121,  26,  59,  56,  89, 122,  27,
     28,  61,  94, 127, 124,  29,  62,  95,
     92, 125,  30,  63,  60,  93, 126,  31
};

// SplitMix64: fast deterministic generation of independent-looking random words.
static inline std::uint64_t splitmix64(std::uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static void random_state(State &x, std::uint64_t sample)
{
    const std::uint64_t low  = splitmix64(2 * sample);
    const std::uint64_t high = splitmix64(2 * sample + 1);

    for (int i = 0; i < 64; ++i) {
        x[i]      = static_cast<std::uint8_t>((low  >> i) & 1ULL);
        x[64 + i] = static_cast<std::uint8_t>((high >> i) & 1ULL);
    }
}

static void sbox_pass(State &x)
{
    for (int cell = 0; cell < 32; ++cell) {
        std::uint8_t value = 0;

        // The paper defines b0 as the least-significant bit. Therefore the
        // nibble w_cell has value sum_j b_(4*cell+j) * 2^j.
        for (int j = 0; j < 4; ++j) {
            value |= static_cast<std::uint8_t>(x[4 * cell + j] << j);
        }

        value = SBOX[value];

        for (int j = 0; j < 4; ++j) {
            x[4 * cell + j] = static_cast<std::uint8_t>((value >> j) & 1U);
        }
    }
}

static void perm_bits(State &x)
{
    State y{};
    for (int i = 0; i < 128; ++i) {
        y[P128[i]] = x[i];
    }
    x = y;
}

// This is the same round boundary as the uploaded PRESENT test program:
// every round applies an S-box layer, while the last tested round does not
// apply the following permutation.
//
// No round key or round constant is added here. This tests the keyless/zero-key
// S-box-permutation core used by the geometric correlation evaluator.
static void gift128_core(State &x, int rounds)
{
    for (int r = 0; r < rounds; ++r) {
        sbox_pass(x);
        if (r + 1 < rounds) {
            perm_bits(x);
        }
    }
}

// Convert a list (S-box position, nibble value) to a 128-bit vector.
// Example: {{0, 0x1}} means difference/mask 0x1 in nibble 0, i.e. bit b0.
static State make_nibble_vector(
    const std::vector<std::pair<int, std::uint8_t>> &active_nibbles)
{
    State bits{};

    for (const auto &[cell, value] : active_nibbles) {
        if (cell < 0 || cell >= 32) {
            throw std::invalid_argument("S-box position must be in 0..31");
        }
        if (value >= 16) {
            throw std::invalid_argument("nibble value must be in 0x0..0xF");
        }

        for (int j = 0; j < 4; ++j) {
            bits[4 * cell + j] ^= static_cast<std::uint8_t>((value >> j) & 1U);
        }
    }

    return bits;
}

static inline void xor_state(State &x, const State &difference)
{
    for (int i = 0; i < 128; ++i) {
        x[i] ^= difference[i];
    }
}

// Return <mask, y xor y'> over GF(2).
static inline int output_parity(
    const State &y,
    const State &y_prime,
    const State &mask)
{
    int parity = 0;
    for (int i = 0; i < 128; ++i) {
        parity ^= static_cast<int>(mask[i] & (y[i] ^ y_prime[i]));
    }
    return parity;
}

int main()
{
    // =====================================================================
    // Directly modify these four parameters, as in the PRESENT test code.
    // =====================================================================
    constexpr int ROUND = 7;
    constexpr int NUM = 26;  // number of plaintext pairs = 2^NUM
    const std::uint64_t seed =
        (static_cast<std::uint64_t>(std::random_device{}()) << 32) ^
        static_cast<std::uint64_t>(std::random_device{}());
    // Format: {S-box position, 4-bit difference/mask value}.
    // S-box positions are 0..31. Nibble values are 0x0..0xF.
    const std::vector<std::pair<int, std::uint8_t>> INPUT_DIFF = {
        {24, 0x8}
    };

    const std::vector<std::pair<int, std::uint8_t>> OUTPUT_MASK = {
        {30, 0x2}
    };

    const State difference = make_nibble_vector(INPUT_DIFF);
    const State mask = make_nibble_vector(OUTPUT_MASK);

    const std::uint64_t samples = 1ULL << NUM;
    long long signed_sum = 0;

    // Compile with -fopenmp to use all CPU cores. Without -fopenmp, the pragma
    // is ignored and the same code runs sequentially.
    #pragma omp parallel for reduction(+:signed_sum) schedule(static)
    for (std::uint64_t test = 0; test < samples; ++test) {
        State x{};
        State x_prime{};

         random_state(x, test ^ seed);
        x_prime = x;
        xor_state(x_prime, difference);

        gift128_core(x, ROUND);
        gift128_core(x_prime, ROUND);

        signed_sum += (output_parity(x, x_prime, mask) == 0) ? 1 : -1;
    }

    const double correlation =
        static_cast<double>(signed_sum) / static_cast<double>(samples);
    const double standard_error = std::sqrt(
        std::max(0.0, 1.0 - correlation * correlation) /
        static_cast<double>(samples));

    std::cout << std::setprecision(17);
    std::cout << "rounds          : " << ROUND << '\n';
    std::cout << "samples         : 2^" << NUM << " = " << samples << '\n';
    std::cout << "signed sum      : " << signed_sum << '\n';
    std::cout << "correlation     : " << correlation << '\n';
    std::cout << "standard error  : " << standard_error << '\n';

    if (correlation == 0.0) {
        std::cout << "log2(abs(cor))  : -inf\n";
    } else {
        std::cout << "log2(abs(cor))  : "
                  << std::log2(std::abs(correlation)) << '\n';
    }

    return 0;
}
