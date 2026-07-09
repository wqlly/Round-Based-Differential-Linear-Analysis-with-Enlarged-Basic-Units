#include <iostream>
#include <array>
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <omp.h>

using namespace std;

constexpr int N = 257;

// ===============================
// 参数设置
// ===============================
constexpr int ROUND = 5;
constexpr uint64_t TEST_NUM = (1ULL << 20);

constexpr int DIFF0_BIT = 0;
constexpr int DIFF1_BIT = 32;
constexpr int DIFF2_BIT = 254;
// ===============================

using State = array<uint8_t, N>;
using CounterArray = array<long long, N>;

// 每个线程使用独立的随机数生成器
inline uint64_t rand64()
{
    static thread_local mt19937_64 gen(
        123456789ULL + static_cast<uint64_t>(omp_get_thread_num())
    );
    return gen();
}

void random_state(State& s)
{
    for (int i = 0; i < N;)
    {
        const uint64_t r = rand64();

        for (int j = 0; j < 64 && i < N; ++j, ++i)
            s[i] = static_cast<uint8_t>((r >> j) & 1ULL);
    }
}

void Chi(const State& in, State& out)
{
    for (int i = 0; i < N; ++i)
    {
        out[i] = static_cast<uint8_t>(
            in[i] ^ ((in[(i + 1) % N] ^ 1U) & in[(i + 2) % N])
        );
    }
}

void L1(const State& in, State& out)
{
    State tmp{};

    for (int i = 0; i < N; ++i)
    {
        tmp[i] = static_cast<uint8_t>(
            in[i] ^ in[(i + 3) % N] ^ in[(i + 8) % N]
        );
    }

    for (int i = 0; i < N; ++i)
        out[i] = tmp[(12 * i) % N];
}

void one_tail_round(State& s, bool do_linear)
{
    State t{}, u{};

    Chi(s, t);

    if (do_linear)
    {
        L1(t, u);
        s = u;
    }
    else
    {
        s = t;
    }
}

void tail_function(State& s)
{
    for (int r = 0; r < ROUND; ++r)
        one_tail_round(s, r < ROUND - 1);
}

/*
 * 固定输入三阶差分：
 *   DIFF0_BIT = 0
 *   DIFF1_BIT = 32
 *   DIFF2_BIT = 2
 *
 * 一次实验同时统计 LOCATION = 0,...,256。
 * 这样只需计算一次 8 个状态，不需要为每个 LOCATION
 * 重新运行 TEST_NUM 次实验。
 */
CounterArray compute_all_correlations()
{
    const int thread_num = omp_get_max_threads();

    // 每个线程独立维护 257 个计数器，避免原子操作
    vector<CounterArray> local_counters(thread_num);
    for (auto& counters : local_counters)
        counters.fill(0);

#pragma omp parallel default(none) shared(local_counters)
    {
        const int tid = omp_get_thread_num();
        CounterArray& counter = local_counters[tid];

#pragma omp for schedule(static)
        for (uint64_t test = 0; test < TEST_NUM; ++test)
        {
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
            x001[DIFF2_BIT] ^= 1U;

            x110[DIFF0_BIT] ^= 1U;
            x110[DIFF1_BIT] ^= 1U;

            x101[DIFF0_BIT] ^= 1U;
            x101[DIFF2_BIT] ^= 1U;

            x011[DIFF1_BIT] ^= 1U;
            x011[DIFF2_BIT] ^= 1U;

            x111[DIFF0_BIT] ^= 1U;
            x111[DIFF1_BIT] ^= 1U;
            x111[DIFF2_BIT] ^= 1U;

            tail_function(x000);
            tail_function(x100);
            tail_function(x010);
            tail_function(x001);
            tail_function(x110);
            tail_function(x101);
            tail_function(x011);
            tail_function(x111);

            // 一次性统计全部 257 个单比特输出掩码
            for (int location = 0; location < N; ++location)
            {
                const uint8_t d =
                    x000[location] ^
                    x100[location] ^
                    x010[location] ^
                    x001[location] ^
                    x110[location] ^
                    x101[location] ^
                    x011[location] ^
                    x111[location];

                counter[location] += (d == 0) ? 1LL : -1LL;
            }
        }
    }

    // 合并所有线程的局部计数器
    CounterArray total_counter{};
    total_counter.fill(0);

    for (int tid = 0; tid < thread_num; ++tid)
    {
        for (int location = 0; location < N; ++location)
            total_counter[location] += local_counters[tid][location];
    }

    return total_counter;
}

int main()
{
    cout << scientific << setprecision(8);

    omp_set_max_active_levels(1);
    omp_set_dynamic(0);

    cout << "============================================================\n";
    cout << "Third-order fixed input differences, scan LOCATION 0~256\n";
    cout << "ROUND       = " << ROUND << '\n';
    cout << "DIFF bits   = (" << DIFF0_BIT << ", "
         << DIFF1_BIT << ", " << DIFF2_BIT << ")\n";
    cout << "TEST_NUM    = " << TEST_NUM << '\n';
    cout << "Threads     = " << omp_get_max_threads() << '\n';
    cout << "============================================================\n";
    cout << "location\tcorrelation\t\tlog2(|correlation|)\n";
    cout << "============================================================\n";

    const CounterArray counters = compute_all_correlations();

    int best_location = -1;
    double best_abs_corr = -1.0;
    double best_corr = 0.0;

    for (int location = 0; location < N; ++location)
    {
        const double corr =
            static_cast<double>(counters[location]) /
            static_cast<double>(TEST_NUM);

        const double abs_corr = fabs(corr);
        const double logv =
            (abs_corr > 0.0) ? log2(abs_corr) : -INFINITY;

        cout << location << '\t'
             << corr << "\t"
             << logv << '\n';

        if (abs_corr > best_abs_corr)
        {
            best_abs_corr = abs_corr;
            best_corr = corr;
            best_location = location;
        }
    }

    const double best_log =
        (best_abs_corr > 0.0) ? log2(best_abs_corr) : -INFINITY;

    cout << "============================================================\n";
    cout << "Maximum absolute correlation:\n";
    cout << "LOCATION = " << best_location
         << ", corr = " << best_corr
         << ", log2(|corr|) = " << best_log << '\n';
    cout << "============================================================\n";

    return 0;
}
