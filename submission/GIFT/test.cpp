#include<iostream>
#include<random>
#include<cmath>
#include<thread>
#include<vector>
#include<cstdint>
#include<cstring>
#include<iomanip>
#include<mutex>
using namespace std;

// 全局常量定义
const int NUM_KEYS = 100;                // 测试密钥总数
const long long NUM_TESTS_PER_KEY = 1LL << 30; // 每个密钥的测试次数（降低到2^20，避免耗时过久，可调整）
const int ROUNDS = 8;                    // 加密轮数
const int THREAD_COUNT = 8;              // 密钥维度的线程数

// GIFT-64 S盒
int sbox[] = { 0x1, 0xa, 0x4, 0xc, 0x6, 0xf, 0x3, 0x9, 0x2, 0xd, 0xb, 0x7, 0x5, 0x0, 0x8, 0xe };
// GIFT-64 置换表（位序：p[i] 表示第i位置换到p[i]位，i∈[0,63]，0=最低位，63=最高位）
int p[] = { 0, 17, 34, 51, 48, 1, 18, 35, 32, 49, 2, 19, 16, 33, 50, 3,
            4, 21, 38, 55, 52, 5, 22, 39, 36, 53, 6, 23, 20, 37, 54, 7,
            8, 25, 42, 59, 56, 9, 26, 43, 40, 57, 10, 27, 24, 41, 58, 11,
            12, 29, 46, 63, 60, 13, 30, 47, 44, 61, 14, 31, 28, 45, 62, 15 };

// 轮常数（文档给出的1-48轮具体值，6位：c5(最高位)~c0(最低位)）
const uint8_t RC[] = {
    0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E,
    0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E, 0x1C, 0x38,
    0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A, 0x34, 0x29, 0x12, 0x24, 0x08, 0x11, 0x22, 0x04
};

// 存储单个密钥的两个log_bias结果（61位和63位差分）
struct KeyBiasResult {
    double log_bias_61;  // x_enc[61] ^ x1_enc[61] 的log2(bias)
    double log_bias_63;  // x_enc[63] ^ x1_enc[63] 的log2(bias)
    uint64_t key;        // 对应的密钥（可选，用于调试）
};

// 全局结果数组（存储所有密钥的结果）
vector<KeyBiasResult> g_key_results(NUM_KEYS);
// 线程安全锁（可选，用于调试输出）
mutex g_print_mutex;

// ===================== 核心修改：S盒变换（匹配位序：X[0]=最低位，X[63]=最高位）=====================
void sboxpass(int x[])
{
    for (int i = 0; i < 16; i++)
    {
        int nibble = 0;
        nibble |= x[4*i + 0] << 0;  // b0（最低位）
        nibble |= x[4*i + 1] << 1;  // b1
        nibble |= x[4*i + 2] << 2;  // b2
        nibble |= x[4*i + 3] << 3;  // b3（最高位）

        nibble = sbox[nibble];

        x[4*i + 0] = (nibble >> 0) & 0x1;
        x[4*i + 1] = (nibble >> 1) & 0x1;
        x[4*i + 2] = (nibble >> 2) & 0x1;
        x[4*i + 3] = (nibble >> 3) & 0x1;
    }
}

// ===================== 位置换（匹配位序：X[0]=最低位，X[63]=最高位）=====================
void perm(int x[])
{
    int y[64] = { 0 };
    for (int i = 0; i < 64; i++) {
        y[p[i]] = x[i];
    }
    for (int i = 0; i < 64; i++) {
        x[i] = y[i];
    }
}

// ===================== GIFT-64 密钥调度（位序匹配）=====================
void key_schedule(uint64_t key, uint32_t rk[48], uint8_t rc[48]) {
    uint16_t k3 = (uint16_t)(key >> 48);
    uint16_t k2 = (uint16_t)(key >> 32);
    uint16_t k1 = (uint16_t)(key >> 16);
    uint16_t k0 = (uint16_t)key;

    for (int i = 0; i < 48; ++i) {
        rk[i] = (uint32_t)k1 << 16 | k0;
        rc[i] = RC[i];

        auto rot_right_2 = [](uint16_t val) {
            return (val >> 2) | ((val & 0b11) << 14);
        };
        auto rot_right_12 = [](uint16_t val) {
            return (val >> 12) | ((val & 0xFFF) << 4);
        };

        uint16_t new_k3 = rot_right_2(k1);
        uint16_t new_k2 = rot_right_12(k0);
        uint16_t new_k1 = rot_right_2(k3);
        uint16_t new_k0 = rot_right_12(k2);

        k3 = new_k3;
        k2 = new_k2;
        k1 = new_k1;
        k0 = new_k0;
    }
}

// ===================== 轮密钥异或（匹配位序：X[0]=最低位，X[63]=最高位）=====================
void add_round_key(int state[64], uint32_t rk, uint8_t rc) {
    uint16_t U = (uint16_t)(rk >> 16);
    uint16_t V = (uint16_t)rk;

    for (int i = 0; i < 16; ++i) {
        int bit_pos = 4 * i + 1;
        state[bit_pos] ^= (U >> i) & 1;
    }
    for (int i = 0; i < 16; ++i) {
        int bit_pos = 4 * i;
        state[bit_pos] ^= (V >> i) & 1;
    }
    state[63] ^= 1;
    state[23] ^= (rc >> 5) & 1;
    state[19] ^= (rc >> 4) & 1;
    state[15] ^= (rc >> 3) & 1;
    state[11] ^= (rc >> 2) & 1;
    state[7]  ^= (rc >> 1) & 1;
    state[3]  ^= rc & 1;
}

// GIFT-64 加密主函数（位序：state[0]=最低位，state[63]=最高位）
void gift_encryption(int state[64],int rounds) {
    //uint32_t rk[48];
    //uint8_t rc[48];
    //key_schedule(key, rk, rc);

    for (int i = 0; i < rounds; ++i) {
        //add_round_key(state, rk[i], rc[i]);
        sboxpass(state);
        if (i < rounds - 1) perm(state);
    }
}

// 线程函数：处理指定范围的key_idx（单个key_idx内部单线程跑所有测试）
void thread_func(int start_key_idx, int end_key_idx) {
    // 每个线程独立的随机数生成器（避免线程竞争）
    random_device rd;
    //mt19937_64 gen_key(rd());
    //uniform_int_distribution<uint64_t> key_dis(0, UINT64_MAX);
    mt19937 gen_plain(rd());
    uniform_int_distribution<int> plain_dis(0, 1);

    for (int key_idx = start_key_idx; key_idx < end_key_idx; ++key_idx) {
        // 1. 生成当前密钥
        //uint64_t cur_key = key_dis(gen_key);
        double cor_61 = 0.0;  // 61位差分的相关性累加
        double cor_63 = 0.0;  // 63位差分的相关性累加

        // 2. 单个密钥的所有测试（单线程）
        for (long long test = 0; test < NUM_TESTS_PER_KEY; ++test) {
            int x[64], x1[64];
            // 生成随机明文
            for (int i = 0; i < 64; i++) {
                x[i] = plain_dis(gen_plain);
                x1[i] = x[i];
            }
            x1[0] ^= 1;  // 明文差分：第0位翻转

            // 加密x和x1
            int x_enc[64], x1_enc[64];
            memcpy(x_enc, x, sizeof(x_enc));
            memcpy(x1_enc, x1, sizeof(x1_enc));
            gift_encryption(x_enc, ROUNDS);
            gift_encryption(x1_enc, ROUNDS);

            // 计算两个输出差分的相关性
            int out_diff_61 = x_enc[61] ^ x1_enc[61];
            int out_diff_63 = x_enc[63] ^ x1_enc[63];
            cor_61 += (out_diff_61 == 0) ? 1 : -1;
            cor_63 += (out_diff_63 == 0) ? 1 : -1;
        }

        // 3. 计算当前密钥的bias和log_bias
        double bias_61 = cor_61 / NUM_TESTS_PER_KEY;
        double bias_63 = cor_63 / NUM_TESTS_PER_KEY;
        double log_bias_61 = log2(abs(bias_61));
        double log_bias_63 = log2(abs(bias_63));

        // 4. 存储结果到全局数组
        g_key_results[key_idx].log_bias_61 = log_bias_61;
        g_key_results[key_idx].log_bias_63 = log_bias_63;
        //g_key_results[key_idx].key = cur_key;

        // 5. 线程安全的调试输出
        lock_guard<mutex> lock(g_print_mutex);
        cout << "Key " << (key_idx + 1) << " result: " << fixed << setprecision(10)
             << "bias_61 = " << bias_61 << ", log2(bias_61) = " << log_bias_61
             << " | bias_63 = " << bias_63 << ", log2(bias_63) = " << log_bias_63 << endl;
        cout << "----------------------------------------" << endl;
    }
}

int main() {
    cout << "开始测试 " << NUM_KEYS << " 个密钥，每个密钥测试 " << NUM_TESTS_PER_KEY << " 次..." << endl;
    cout << "线程数：" << THREAD_COUNT << "（按密钥维度并行）" << endl;
    cout << "加密轮数：" << ROUNDS << endl;
    cout << "========================================" << endl;

    // 1. 分配每个线程的密钥范围
    vector<thread> threads;
    int keys_per_thread = NUM_KEYS / THREAD_COUNT;
    int remaining_keys = NUM_KEYS % THREAD_COUNT;
    int start_idx = 0;

    for (int t = 0; t < THREAD_COUNT; ++t) {
        int end_idx = start_idx + keys_per_thread;
        if (t == THREAD_COUNT - 1) {
            end_idx += remaining_keys;  // 最后一个线程处理剩余密钥
        }
        threads.emplace_back(thread_func, start_idx, end_idx);
        start_idx = end_idx;
    }

    // 2. 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    // 3. 计算所有密钥的log_bias平均值
    double avg_log_bias_61 = 0.0;
    double avg_log_bias_63 = 0.0;
    for (int i = 0; i < NUM_KEYS; ++i) {
        avg_log_bias_61 += g_key_results[i].log_bias_61;
        avg_log_bias_63 += g_key_results[i].log_bias_63;
    }
    avg_log_bias_61 /= NUM_KEYS;
    avg_log_bias_63 /= NUM_KEYS;

    // 4. 输出最终平均值
    cout << "========================================" << endl;
    cout << "所有 " << NUM_KEYS << " 个密钥的log_bias平均值：" << endl;
    cout << fixed << setprecision(10);
    cout << "x_enc[61] ^ x1_enc[61] 的log2(bias)平均值：" << avg_log_bias_61 << endl;
    cout << "x_enc[63] ^ x1_enc[63] 的log2(bias)平均值：" << avg_log_bias_63 << endl;

    return 0;
}