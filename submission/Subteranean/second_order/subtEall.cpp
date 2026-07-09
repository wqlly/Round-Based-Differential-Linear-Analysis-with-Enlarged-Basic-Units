//
#include<iostream>
#include<cstdlib>
#include<cmath>
#include<cstdint>
#include<random>
#include<vector>
#define ROL(x,t) ((x<<t)|x>>(32-t))
using namespace std;
//const uint32_t rc[12] = { 0x00000058,0x00000038,0x000003C0,0x000000D0,0x00000120,0x00000014,0x00000060,0x0000002C,0x00000380,0x000000F0,0x000001A0,0x00000012 };
//random number gen
uint32_t p = 0x7fffffff; //128比特
random_device rd;
mt19937 gen(rd());
uniform_int_distribution<unsigned int> dis(0, 0xffffffff);

vector<unsigned int> L1(vector<unsigned int> c)
{
    // vector<unsigned int> c_2(257);
    vector<unsigned int> c_3(257);
    // c_1[(size_1 / 2) - 3] = c[(size_1 / 2)] ^ c[(size_1 / 2) - 3];
   //  c_1[(size_1 / 2) - 2] = c[(size_2 / 2) - 2] ^ c[(size_1 / 2) - 1];
   //  c_1[(size_1 / 2)] = c[(size_1 / 2)] ^ c[(size_1 / 2) - 2];
   //  c_1[(size_1 / 2) - 1] = c[(size_1 / 2)] ^ c[(size_1 / 2) - 3] ^ c[(size_1 / 2) - 1];
    for (int u = 0; u < 257; u++)
    {
        c_3[u] = c[(u) % 257] ^ c[(u + 3) % 257] ^ c[(u + 8) % 257];
    }
    for (int u = 0; u < 257; u++)
    {
        c[u] = c_3[(12 * u) % 257];
    }

    return c;
}
vector<unsigned int> Chi(vector<unsigned int> c)
{
    // vector<unsigned int> c_2(257);
    vector<unsigned int> c_3(257);
    // c_1[(size_1 / 2) - 3] = c[(size_1 / 2)] ^ c[(size_1 / 2) - 3];
   //  c_1[(size_1 / 2) - 2] = c[(size_1 / 2) - 2] ^ c[(size_1 / 2) - 1];
   //  c_1[(size_1 / 2)] = c[(size_1 / 2)] ^ c[(size_1 / 2) - 2];
   //  c_1[(size_1 / 2) - 1] = c[(size_1 / 2)] ^ c[(size_1 / 2) - 3] ^ c[(size_1 / 2) - 1];
    for (int u = 0; u < 257; u++)
    {
        c_3[u] = c[(u) % 257] ^ ((c[(u + 1) % 257] ^ 1) & c[(u + 2) % 257]);
    }
    return c_3;
}
vector<unsigned int> round_f(vector<unsigned int> c, int r)
{
    vector<unsigned int> c_3(257);
    c_3 = Chi(L1(c));
    for (int i = 0; i < r - 1; i++)
    {
        c_3 = Chi(L1(c_3));
    }
    return c_3;
}
void D_L(vector<unsigned int> diff, int r, int location) {//compute the correlation for given diff and all-one-bit mask

    vector<unsigned int> state(257);
    vector<unsigned int> statep(257);
    uint32_t a;
    long counter = 0;
    // long TEST_NUM = 0x2;
    //long TEST_NUM = 0x1ffffff;
    long TEST_NUM = 0xfff;
    double cor = 0;
    for (int test = 0; test <= TEST_NUM; test++) {
        for (int i = 0; i < 8; i++)
        {
            a = (dis(gen));
            for (int j = 0; j < 32; j++)
            {
                state[i * 32 + j] = (a >> j) & 0x1;
                statep[i * 32 + j] = state[i * 32 + j] ^ diff[i * 32 + j];
            }
        }
        state[256] = (dis(gen)) & 0x1;
        statep[256] = state[256] ^ diff[256];
        state = Chi(state);
        statep = Chi(statep);
        state[0] = state[0] ^ 1;
        statep[0] = statep[0] ^ 1;
        state = L1(state);
        statep = L1(statep);

        
        state = Chi(state);
        statep = Chi(statep);
        state[0] = state[0] ^ 1;
        statep[0] = statep[0] ^ 1;
        state = L1(state);
        statep = L1(statep);

        state = Chi(state);
        statep = Chi(statep);
        state[0] = state[0] ^ 1;
        statep[0] = statep[0] ^ 1;
        state = L1(state);
        statep = L1(statep);

     //   state = Chi(state);
    //    statep = Chi(statep);
    //    state[0] = state[0] ^ 1;
    //    statep[0] = statep[0] ^ 1;
    //    state = L1(state);
    //    statep = L1(statep);

        state = Chi(state);
        statep = Chi(statep);
        //  state = Chi(state);
       //   statep = Chi(statep);
       //   state = L1(state);
        //  statep = L1(statep);
        bool linear_approx = 0;
        linear_approx = state[location] ^ statep[location];
        if (linear_approx == 0) { counter++; }
        else { counter--; }
    }
    //  cout << "Correlation after mindle part  " << " = 2 ^ ";
    //  cout << (log2(abs(counter)) - log2(TEST_NUM + 1)) << " ";
    cout << double(counter) / double(TEST_NUM + 1) << " " << endl;
}
void HD_L_2n_all(vector<unsigned int> diff, vector<unsigned int> diff_1) {//compute the correlation for given diff and all-one-bit mask

    vector<unsigned int> state(257);
    vector<unsigned int> statep(257);
    vector<unsigned int> statep_1(257);
    vector<unsigned int> statep_2(257);
    uint32_t a;
    long counter[257];
    for (int i = 0; i < 257; i++)
    {
        counter[i] = 0;
    }
    // long TEST_NUM = 0x2;
    long TEST_NUM = 0x3ffffff;
    // long TEST_NUM = 0xfff;
    double cor = 0;
    for (int test = 0; test <= TEST_NUM; test++) {
        for (int i = 0; i < 8; i++)
        {
            a = (dis(gen));
            for (int j = 0; j < 32; j++)
            {
                state[i * 32 + j] = (a >> j) & 0x1;
                statep[i * 32 + j] = state[i * 32 + j] ^ diff[i * 32 + j];
                statep_1[i * 32 + j] = state[i * 32 + j] ^ diff_1[i * 32 + j];
                statep_2[i * 32 + j] = state[i * 32 + j] ^ diff_1[i * 32 + j] ^ diff[i * 32 + j];

            }
        }
        state[256] = (dis(gen)) & 0x1;
        statep[256] = state[256] ^ diff[256];
        statep_1[256] = state[256] ^ diff_1[256];
        statep_2[256] = state[256] ^ diff_1[256] ^ diff[256];

        state = Chi(state);
        statep = Chi(statep);
        statep_1 = Chi(statep_1);
        statep_2 = Chi(statep_2);

        //state[0] = state[0] ^ 1;
        //statep[0] = statep[0] ^ 1;
        //statep_1[0] = statep_1[0] ^ 1;
        //statep_2[0] = statep_2[0] ^ 1;

        state = L1(state);
        statep = L1(statep);
        statep_1 = L1(statep_1);
        statep_2 = L1(statep_2);

       

        state = Chi(state);
        statep = Chi(statep);
        statep_1 = Chi(statep_1);
        statep_2 = Chi(statep_2);

        //state[0] = state[0] ^ 1;
        //statep[0] = statep[0] ^ 1;
        //statep_1[0] = statep_1[0] ^ 1;
        //statep_2[0] = statep_2[0] ^ 1;

        state = L1(state);
        statep = L1(statep);
        statep_1 = L1(statep_1);
        statep_2 = L1(statep_2);


        state = Chi(state);
        statep = Chi(statep);
        statep_1 = Chi(statep_1);
        statep_2 = Chi(statep_2);

        //state[0] = state[0] ^ 1;
        //statep[0] = statep[0] ^ 1;
        //statep_1[0] = statep_1[0] ^ 1;
        //statep_2[0] = statep_2[0] ^ 1;

        state = L1(state);
        statep = L1(statep);
        statep_1 = L1(statep_1);
        statep_2 = L1(statep_2);


        state = Chi(state);
        statep = Chi(statep);
        statep_1 = Chi(statep_1);
        statep_2 = Chi(statep_2);

        //state[0] = state[0] ^ 1;
        //statep[0] = statep[0] ^ 1;
        //statep_1[0] = statep_1[0] ^ 1;
        //statep_2[0] = statep_2[0] ^ 1;

        state = L1(state);
        statep = L1(statep);
        statep_1 = L1(statep_1);
        statep_2 = L1(statep_2);


        state = Chi(state);
        statep = Chi(statep);
        statep_1 = Chi(statep_1);
        statep_2 = Chi(statep_2);
        //  state = Chi(state);
        //   statep = Chi(statep);
        //   state = L1(state);
        //  statep = L1(statep);
        for (int location = 0; location < 257; location++)
        {
            bool linear_approx = 0;
            linear_approx = state[location] ^ statep[location] ^ statep_1[location] ^ statep_2[location];
            if (linear_approx == 0) { counter[location]++; }
            else { counter[location]--; }
        }
    }
    //  cout << "Correlation after mindle part  " << " = 2 ^ ";
    //  cout << (log2(abs(counter)) - log2(TEST_NUM + 1)) << " ";
    for (int i = 0; i < 257; i++)
    {
        cout << "(" << i << "," << double(counter[i]) / double(TEST_NUM + 1) << ")" << endl;
    }
}
int main() {

    vector<unsigned int> diff, c_1, diff_1;
    for (int i = 0; i < 257; i++)
    {
        diff.push_back(0);
        diff_1.push_back(0);
    }
    //  0, 77, 84, 87, 196, 203, 206, 247, 254
    diff[0] = 1;
    diff_1[32] = 1;
    for (int i = 0; i < 257; i++)
    {
        if (diff[i] != 0)
        {
            // cout << i << " ";
        }
    }
    cout << endl;
    //  diff = L1(diff);
    for (int i = 0; i < 257; i++)
    {
        if (diff[i] != 0)
        {
            // cout << i << " ";
        }
    }
    //   diff = L1(diff);
    cout << endl;
    for (int i = 0; i < 1; i++)
    {
        cout << i << " " << endl;
        HD_L_2n_all(diff, diff_1);
    }
    return 0;
}
