#include<iostream>
#include<random>
#include<cmath>

using namespace std;

int Sbox[] = { 0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2 };
int P[] = { 0,16,32,48,1,17,33,49,2,18,34,50,3,19,35,51,4,20,36,52,5,21,37,53,6,22,38,54,7,23,39,55,8,24,40,56,9,25,41,57,10,26,42,58,11,27,43,59,12,28,44,60,13,29,45,61,14,30,46,62,15,31,47, 63 };

int dot( int mask[], int X[] )
{
    int a = 0;
    for ( int i = 0; i < 64; i++ )
        a ^= mask[i] & X[i];
    return a;
}

void Perm( int X[] )
{
    int Y[64] = { 0 };
    for ( int i = 0; i < 64; i++ )
        Y[ P[i] ] = X[i];
    for ( int i = 0; i < 64; i++ )
        X[i] = Y[i];
}

void SboxPass( int X[] )
{
    for ( int i = 0; i < 16; i++ )
    {
        int a = 0;
        for ( int j = 0; j < 4; j++ )
            a += X[4 * i + j] << (3 - j);

        a = Sbox[a];

        for ( int j = 0; j < 4; j++ )
            X[4 * i + j] = a >> (3 - j) & 0x1;
    }
}

void bits2nibbles( int Bits[], int Nibs[] )
{
    for ( int i = 0; i < 16; i++ )
    {
        int a = 0;
        for ( int j = 0; j < 4; j++ )
            a += Bits[4 * i + j] << (3 - j);
        Nibs[i] = a;
    }
}

void nibbles2bits( int Nibs[], int Bits[] )
{
    for ( int i = 0; i < 16; i++ )
    {
        for ( int j = 0; j < 4; j++ )
            Bits[4 * i + j] = Nibs[i] >> (3 - j) & 0x1;
    }
}

void PresentEncryption( int P[], int R )
{
    for ( int i = 0; i < R; i++ )
    {
        SboxPass( P );
        if ( i < R - 1 )
            Perm( P );
    }
}

int main()
{
    random_device rd;
    mt19937 gen( rd() );
    uniform_int_distribution<int> dis(0, 1); // 范围 [1, 100]
    int NUM = 26;

    double cor = 0;
    for ( int test = 0; test < ( 1 << NUM ); test++ )
    {
        if ( test % (1 << 26) == 0 )
            cout << test << endl;
        int X[64];
        int X1[64];
        for ( int i = 0; i < 64; i++ )
        {
            X[i] = dis( gen );
            X1[i] = X[i];
        }

        //X1[49] = X[49] ^ 1;
        //X1[50] = X[50] ^ 1;
        //X1[51] = X[51] ^ 1;

        //X1[61] = X[61] ^ 1;
        //X1[62] = X[62] ^ 1;
        //X1[63] = X[63] ^ 1;
        X1[20] = X[20] ^ 1;
        X1[21] = X[21] ^ 1;
        X1[23] = X[23] ^ 1;
        PresentEncryption( X, 6 );
        PresentEncryption( X1, 6 );

        if ( ( X[50] ^ X1[50] ) == 0 )
            cor += 1; 
        else
            cor -= 1; 
    }

    cout << cor / ( 1 << 26 ) << endl;
    cout << log2( cor / ( 1 << 26 ) ) << endl;
}
