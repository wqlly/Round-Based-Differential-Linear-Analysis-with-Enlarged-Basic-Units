# Round-Based Differential-Linear Analysis with Enlarged Basic Units

This repository contains the source code for the paper:

**Round-Based Differential-Linear Analysis with Enlarged Basic Units**

The code includes differential-linear and higher-order differential-linear analysis for the following lightweight cryptographic primitives:

- PRESENT
- GIFT-64 / GIFT-128
- Subterranean-2.0
- Koala-p

The main idea is to improve the round-based geometric approximation by using enlarged basic units.

For PRESENT and GIFT, several S-boxes are combined into Super S-box units.

For Subterranean-2.0 and Koala-p, enlarged local nonlinear units are used.

## Directory Structure

```text
submission/
├── PRESENT/
├── GIFT/
├── Subteranean/
│   ├── first_order/
│   ├── second_order/
│   └── third_order/
└── Koala/
    ├── first/
    ├── second/
    └── third/
```

## Requirements

The Python programs require:

- Python 3
- NumPy

Install NumPy with:

```bash
pip install numpy
```

The C++ programs require a compiler supporting C++17.

Some programs use OpenMP for parallel computation.

A typical compilation command is:

```bash
g++ -O3 -std=c++17 -fopenmp program.cpp -o program
```

## Usage

Run Python programs with:

```bash
python3 program.py
```

Run C++ programs with:

```bash
./program
```
