#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometric differential-linear evaluation for GIFT-128 using 16-bit
Super S-boxes.

GIFT-128 contains 32 parallel 4-bit S-boxes.  The program represents them as
eight 16-bit factors and alternates between two factorizations:

    B orientation, contiguous rows:
        (0,1,2,3), (4,5,6,7), ..., (28,29,30,31)

    A orientation, strided groups:
        (0,8,16,24), (1,9,17,25), ..., (7,15,23,31)

The propagation cycle is

    B -- PRE1 + S-box layer --> A
      -- PRE2 + S-box layer --> B.

PRE1 is internal to each 16-bit factor and implements

    u[r][q][s] -> v[r][3+s-q][s].

PRE2 converts an A coordinate u[r][c][s] by

    a = 2*c + r//4,
    b = r % 4,

and then applies the GIFT-128 bit permutation.  In the left-to-right bit
numbering used by the Python arrays, its forward form is

    x = (a//4 + 2*s - 2*b - 2) mod 8,
    y = a mod 4,
    z = s.

The extra -2 is the numbering conversion relative to the commonly written
right-to-left formula.  make_pre2_gift128() uses the inverse relation because
the tensor lookup is out[v] = in[pre[v]].  self_test() checks PRE1, PRE2 and
the supplied inverse permutation on all 128 one-bit positions.
"""
from __future__ import annotations

import itertools
import math
from typing import Iterable, Sequence

import numpy as np

# Parameters are written directly in the __main__ block at the end.

# =============================================================================
# GIFT-128 constants.
# =============================================================================
N_BITS = 128
N_SBOX = 32
N_GROUPS = 8
GROUP_WIDTH = 4
TENSOR_SIZE = 16 ** GROUP_WIDTH

SBOX = [
    0x1, 0xA, 0x4, 0xC,
    0x6, 0xF, 0x3, 0x9,
    0x2, 0xD, 0xB, 0x7,
    0x5, 0x0, 0x8, 0xE,
]

# Supplied inverse-permutation table in the bit numbering used by the uploaded
# GIFT-64 program.  pass_inv_p() intentionally preserves that convention.
INVP=[12, 1, 6, 11, 28, 17, 22, 27, 44, 33, 38, 43, 60, 49, 54, 59, 76, 65, 70, 75, 92, 81, 86, 91, 108, 97, 102, 107, 124, 113, 118, 123, 8, 13, 2, 7, 24, 29, 18, 23, 40, 45, 34, 39, 56, 61, 50, 55, 72, 77, 66, 71, 88, 93, 82, 87, 104, 109, 98, 103, 120, 125, 114, 119, 4, 9, 14, 3, 20, 25, 30, 19, 36, 41, 46, 35, 52, 57, 62, 51, 68, 73, 78, 67, 84, 89, 94, 83, 100, 105, 110, 99, 116, 121, 126, 115, 0, 5, 10, 15, 16, 21, 26, 31, 32, 37, 42, 47, 48, 53, 58, 63, 64, 69, 74, 79, 80, 85, 90, 95, 96, 101, 106, 111, 112, 117, 122, 127]

# A orientation: fix row parity and column, vary floor(row/2).
# Group number is g = 4*(row mod 2) + column.
GROUPS_A = tuple(
    tuple(4 * (parity + 2 * q) + column for q in range(4))
    for parity in range(2)
    for column in range(4)
)
#print(GROUPS_A)
# B orientation: four consecutive S-boxes in each row.
GROUPS_B = tuple(
    tuple(4 * row + column for column in range(4))
    for row in range(8)
)
#print(GROUPS_B)

def dot(a: int, x: int, n: int) -> int:
    """Binary inner product of two n-bit integers."""
    z = a & x
    parity = 0
    for i in range(n):
        parity ^= (z >> i) & 1
    return parity


def fwt(v: np.ndarray) -> None:
    """In-place Walsh-Hadamard transform."""
    n = len(v)
    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            for j in range(i, i + h):
                x = v[j]
                y = v[j + h]
                v[j] = x + y
                v[j + h] = x - y
        h <<= 1


def gen_ddt(sbox: Sequence[int], n: int) -> np.ndarray:
    size = 1 << n
    ddt = np.zeros((size, size), dtype=np.float64)
    for u in range(size):
        for x in range(size):
            ddt[sbox[x] ^ sbox[x ^ u], u] += 1.0
    return ddt / size


def gen_lat(sbox: Sequence[int], n: int) -> np.ndarray:
    size = 1 << n
    lat = np.zeros((size, size), dtype=np.float64)
    for b in range(size):
        v = np.empty(size, dtype=np.float64)
        for x in range(size):
            v[x] = 1.0 if dot(b, sbox[x], n) == 0 else -1.0
        fwt(v)
        lat[b] = v
    return lat / size


def pass_inv_p(x: np.ndarray) -> np.ndarray:
    """Propagate a 128-bit mask backward through the GIFT-128 permutation."""
    if x.shape != (N_BITS,):
        raise ValueError(f"expected shape ({N_BITS},), got {x.shape}")
    y = np.zeros(N_BITS, dtype=x.dtype)
    for i in range(N_BITS):
        y[INVP[i]] = x[i]
    return y


def log2_abs(x: float) -> float:
    return math.log2(abs(x)) if x != 0.0 else -math.inf


def tuple_to_index(t: Sequence[int]) -> int:
    if len(t) != 4:
        raise ValueError("a Super S-box coordinate must contain four nibbles")
    return ((int(t[0]) * 16 + int(t[1])) * 16 + int(t[2])) * 16 + int(t[3])


def index_to_tuple(index: int) -> tuple[int, int, int, int]:
    return (
        (index >> 12) & 0xF,
        (index >> 8) & 0xF,
        (index >> 4) & 0xF,
        index & 0xF,
    )


def cell_nibble(bits: np.ndarray, sbox_position: int) -> int:
    base = 4 * sbox_position
    return (
        8 * int(bits[base])
        + 4 * int(bits[base + 1])
        + 2 * int(bits[base + 2])
        + int(bits[base + 3])
    )


def init_super(x: np.ndarray) -> np.ndarray:
    """Build the eight B-oriented contiguous-row 16-bit factors."""
    if x.shape != (N_SBOX, 16):
        raise ValueError(f"expected X shape ({N_SBOX},16), got {x.shape}")

    g = np.empty((N_GROUPS, 16, 16, 16, 16), dtype=np.float64)
    for group_index, group in enumerate(GROUPS_B):
        t = np.multiply.outer(x[group[0]], x[group[1]])
        t = np.multiply.outer(t, x[group[2]])
        t = np.multiply.outer(t, x[group[3]])
        g[group_index] = t.reshape(16, 16, 16, 16)
    return g


def apply_axis(tensor: np.ndarray, matrix: np.ndarray, axis: int) -> np.ndarray:
    y = np.tensordot(matrix, tensor, axes=([1], [axis]))
    return np.moveaxis(y, 0, axis)


def apply_sbox(tensor: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    for axis in range(GROUP_WIDTH):
        tensor = apply_axis(tensor, matrix, axis)
    return tensor


def make_pre1_gift() -> np.ndarray:
    """
    GIFT local 16-bit permutation, unchanged from the GIFT-64 code.

    For each bit s of input axis q, the corresponding axis is
        (s + 3 - q) mod 4.
    This reflection is self-inverse, so the same table is used for lookup.
    """
    pre = np.empty(TENSOR_SIZE, dtype=np.uint32)
    for index in range(TENSOR_SIZE):
        v = index_to_tuple(index)
        u = [0, 0, 0, 0]
        for q in range(4):
            nibble = v[q]
            for s in range(4):
                if (nibble >> (3 - s)) & 1:
                    u[(s + 3 - q) % 4] |= 1 << (3 - s)
        pre[index] = tuple_to_index(u)
    return pre


def make_pre2_gift128() -> list[np.ndarray]:
    """
    Construct the A -> B lookup tables for the second permutation step.

    The source A-oriented coordinates are ``u[r][c][s]`` with

        0 <= r < 8, 0 <= c,s < 4.

    First convert them to physical row/column coordinates

        a = 2*c + r//4,
        b = r % 4.

    With the left-to-right bit numbering used by this program, the forward
    GIFT-128 mapping is

        x = (a//4 + 2*s - 2*b - 2) % 8,
        y = a % 4,
        z = s.

    The lookup table is used as ``out[v] = in[pre[v]]``, so we need the
    inverse relation for a fixed B-oriented output bit ``[x][y][s]``:

        r = 4*(y % 2) + (s + 3 - x//2) % 4,
        c = 2*(x % 2) + y//2,
        source bit = s.

    Every B-oriented output factor receives bits from all eight A-oriented
    source factors.  Therefore each table has shape ``(16**4, 8)``.
    """
    all_pre: list[np.ndarray] = []

    for x in range(N_GROUPS):
        pre = np.empty((TENSOR_SIZE, N_GROUPS), dtype=np.uint32)

        for index in range(TENSOR_SIZE):
            v = index_to_tuple(index)
            u = [[0, 0, 0, 0] for _ in range(N_GROUPS)]

            for y in range(GROUP_WIDTH):
                nibble = v[y]
                c = 2 * (x & 1) + (y >> 1)

                for s in range(4):
                    if (nibble >> (3 - s)) & 1:
                        r = 4 * (y & 1) + ((s + 3 - (x >> 1)) & 3)
                        u[r][c] |= 1 << (3 - s)

            for r in range(N_GROUPS):
                pre[index, r] = tuple_to_index(u[r])

        all_pre.append(pre)

    return all_pre


PRE1 = make_pre1_gift()
PRE2 = make_pre2_gift128()


def apply_p_internal(tensor: np.ndarray, pre: np.ndarray) -> np.ndarray:
    return tensor.reshape(-1)[pre].reshape(16, 16, 16, 16)


def apply_inter_p(g: np.ndarray, pre_all: Sequence[np.ndarray]) -> np.ndarray:
    """Apply pre2, multiplying the contributions of all eight source factors."""
    flat = [g[i].reshape(-1) for i in range(N_GROUPS)]
    out = np.empty_like(g)

    for output_group in range(N_GROUPS):
        pre = pre_all[output_group]
        values = np.ones(TENSOR_SIZE, dtype=np.float64)
        for source_group in range(N_GROUPS):
            values *= flat[source_group][pre[:, source_group]]
        out[output_group] = values.reshape(16, 16, 16, 16)

    return out


def propagate(x: np.ndarray, matrix: np.ndarray, n: int) -> tuple[np.ndarray, str]:
    """Propagate n intermediate S-box/permutation steps."""
    if n < 0:
        raise ValueError("number of propagated intermediate rounds must be nonnegative")

    g = init_super(x)
    orientation = "B"

    for _ in range(n):
        if orientation == "B":
            # B -> A: PRE1 is internal to each 16-bit factor.
            for group in range(N_GROUPS):
                g[group] = apply_p_internal(g[group], PRE1)
            for group in range(N_GROUPS):
                g[group] = apply_sbox(g[group], matrix)
            orientation = "A"
        else:
            # A -> B: PRE2 combines all eight source factors.
            g = apply_inter_p(g, PRE2)
            for group in range(N_GROUPS):
                g[group] = apply_sbox(g[group], matrix)
            orientation = "B"

    return g, orientation


def eval_mask(g: np.ndarray, orientation: str, bits: np.ndarray) -> float:
    value = 1.0
    groups = GROUPS_A if orientation == "A" else GROUPS_B

    for group_index, group in enumerate(groups):
        coordinate = tuple(cell_nibble(bits, position) for position in group)
        value *= g[group_index][coordinate]

    return float(value)


def final_layer(
    g: np.ndarray,
    orientation: str,
    matrix: np.ndarray,
    mask: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Apply the final S-box correlation matrix to the selected output mask cells."""
    length = len(mask)
    if length == 0:
        raise ValueError("MASK must contain at least one active S-box position")

    result_length = 1 << (4 * length)
    values = np.zeros(result_length, dtype=np.float64)

    for value in range(result_length):
        v = np.zeros(N_BITS, dtype=np.int8)
        for j, (position, _mask_value) in enumerate(mask):
            if not 0 <= position < N_SBOX:
                raise ValueError(f"mask S-box position {position} is outside 0..31")
            nibble = (value >> (4 * j)) & 0xF
            for bit in range(4):
                v[4 * position + bit] = (nibble >> (3 - bit)) & 1

        u = pass_inv_p(v)
        values[value] = eval_mask(g, orientation, u)

    combined_matrix = matrix
    for _ in range(length - 1):
        combined_matrix = np.kron(combined_matrix, matrix)

    return combined_matrix @ values


def evaluate_superbox(
    rounds: int,
    lat: np.ndarray,
    ddt: np.ndarray,
    diff: Sequence[tuple[int, int]],
    mask: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Evaluate the geometric DL correlation vector for one distinguisher."""
    if rounds < 2:
        raise ValueError("the Super S-box evaluator requires rounds >= 2")
    if lat.shape != (16, 16):
        raise ValueError(f"LAT must have shape (16,16), got {lat.shape}")
    if ddt.shape != (16, 16):
        raise ValueError(f"DDT must have shape (16,16), got {ddt.shape}")
    if not diff:
        raise ValueError("DIFF must contain at least one active S-box")
    if not mask:
        raise ValueError("MASK must contain at least one active S-box")

    diff_positions = [position for position, _ in diff]
    mask_positions = [position for position, _ in mask]
    if len(set(diff_positions)) != len(diff_positions):
        raise ValueError("DIFF contains duplicate S-box positions")
    if len(set(mask_positions)) != len(mask_positions):
        raise ValueError("MASK contains duplicate S-box positions")

    for position, value in diff:
        if not 0 <= position < N_SBOX:
            raise ValueError(f"difference S-box position {position} is outside 0..31")
        if not 0 <= value < 16:
            raise ValueError(f"difference value {value} is outside 0..15")

    for position, value in mask:
        if not 0 <= position < N_SBOX:
            raise ValueError(f"mask S-box position {position} is outside 0..31")
        if not 0 <= value < 16:
            raise ValueError(f"mask value {value} is outside 0..15")

    matrix = lat ** 2
    correlation = np.zeros(1 << (4 * len(mask)), dtype=np.float64)
    output_differences: Iterable[tuple[int, ...]] = itertools.product(
        *([range(16)] * len(diff))
    )

    for combination in output_differences:
        probability = 1.0
        valid = True
        for i, output_difference in enumerate(combination):
            p = ddt[output_difference, diff[i][1]]
            if p == 0.0:
                valid = False
                break
            probability *= p
        if not valid:
            continue

        x = np.zeros((N_SBOX, 16), dtype=np.float64)
        for i, output_difference in enumerate(combination):
            x[diff[i][0], output_difference] = 1.0

        # Inactive S-boxes have the zero difference with probability one.
        for position in range(N_SBOX):
            if np.sum(x[position]) == 0.0:
                x[position, 0] = 1.0

        for position in range(N_SBOX):
            fwt(x[position])

        g, orientation = propagate(x, matrix, rounds - 2)
        correlation += probability * final_layer(g, orientation, matrix, mask)

    return correlation


def getBias_Opt4_Superbox(ROUND, LAT, Diff, Mask):
    """Keep the same calling interface as the previous PRESENT program."""
    DDT = gen_ddt(SBOX, 4)
    return evaluate_superbox(ROUND, LAT, DDT, Diff, Mask)


def _one_hot_tensor_index(axis: int, bit: int) -> int:
    coord = [0, 0, 0, 0]
    coord[axis] = 1 << (3 - bit)
    return tuple_to_index(coord)


def self_test() -> None:
    """Check PRE1, PRE2 and INVP on all 128 one-bit positions."""
    if len(INVP) != N_BITS or sorted(INVP) != list(range(N_BITS)):
        raise AssertionError("INVP is not a permutation of 0..127")

    if PRE1.shape != (TENSOR_SIZE,):
        raise AssertionError(f"PRE1 has wrong shape: {PRE1.shape}")
    if len(PRE2) != N_GROUPS:
        raise AssertionError(f"PRE2 must contain {N_GROUPS} tables")
    for table in PRE2:
        if table.shape != (TENSOR_SIZE, N_GROUPS):
            raise AssertionError(f"PRE2 table has wrong shape: {table.shape}")

    # B -> A: PRE1 is internal.  For an A-oriented output bit, INVP gives
    # the corresponding bit in the B-oriented source factor with the same
    # factor number.
    for output_group, output_nibbles in enumerate(GROUPS_A):
        source_nibbles = GROUPS_B[output_group]
        source_axis = {nibble: axis for axis, nibble in enumerate(source_nibbles)}

        for output_axis, output_nibble in enumerate(output_nibbles):
            for bit in range(4):
                output_index = _one_hot_tensor_index(output_axis, bit)
                source_index = int(PRE1[output_index])
                source_coord = index_to_tuple(source_index)

                source_global_bit = INVP[4 * output_nibble + bit]
                source_nibble = source_global_bit // 4
                source_bit = source_global_bit % 4
                if source_nibble not in source_axis:
                    raise AssertionError(
                        "PRE1 maps outside its source factor at "
                        f"group={output_group}, axis={output_axis}, bit={bit}"
                    )

                expected_coord = [0, 0, 0, 0]
                expected_coord[source_axis[source_nibble]] = 1 << (3 - source_bit)
                if source_coord != tuple(expected_coord):
                    raise AssertionError(
                        f"PRE1 mismatch at group={output_group}, "
                        f"axis={output_axis}, bit={bit}"
                    )

    # A -> B: PRE2 combines the eight A-oriented source factors.
    for output_group, output_nibbles in enumerate(GROUPS_B):
        for output_axis, output_nibble in enumerate(output_nibbles):
            for bit in range(4):
                output_index = _one_hot_tensor_index(output_axis, bit)
                row = PRE2[output_group][output_index]

                source_global_bit = INVP[4 * output_nibble + bit]
                source_nibble = source_global_bit // 4
                source_bit = source_global_bit % 4

                source_group = next(
                    group_index
                    for group_index, group in enumerate(GROUPS_A)
                    if source_nibble in group
                )
                source_axis = GROUPS_A[source_group].index(source_nibble)

                for group_index in range(N_GROUPS):
                    coord = index_to_tuple(int(row[group_index]))
                    expected_coord = [0, 0, 0, 0]
                    if group_index == source_group:
                        expected_coord[source_axis] = 1 << (3 - source_bit)
                    if coord != tuple(expected_coord):
                        raise AssertionError(
                            "PRE2 mismatch at "
                            f"out_group={output_group}, out_axis={output_axis}, "
                            f"bit={bit}, source_group={group_index}"
                        )

    # Direct backward-mask test.
    for output_bit in range(N_BITS):
        x = np.zeros(N_BITS, dtype=np.int8)
        x[output_bit] = 1
        y = pass_inv_p(x)
        positions = np.flatnonzero(y)
        if len(positions) != 1 or int(positions[0]) != INVP[output_bit]:
            raise AssertionError(f"pass_inv_p mismatch at output bit {output_bit}")

    print("Self-test passed: PRE1, PRE2 and INVP agree on all 128 bit positions.")


if __name__ == "__main__":
    self_test()
    LAT = gen_lat(SBOX, 4)
    #print(LAT)

##    tests = [(8, [[9, 8]], [[ 6, 2]],2)]
##    for r, d, m, i in tests:
##        print(f"ROUND {r}")
##        res = getBias_Opt4_Superbox(r, LAT, d, m)[i]
##        print(f"Result: {res}")
##        print(f"log2: {log2_abs(res)}")
    fin = 0
    for A in range(32):
        for B in range(32):
            tests = [(8, [[A, 8]], [[B, 2]], 2)]
            for r, d, m, i in tests:
                res = getBias_Opt4_Superbox(r, LAT, d, m)[i]
                if abs(res) > abs(fin):
                    fin = res
                    print(f"input/output:{A} {8} / {B} {2}")
                    print(f"Result: {res}")
                    print(f"log2:{log2_abs(res)}")
##    fin = 0
##    for A in range(32):
##        #print(A)
##        for a in range(1, 16):
##            for B in range(32):
##                for b in range(1, 16):
##                    tests = [(18, [[A, a]], [[B, b]], b)]
##                    for r, d, m, i in tests:
##                        res = getBias_Opt4_Superbox(r, LAT, d, m)[i]
##                        if abs(res) > abs(fin):
##                            fin = res
##                            print(f"input/output:{A} {a} / {B} {b}")
##                            print(f"Result: {res}")
##                            print(f"log2: {log2_abs(res)}")
##test=[48, 1, 18, 35, 112, 65, 82, 99, 52, 5, 22, 39, 116, 69, 86, 103, 56, 9, 26, 43, 120, 73, 90, 107, 60, 13, 30, 47, 124, 77, 94, 111, 32, 49, 2, 19, 96, 113, 66, 83, 36, 53, 6, 23, 100, 117, 70, 87, 40, 57, 10, 27, 104, 121, 74, 91, 44, 61, 14, 31, 108, 125, 78, 95, 16, 33, 50, 3, 80, 97, 114, 67, 20, 37, 54, 7, 84, 101, 118, 71, 24, 41, 58, 11, 88, 105, 122, 75, 28, 45, 62, 15, 92, 109, 126, 79, 0, 17, 34, 51, 64, 81, 98, 115, 4, 21, 38, 55, 68, 85, 102, 119, 8, 25, 42, 59, 72, 89, 106, 123, 12, 29, 46, 63, 76, 93, 110, 127]
##for i in range(128):
##    a=i//16
##    b=(i-a*16)//4
##    c=i-a*16-b*4
##    tes=test[i]
##    x=tes//16
##    y=(tes-x*16)//4
##    z=tes-x*16-y*4
##    print('a=',a,'b=',b,'c=',c,'x=',x,'y=',y,'z=',z)
##    
