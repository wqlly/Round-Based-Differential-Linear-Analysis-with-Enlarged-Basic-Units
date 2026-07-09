#!/usr/bin/env python3
"""
Hybrid differential-linear estimator for Koala-p.

The analyzed six-chi-layer structure follows the paper's shifted boundary:

    Delta_0 -- chi -- L -- chi -- L --> Delta_2
            -- chi -- L -- chi -- L -- chi -- L -- chi --> lambda,

where Koala's linear part is

    pi    : y_i = x_{121 i},
    theta : z_i = y_i + y_{i+3} + y_{i+10},
    L     = theta o pi.

Thus one exact front step in this program is chi followed by L. This is the
same boundary used by the paper for the Koala-p distinguisher

    [0] --(two exact chi/L steps)-->
    [0,77,84,87,196,203,206,247,254]
    --(four-round geometric tail)--> [232].

The first two steps are expanded as the complete exact differential hull.
Every resulting Delta_2 is evaluated by a byte/block one-wise round-based
geometric tail. Worker processes evaluate tail endpoints in parallel and the
current partial correlation is printed periodically.

Affine iota constants do not affect differences. In the value-mask-zero
squared-correlation tail used here, their signs also disappear.

Run
---
Edit the FIXED_* constants near the top of the file, then execute:

    python3 koala_hybrid_fixed.py
"""
from __future__ import annotations

# Prevent NumPy/BLAS from spawning its own thread pool inside every process.
# This matters a lot when running with tens or hundreds of Python workers.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

N = 257
MASK_N = (1 << N) - 1
INV121 = pow(121, -1, N)  # 17 modulo 257

# ============================================================================
# Fixed paper parameters
# Edit only this section when changing the distinguisher.
# No command-line arguments are required.
# ============================================================================
FIXED_ROUNDS = 6
FIXED_FRONT_ROUNDS = 2
FIXED_INPUT_DIFF = "0"
FIXED_TARGET = "160"
FIXED_BLOCK_SIZE = 4
FIXED_WORKERS = 16
FIXED_BATCH_SIZE = 2
FIXED_PROGRESS_EVERY = 100.0

# Optional truncation/debug controls.
FIXED_MIN_PROB_LOG2 = None
FIXED_MAX_INSTANCES = None
FIXED_SORT = "prob"

# Output controls.
FIXED_OUTPUT_JSON = "koala_6round_result.json"
FIXED_DUMP_DISTRIBUTION = None
FIXED_DUMP_CONTRIBS = None

# Multiprocessing controls.
FIXED_START_METHOD = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
FIXED_MAXTASKS_PER_CHILD = None

# Special modes. Normally keep both False.
FIXED_RUN_SELF_TEST = False
FIXED_RUN_PAPER_CHARACTERISTIC_ONLY = False

# Same 2x2 matrices as in the uploaded Subt.py.  In the value-mask-zero
# restriction, C_chi[v,u]^2 is handled by kron(M, M), i.e. a 4x4 state matrix.
M00 = 0.5 * np.array([[1, 1], [1, 1]], dtype=np.float64)
M01 = 0.5 * np.array([[1, -1], [1, -1]], dtype=np.float64)
M10 = 0.5 * np.array([[1, 1], [-1, 1]], dtype=np.float64)
M11 = 0.5 * np.array([[1, -1], [-1, -1]], dtype=np.float64)
MT = [M00, M01, M10, M11]
BIT4 = np.stack([np.kron(M, M) for M in MT])


def parity(x: int) -> int:
    return x.bit_count() & 1


def log2_abs(x: float) -> float:
    return math.log2(abs(x)) if x != 0.0 else -math.inf


def support_to_mask(supp: Iterable[int], n: int = N) -> int:
    out = 0
    for i in supp:
        if i < 0 or i >= n:
            raise ValueError(f"bit index {i} is outside [0,{n})")
        out ^= 1 << i
    return out


def mask_to_support(mask: int, n: int = N) -> List[int]:
    return [i for i in range(n) if (mask >> i) & 1]


def parse_support(text: str, n: int = N) -> List[int]:
    """Parse '0,64,85', '[0,64,85]', '0', or '' into a support list."""
    text = text.strip().replace("[", "").replace("]", "")
    if not text:
        return []
    out = [int(x.strip()) for x in text.split(",") if x.strip()]
    for i in out:
        if i < 0 or i >= n:
            raise ValueError(f"bit index {i} is outside [0,{n})")
    return out


def format_support(mask: int, max_len: int = 20) -> str:
    supp = mask_to_support(mask)
    if len(supp) <= max_len:
        return "[" + ",".join(map(str, supp)) + "]"
    head = ",".join(map(str, supp[:max_len]))
    return f"[{head},...;wt={len(supp)}]"


def make_block_sizes(n: int = N, block_size: int = 8) -> List[int]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    q, r = divmod(n, block_size)
    out = [block_size] * q
    if r:
        out.append(r)
    return out


@lru_cache(maxsize=None)
def sign_table(m: int) -> np.ndarray:
    """sign_table[m][delta_block, mask] = (-1)^{<delta_block, mask>}."""
    d = 1 << m
    tab = np.empty((d, d), dtype=np.float64)
    for delta in range(d):
        for a in range(d):
            tab[delta, a] = -1.0 if parity(delta & a) else 1.0
    return tab


@lru_cache(maxsize=None)
def block_coefficients(m: int) -> np.ndarray:
    """Return coeff[ctx,u,:,:] for a block of m consecutive bits.

    ctx has m+1 bits:
        ctx bit 0     = V_{start-1},
        ctx bit t + 1 = V_{start+t}.

    For local bit t, the bit matrix is determined by
        a = V_{j-1} = ctx[t],
        b = U_j + V_j = u_t xor ctx[t+1].

    coeff[ctx,u] is the ordered product of the m 4x4 bit matrices:
        A_{start+m-1} ... A_start.

    The final chunk matrix for a factor gamma_b is
        sum_u gamma_b[u] * coeff[ctx,u].
    """
    if m <= 0:
        raise ValueError("block length must be positive")
    d = 1 << m
    uvals = np.arange(d, dtype=np.uint16)
    prod0 = np.broadcast_to(np.eye(4, dtype=np.float64), (d, 4, 4)).copy()
    coeff = np.empty((1 << (m + 1), d, 4, 4), dtype=np.float64)

    for ctx in range(1 << (m + 1)):
        prod = prod0.copy()
        for t in range(m):
            a = (ctx >> t) & 1
            vj = (ctx >> (t + 1)) & 1
            idx = a * 2 + (((uvals >> t) & 1) ^ vj)
            prod = BIT4[idx] @ prod
        coeff[ctx] = prod
    return coeff


class ByteOneWiseKoalaDL:
    """Byte-wise tensor approximation using the transition C_chi[v,u]^2.

    The maintained vector is

        gamma[u] ~= prod_b gamma_b[u_b]

    over the chosen block partition.  For N=257 and block_size=8, there are
    32 factors of length 256 and one factor of length 2.
    """

    def __init__(self, block_sizes: Sequence[int]):
        self.block_sizes = tuple(int(x) for x in block_sizes)
        if any(m <= 0 for m in self.block_sizes):
            raise ValueError("all block sizes must be positive")
        self.n = sum(self.block_sizes)
        if self.n != N:
            raise ValueError(f"this implementation is configured for N={N}, got {self.n}")
        self.dims = tuple(1 << m for m in self.block_sizes)
        self.starts: List[int] = []
        s = 0
        for m in self.block_sizes:
            self.starts.append(s)
            s += m

        # Precompute sign tables and chunk coefficient tables inside each worker.
        self._sign_tables = {m: sign_table(m) for m in set(self.block_sizes)}
        for m in set(self.block_sizes):
            _ = block_coefficients(m)

        self._linear_pullback_parts = self._precompute_linear_pullbacks()

    def mask_to_block_values(self, mask: int) -> List[int]:
        return [int((mask >> start) & ((1 << m) - 1)) for start, m in zip(self.starts, self.block_sizes)]

    def initial_factors_from_mask(self, diff_mask: int) -> List[np.ndarray]:
        parts = self.mask_to_block_values(diff_mask)
        return [self._sign_tables[m][p] for p, m in zip(parts, self.block_sizes)]

    def evaluate_mask(self, factors: Sequence[np.ndarray], mask: int) -> float:
        parts = self.mask_to_block_values(mask)
        out = 1.0
        for f, p in zip(factors, parts):
            out *= float(f[p])
            if out == 0.0:
                break
        return out

    def _tables_from_factors(self, factors: Sequence[np.ndarray]) -> List[np.ndarray]:
        if len(factors) != len(self.block_sizes):
            raise ValueError(f"expected {len(self.block_sizes)} factors, got {len(factors)}")
        tables: List[np.ndarray] = []
        for i, (f, m, d) in enumerate(zip(factors, self.block_sizes, self.dims)):
            arr = np.asarray(f, dtype=np.float64)
            if arr.shape != (d,):
                raise ValueError(f"factor {i} must have shape ({d},), got {arr.shape}")
            coeff = block_coefficients(m)
            # table[ctx] = sum_u arr[u] * coeff[ctx,u].
            table = np.einsum("u,curs->crs", arr, coeff, optimize=True)
            tables.append(table)
        return tables

    def pass_chi(self, factors: Sequence[np.ndarray], force_zero_to_one: bool = True) -> List[np.ndarray]:
        """Apply gamma'[v] = sum_u C_chi[v,u]^2 gamma[u], then refactor by blocks.

        Only unit-block output coordinates are computed:
            output_factor[b][a] = gamma'[tau_b(a)].
        """
        tables = self._tables_from_factors(factors)
        base = [tab[0] for tab in tables]
        k = len(self.block_sizes)

        # prefix[i] = M_{i-1} ... M_0
        prefix = [np.eye(4, dtype=np.float64)]
        for i in range(k):
            prefix.append(base[i] @ prefix[-1])

        # suffix[i] = M_{k-1} ... M_i
        suffix: List[Optional[np.ndarray]] = [None] * (k + 1)
        suffix[k] = np.eye(4, dtype=np.float64)
        for i in range(k - 1, -1, -1):
            suffix[i] = suffix[i + 1] @ base[i]  # type: ignore[operator]

        # Middle product for the wrap case: M_{k-2} ... M_1.
        if k >= 3:
            middle_no_ends = np.eye(4, dtype=np.float64)
            for i in range(1, k - 1):
                middle_no_ends = base[i] @ middle_no_ends
        else:
            middle_no_ends = np.eye(4, dtype=np.float64)

        out: List[np.ndarray] = []
        for target, m in enumerate(self.block_sizes):
            d = 1 << m
            vals = np.empty(d, dtype=np.float64)

            if target < k - 1:
                left = suffix[target + 2]
                right = prefix[target]
                assert left is not None
                for a in range(d):
                    if force_zero_to_one and a == 0:
                        vals[a] = 1.0
                        continue
                    ctx_target = a << 1
                    ctx_next = (a >> (m - 1)) & 1
                    P = left @ tables[target + 1][ctx_next] @ tables[target][ctx_target] @ right
                    vals[a] = float(np.trace(P))
            else:
                # The last block's last bit is followed cyclically by block 0.
                for a in range(d):
                    if force_zero_to_one and a == 0:
                        vals[a] = 1.0
                        continue
                    ctx_target = a << 1
                    ctx_next = (a >> (m - 1)) & 1
                    P = tables[target][ctx_target] @ middle_no_ends @ tables[0][ctx_next]
                    vals[a] = float(np.trace(P))
            out.append(vals)
        return out

    @staticmethod
    def linear_T_mask_koala(v: int) -> int:
        """Return L^T v for Koala's L = theta o pi.

        pi maps y_i = x_{121 i}, and theta maps
            z_i = y_i + y_{i+3} + y_{i+10}.
        Hence
            z_i = x_{121 i} + x_{121(i+3)} + x_{121(i+10)}.

        Therefore one output-mask bit i pulls back to input-mask bits
            121*i, 121*(i+3), 121*(i+10) modulo 257.
        """
        u = 0
        x = v
        while x:
            lb = x & -x
            i = lb.bit_length() - 1
            u ^= 1 << ((121 * i) % N)
            u ^= 1 << ((121 * (i + 3)) % N)
            u ^= 1 << ((121 * (i + 10)) % N)
            x ^= lb
        return u

    def _precompute_linear_pullbacks(self) -> List[np.ndarray]:
        """For every unit-block output mask, precompute block parts of L^T mask."""
        out: List[np.ndarray] = []
        k = len(self.block_sizes)
        for start, m in zip(self.starts, self.block_sizes):
            d = 1 << m
            arr = np.empty((d, k), dtype=np.uint16)
            for a in range(d):
                v = int(a) << start
                u = self.linear_T_mask_koala(v)
                arr[a, :] = self.mask_to_block_values(u)
            out.append(arr)
        return out

    def pass_linear(self, factors: Sequence[np.ndarray], force_zero_to_one: bool = True) -> List[np.ndarray]:
        """Pass through Koala L=theta o pi under the rank-one refactorization.

        Since we use squared correlations, affine constants do not matter.  For
        a linear layer, gamma'[v] = gamma[L^T v].
        """
        out: List[np.ndarray] = []
        factors_np = [np.asarray(f, dtype=np.float64) for f in factors]
        for pull in self._linear_pullback_parts:
            vals = np.ones(pull.shape[0], dtype=np.float64)
            for j, f in enumerate(factors_np):
                vals *= f[pull[:, j]]
            if force_zero_to_one:
                vals[0] = 1.0
            out.append(vals)
        return out

    def tail_correlation(self, tail_rounds: int, diff_mask: int, target_mask: int) -> float:
        """Approximate the tail correlation from diff_mask to target_mask.

        The tail of length t is
            chi, L, chi, L, ..., chi
        with t chi layers and t-1 linear layers.  This matches the structure
        chi o rho_{...} o ... used for Koala in the paper, ignoring iota
        constants after squaring.
        """
        if tail_rounds < 0:
            raise ValueError("tail_rounds must be nonnegative")
        factors = self.initial_factors_from_mask(diff_mask)
        if tail_rounds == 0:
            return self.evaluate_mask(factors, target_mask)
        for r in range(tail_rounds):
            factors = self.pass_chi(factors)
            if r < tail_rounds - 1:
                factors = self.pass_linear(factors)
        return self.evaluate_mask(factors, target_mask)


# ---------------------------------------------------------------------------
# Exact differential expansion for the first full Koala rho rounds.
# ---------------------------------------------------------------------------


def gf2_independent_basis(columns: Iterable[int]) -> List[int]:
    """Return an independent GF(2) basis for integer bit-vector columns."""
    basis_by_lead: Dict[int, int] = {}
    for c in columns:
        x = c
        while x:
            lead = x.bit_length() - 1
            b = basis_by_lead.get(lead)
            if b is None:
                basis_by_lead[lead] = x
                break
            x ^= b
    return list(basis_by_lead.values())


@lru_cache(maxsize=None)
def chi_diff_distribution(delta: int) -> Tuple[Tuple[int, float], ...]:
    """Exact differential distribution of chi for a fixed input difference.

    For chi_i(x) = x_i + (x_{i+1}+1) x_{i+2}, the output difference is affine
    in x:

        D_i = d_i + d_{i+2} + d_{i+1} d_{i+2}
              + d_{i+1} x_{i+2} + d_{i+2} x_{i+1}.

    Hence the distribution is uniform over one affine subspace.  We enumerate
    the span of the nonzero columns of this linear map.
    """
    delta &= MASK_N
    base = 0
    for i in range(N):
        di = (delta >> i) & 1
        d1 = (delta >> ((i + 1) % N)) & 1
        d2 = (delta >> ((i + 2) % N)) & 1
        if di ^ d2 ^ (d1 & d2):
            base ^= 1 << i

    cols: List[int] = []
    # Variable x_j contributes to output j-2 if d_{j-1}=1, and to output j-1
    # if d_{j+1}=1.
    for j in range(N):
        col = 0
        if (delta >> ((j - 1) % N)) & 1:
            col ^= 1 << ((j - 2) % N)
        if (delta >> ((j + 1) % N)) & 1:
            col ^= 1 << ((j - 1) % N)
        if col:
            cols.append(col)

    basis = gf2_independent_basis(cols)
    outs = [base]
    for b in basis:
        outs += [x ^ b for x in outs]
    prob = math.ldexp(1.0, -len(basis))
    return tuple((x, prob) for x in outs)


@lru_cache(maxsize=None)
def linear_forward_basis_koala(j: int) -> int:
    """Return L(e_j) for Koala's L = theta o pi.

    Because 121^{-1} = 17 modulo 257, input bit j contributes to
    output bits 17*j, 17*j-3 and 17*j-10.
    """
    base = (INV121 * j) % N
    return (
        (1 << base)
        ^ (1 << ((base - 3) % N))
        ^ (1 << ((base - 10) % N))
    )


@lru_cache(maxsize=None)
def linear_forward_koala(mask: int) -> int:
    """Apply Koala's L = theta o pi to a difference mask."""
    out = 0
    x = mask & MASK_N
    while x:
        lb = x & -x
        j = lb.bit_length() - 1
        out ^= linear_forward_basis_koala(j)
        x ^= lb
    return out


def rho_diff_distribution_once(dist: Dict[int, float]) -> Dict[int, float]:
    """Propagate through one shifted Koala step: chi followed by L."""
    out: Dict[int, float] = defaultdict(float)
    for d, p in dist.items():
        for d_chi, q in chi_diff_distribution(d):
            out[linear_forward_koala(d_chi)] += p * q
    return dict(out)

def exact_front_distribution(input_diff: int, front_rounds: int = 2) -> Dict[int, float]:
    """Exact distribution after front_rounds shifted Koala chi-then-L steps."""
    if front_rounds < 0:
        raise ValueError("front_rounds must be nonnegative")
    dist: Dict[int, float] = {input_diff & MASK_N: 1.0}
    for _ in range(front_rounds):
        dist = rho_diff_distribution_once(dist)
    return dist


# ---------------------------------------------------------------------------
# Multiprocessing worker.
# ---------------------------------------------------------------------------


_WORKER_PROP: Optional[ByteOneWiseKoalaDL] = None
_WORKER_TAIL_ROUNDS: Optional[int] = None
_WORKER_TARGET_MASK: Optional[int] = None
_WORKER_DUMP: bool = False


def init_worker(block_size: int, tail_rounds: int, target_mask: int, dump: bool) -> None:
    global _WORKER_PROP, _WORKER_TAIL_ROUNDS, _WORKER_TARGET_MASK, _WORKER_DUMP
    _WORKER_PROP = ByteOneWiseKoalaDL(make_block_sizes(N, block_size))
    _WORKER_TAIL_ROUNDS = int(tail_rounds)
    _WORKER_TARGET_MASK = int(target_mask)
    _WORKER_DUMP = bool(dump)


@dataclass
class BatchResult:
    count: int
    weighted_sum: float
    abs_weighted_sum: float
    prob_sum: float
    max_abs_contribution: float
    max_delta: int
    max_prob: float
    max_corr: float
    rows: Optional[List[Tuple[str, float, float, float, int]]]


def worker_batch(batch: Sequence[Tuple[int, float]]) -> BatchResult:
    if _WORKER_PROP is None or _WORKER_TAIL_ROUNDS is None or _WORKER_TARGET_MASK is None:
        raise RuntimeError("worker has not been initialized")

    vals: List[float] = []
    abs_vals: List[float] = []
    prob_vals: List[float] = []
    rows: Optional[List[Tuple[str, float, float, float, int]]] = [] if _WORKER_DUMP else None
    max_abs = -1.0
    max_delta = 0
    max_prob = 0.0
    max_corr = 0.0

    for delta, prob in batch:
        corr = _WORKER_PROP.tail_correlation(_WORKER_TAIL_ROUNDS, delta, _WORKER_TARGET_MASK)
        contrib = prob * corr
        vals.append(contrib)
        abs_vals.append(abs(contrib))
        prob_vals.append(prob)
        ac = abs(contrib)
        if ac > max_abs:
            max_abs = ac
            max_delta = delta
            max_prob = prob
            max_corr = corr
        if rows is not None:
            rows.append((hex(delta), prob, corr, contrib, delta.bit_count()))

    return BatchResult(
        count=len(batch),
        weighted_sum=math.fsum(vals),
        abs_weighted_sum=math.fsum(abs_vals),
        prob_sum=math.fsum(prob_vals),
        max_abs_contribution=max_abs,
        max_delta=max_delta,
        max_prob=max_prob,
        max_corr=max_corr,
        rows=rows,
    )


# ---------------------------------------------------------------------------
# Driver and reporting.
# ---------------------------------------------------------------------------


def batched(items: Sequence[Tuple[int, float]], batch_size: int) -> Iterator[List[Tuple[int, float]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for i in range(0, len(items), batch_size):
        yield list(items[i : i + batch_size])


def summarize_distribution(dist: Dict[int, float]) -> Dict[str, object]:
    classes = Counter()
    weights = Counter()
    for d, p in dist.items():
        # Probabilities are powers of two for the two-round case from [0].
        lg = round(math.log2(p), 12)
        classes[lg] += 1
        weights[d.bit_count()] += 1
    return {
        "num_differences": len(dist),
        "probability_mass": math.fsum(dist.values()),
        "probability_classes_log2": dict(sorted(classes.items())),
        "weight_distribution": dict(sorted(weights.items())),
    }


def paper_characteristic_mask() -> int:
    return support_to_mask([0, 77, 84, 87, 196, 203, 206, 247, 254])


def run_paper_characteristic(rounds: int, target_mask: int, block_size: int) -> None:
    if rounds < 2:
        raise ValueError("paper-characteristic mode requires rounds >= 2")
    tail_rounds = rounds - 2
    prop = ByteOneWiseKoalaDL(make_block_sizes(N, block_size))
    d2 = paper_characteristic_mask()
    p = 2.0 ** -8
    corr = prop.tail_correlation(tail_rounds, d2, target_mask)
    total = p * corr
    print("Koala paper characteristic mode")
    print(f"  rounds                 : {rounds}")
    print(f"  tail rounds             : {tail_rounds}")
    print(f"  Delta_2                 : {format_support(d2, max_len=40)}")
    print(f"  Pr(prefix)              : {p:.17g} = 2^-8")
    print(f"  tail corr               : {corr:.17g}")
    print(f"  tail log2(abs)          : {log2_abs(corr):.12f}")
    print(f"  combined corr           : {total:.17g}")
    print(f"  combined log2(abs)      : {log2_abs(total):.12f}")


def run_full_hybrid(args: argparse.Namespace) -> None:
    input_supp = parse_support(args.input_diff)
    target_supp = parse_support(args.target)
    input_mask = support_to_mask(input_supp)
    target_mask = support_to_mask(target_supp)

    if args.rounds < args.front_rounds:
        raise ValueError("--rounds must be >= --front-rounds")
    tail_rounds = args.rounds - args.front_rounds

    t0 = time.time()
    full_dist = exact_front_distribution(input_mask, args.front_rounds)
    gen_time = time.time() - t0
    full_summary = summarize_distribution(full_dist)

    dist = full_dist
    if args.min_prob_log2 is not None:
        threshold = 2.0 ** float(args.min_prob_log2)
        dist = {d: p for d, p in dist.items() if p >= threshold}

    items = list(dist.items())
    if args.sort == "prob":
        items.sort(key=lambda x: (-x[1], x[0].bit_count(), x[0]))
    elif args.sort == "weight":
        items.sort(key=lambda x: (x[0].bit_count(), -x[1], x[0]))
    else:
        items.sort(key=lambda x: x[0])

    if args.max_instances is not None:
        items = items[: args.max_instances]

    used_summary = summarize_distribution(dict(items))
    print("Koala-p hybrid DL estimation")
    print(f"  N                       : {N}")
    print(f"  total rounds             : {args.rounds}")
    print(f"  exact front chi/L steps   : {args.front_rounds}")
    print(f"  geometric tail chi layers  : {tail_rounds}")
    print(f"  block size               : {args.block_size}")
    print(f"  input difference         : {input_supp}")
    print(f"  output mask              : {target_supp}")
    print(f"  generated front states   : {full_summary['num_differences']} in {gen_time:.3f}s")
    print(f"  full probability mass    : {full_summary['probability_mass']:.17g}")
    print(f"  instances to evaluate    : {used_summary['num_differences']}")
    print(f"  probability mass used    : {used_summary['probability_mass']:.17g}")
    print(f"  probability classes log2 : {used_summary['probability_classes_log2']}")
    print(f"  Hamming-weight classes   : {used_summary['weight_distribution']}")
    print(f"  workers                  : {args.workers}")
    print(f"  batch size               : {args.batch_size}")
    sys.stdout.flush()

    if args.dump_distribution:
        with open(args.dump_distribution, "w", encoding="utf-8") as f:
            for d, p in items:
                f.write(json.dumps({
                    "delta_hex": hex(d),
                    "weight": d.bit_count(),
                    "probability": p,
                    "probability_log2": math.log2(p),
                    "support": mask_to_support(d),
                }) + "\n")
        print(f"  wrote distribution       : {args.dump_distribution}")

    batches = list(batched(items, args.batch_size))
    partial_sums: List[float] = []
    partial_abs: List[float] = []
    partial_prob: List[float] = []
    done = 0
    best = BatchResult(0, 0.0, 0.0, 0.0, -1.0, 0, 0.0, 0.0, None)
    start = time.time()
    last_report = start

    dump_file = open(args.dump_contribs, "w", encoding="utf-8") if args.dump_contribs else None
    if dump_file is not None:
        dump_file.write(json.dumps({
            "type": "header",
            "rounds": args.rounds,
            "front_rounds": args.front_rounds,
            "tail_rounds": tail_rounds,
            "block_size": args.block_size,
            "input_diff": input_supp,
            "target": target_supp,
        }) + "\n")

    def consume(res: BatchResult) -> None:
        nonlocal done, best, last_report
        done += res.count
        partial_sums.append(res.weighted_sum)
        partial_abs.append(res.abs_weighted_sum)
        partial_prob.append(res.prob_sum)
        if res.max_abs_contribution > best.max_abs_contribution:
            best = res
        if dump_file is not None and res.rows is not None:
            for delta_hex, prob, corr, contrib, wt in res.rows:
                dump_file.write(json.dumps({
                    "delta_hex": delta_hex,
                    "weight": wt,
                    "probability": prob,
                    "probability_log2": math.log2(prob),
                    "tail_correlation": corr,
                    "contribution": contrib,
                    "contribution_log2_abs": log2_abs(contrib),
                }) + "\n")
        now = time.time()
        if now - last_report >= args.progress_every or done == len(items):
            cur = math.fsum(partial_sums)
            rate = done / max(now - start, 1e-9)
            print(
                f"  progress {done:>8}/{len(items)} "
                f"rate={rate:.3f}/s "
                f"partial={cur:.17g} log2abs={log2_abs(cur):.6f}",
                flush=True,
            )
            last_report = now

    if args.workers <= 1:
        init_worker(args.block_size, tail_rounds, target_mask, args.dump_contribs is not None)
        for b in batches:
            consume(worker_batch(b))
    else:
        ctx = mp.get_context(args.start_method)
        with ctx.Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(args.block_size, tail_rounds, target_mask, args.dump_contribs is not None),
            maxtasksperchild=args.maxtasksperchild,
        ) as pool:
            for res in pool.imap_unordered(worker_batch, batches, chunksize=1):
                consume(res)

    if dump_file is not None:
        dump_file.close()

    total = math.fsum(partial_sums)
    total_abs = math.fsum(partial_abs)
    prob_used = math.fsum(partial_prob)
    elapsed = time.time() - start

    result = {
        "rounds": args.rounds,
        "front_rounds": args.front_rounds,
        "tail_rounds": tail_rounds,
        "block_size": args.block_size,
        "input_diff": input_supp,
        "target": target_supp,
        "num_instances": len(items),
        "probability_mass_used": prob_used,
        "estimated_correlation": total,
        "estimated_correlation_log2_abs": log2_abs(total),
        "sum_abs_contributions": total_abs,
        "cancellation_ratio_abs_sum_over_abs_total": (total_abs / abs(total)) if total != 0.0 else math.inf,
        "largest_abs_contribution": best.max_abs_contribution,
        "largest_abs_contribution_log2": log2_abs(best.max_abs_contribution),
        "largest_contribution_delta_hex": hex(best.max_delta),
        "largest_contribution_delta_support": mask_to_support(best.max_delta),
        "largest_contribution_prefix_probability": best.max_prob,
        "largest_contribution_tail_correlation": best.max_corr,
        "elapsed_seconds": elapsed,
        "instances_per_second": len(items) / max(elapsed, 1e-9),
        "front_distribution_summary_full": full_summary,
        "front_distribution_summary_used": used_summary,
    }

    print("result")
    print(f"  instances evaluated      : {len(items)}")
    print(f"  probability mass used    : {prob_used:.17g}")
    print(f"  estimated correlation    : {total:.17g}")
    print(f"  log2(abs)                : {log2_abs(total):.12f}")
    print(f"  sum |contributions|      : {total_abs:.17g}")
    if total != 0.0:
        print(f"  cancellation ratio       : {total_abs / abs(total):.6g}")
    print(f"  largest contribution     : {best.max_abs_contribution:.17g}")
    print(f"  largest contribution log2: {log2_abs(best.max_abs_contribution):.12f}")
    print(f"  largest delta            : {format_support(best.max_delta, max_len=64)}")
    print(f"  largest delta prob       : {best.max_prob:.17g}")
    print(f"  largest delta tail corr  : {best.max_corr:.17g}")
    print(f"  elapsed seconds          : {elapsed:.3f}")
    print(f"  instances/second         : {result['instances_per_second']:.6f}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"  wrote result JSON        : {args.output_json}")


def self_test() -> None:
    print("self-test: exact Koala front from [0] through two chi/L steps")
    one = exact_front_distribution(1 << 0, 1)
    first = support_to_mask([0, 247, 254])
    p1 = one.get(first, 0.0)
    print(f"  first characteristic probability = {p1:.17g}")
    assert abs(p1 - 2.0 ** -2) < 1e-15, p1

    dist = exact_front_distribution(1 << 0, 2)
    summary = summarize_distribution(dist)
    print(json.dumps(summary, indent=2, sort_keys=True))
    assert abs(math.fsum(dist.values()) - 1.0) < 1e-12

    expected = paper_characteristic_mask()
    p2 = dist.get(expected, 0.0)
    print(f"  two-step paper characteristic probability = {p2:.17g}")
    print(f"  log2(probability) = {math.log2(p2) if p2 else -math.inf:.12f}")
    assert abs(p2 - 2.0 ** -8) < 1e-15, p2
    print("  exact front checks passed")

    print("self-test: four-chi-layer geometric tail")
    prop = ByteOneWiseKoalaDL(make_block_sizes(N, 8))
    target = 1 << 232
    corr = prop.tail_correlation(4, expected, target)
    total = (2.0 ** -8) * corr
    print(f"  tail corr  = {corr:.17g}, log2abs={log2_abs(corr):.12f}")
    print(f"  total corr = {total:.17g}, log2abs={log2_abs(total):.12f}")
    print("  paper reference is approximately 2^-21.42")
    print("  self-test passed")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Koala-p exact two-step chi/L hull plus byte-wise one-wise geometric DL tail."
    )
    ap.add_argument("--rounds", type=int, default=6, help="total number of chi layers/rounds in the paper's notation; default 6")
    ap.add_argument("--front-rounds", type=int, default=2, help="number of exact shifted chi-then-L front steps; default 2")
    ap.add_argument("--input-diff", type=str, default="0", help="input difference support, e.g. '0' or '0,64'")
    ap.add_argument("--target", type=str, default="232", help="output mask support; default 232 for the paper six-round distinguisher")
    ap.add_argument("--block-size", type=int, default=8, help="tensor block size; default 8 gives 32 bytes + 1 bit")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="number of worker processes; default os.cpu_count()")
    ap.add_argument("--batch-size", type=int, default=1, help="number of Delta_2 instances per multiprocessing task; default 1")
    ap.add_argument("--start-method", type=str, default="fork" if "fork" in mp.get_all_start_methods() else "spawn", choices=mp.get_all_start_methods())
    ap.add_argument("--maxtasksperchild", type=int, default=None, help="recycle workers after this many batches; default disabled")
    ap.add_argument("--sort", choices=["prob", "weight", "mask"], default="prob", help="order in which Delta_2 instances are evaluated")
    ap.add_argument("--max-instances", type=int, default=None, help="debug: evaluate only the first K front differences after sorting")
    ap.add_argument("--min-prob-log2", type=float, default=None, help="debug/truncation: keep only branches with probability >= 2^{value}")
    ap.add_argument("--progress-every", type=float, default=10.0, help="seconds between progress lines")
    ap.add_argument("--dump-distribution", type=str, default=None, help="optional JSONL file for the exact front distribution")
    ap.add_argument("--dump-contribs", type=str, default=None, help="optional JSONL file for per-Delta_2 contributions")
    ap.add_argument("--output-json", type=str, default=None, help="optional JSON file for final summary")
    ap.add_argument("--paper-characteristic", action="store_true", help="run only the paper's single 2-round characteristic sanity check")
    ap.add_argument("--self-test", action="store_true", help="run built-in checks and exit")
    return ap.parse_args(argv)


def main() -> None:
    """Run the fixed Koala-p configuration defined near the top of this file."""
    if FIXED_RUN_SELF_TEST:
        self_test()
        return

    args = argparse.Namespace(
        rounds=FIXED_ROUNDS,
        front_rounds=FIXED_FRONT_ROUNDS,
        input_diff=FIXED_INPUT_DIFF,
        target=FIXED_TARGET,
        block_size=FIXED_BLOCK_SIZE,
        workers=FIXED_WORKERS,
        batch_size=FIXED_BATCH_SIZE,
        start_method=FIXED_START_METHOD,
        maxtasksperchild=FIXED_MAXTASKS_PER_CHILD,
        sort=FIXED_SORT,
        max_instances=FIXED_MAX_INSTANCES,
        min_prob_log2=FIXED_MIN_PROB_LOG2,
        progress_every=FIXED_PROGRESS_EVERY,
        dump_distribution=FIXED_DUMP_DISTRIBUTION,
        dump_contribs=FIXED_DUMP_CONTRIBS,
        output_json=FIXED_OUTPUT_JSON,
        paper_characteristic=FIXED_RUN_PAPER_CHARACTERISTIC_ONLY,
        self_test=FIXED_RUN_SELF_TEST,
    )

    print("fixed Koala-p parameters")
    print(f"  total rounds            : {FIXED_ROUNDS}")
    print(f"  exact front steps       : {FIXED_FRONT_ROUNDS}")
    print(f"  geometric tail rounds   : {FIXED_ROUNDS - FIXED_FRONT_ROUNDS}")
    print(f"  input difference        : [{FIXED_INPUT_DIFF}]")
    print(f"  output mask             : [{FIXED_TARGET}]")
    print(f"  block size              : {FIXED_BLOCK_SIZE}")
    print(f"  workers                 : {FIXED_WORKERS}")
    print(f"  progress interval       : {FIXED_PROGRESS_EVERY}s")
    print()

    target_mask = support_to_mask(parse_support(FIXED_TARGET))
    if FIXED_RUN_PAPER_CHARACTERISTIC_ONLY:
        run_paper_characteristic(FIXED_ROUNDS, target_mask, FIXED_BLOCK_SIZE)
        return

    run_full_hybrid(args)

if __name__ == "__main__":
    main()

