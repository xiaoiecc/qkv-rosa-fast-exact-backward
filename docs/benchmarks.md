# Benchmarks & Numerical Verification

[简体中文](benchmarks.zh-CN.md) · [Home](../README.md) · [Algorithm](algorithm.md) · [Benchmark Scripts](../benchmarks/README.md)

This report summarizes single-thread CPU timings, coordinate-contraction ratios, and numerical accuracy for the current **single causal interval sweep** engine (`interval_credit.h` + `edit_alphabet.h`).

> **Testbed:** Single pinned Intel Xeon Platinum 8573C thread (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`), 1 warm-up + 5 interleaved timed runs (median reported). Native timings include the NumPy wrapper, buffer allocation, and full backward. Full PyTorch step timings include dense linear projections, causal forward, MSE loss, backward, and SGD update.

---

## 1. Independent Gaussian Inputs ($T = 8192$)

Median backward latency for first-order (`"first"`) and second-order (`"second"`) exact credit across bit widths, together with speedups over the repository's earlier $\widetilde{O}(T^{5/3})$ structural engine:

| Bit width $b$ | 1st-order (ms) | 2nd-order (ms) | Speedup vs. earlier engine (1st / 2nd) |
|---|---:|---:|---:|
| 1 bit | 12.68 | 15.24 | 4.49× / 3.92× |
| 2 bits | 11.73 | 13.48 | 3.53× / 3.21× |
| 4 bits | 9.94 | 10.93 | 3.28× / 3.16× |
| 8 bits | 7.79 | 9.68 | 2.93× / 3.00× |
| 12 bits | 12.79 | 15.41 | 1.62× / 1.89× |
| 16 bits | 9.23 | 11.07 | 1.21× / 1.42× |
| 32 bits | 2.00 | 2.69 | 5.82× / 5.86× |
| 64 bits | 3.75 | 5.24 | 5.77× / 4.96× |

Across 96 independent Gaussian cases ($T \in \{2048, 8192\}$, 8 widths, 3 seeds), the geometric mean speedup is **2.91×** (faster on 96/96 cases). At wider alphabets ($b \ge 24$), causal coordinate contraction eliminates nearly the entire stream:

| Length $T$ | Bit width $b$ | Retained rows after contraction (3 seeds) |
|---:|---:|---|
| 8192 | 8 / 16 | 8192, 8192, 8192 (full stream) |
| 8192 | 24 | 361, 284, 307 |
| 8192 | 32 | 9, 0, 0 |
| 8192 | 64 | 0, 0, 0 |

---

## 2. Full PyTorch Mixer Update Steps

End-to-end `NBitQKVRosa` training step (linear projections + causal forward + MSE loss + backward + SGD):

| Shape $(T, C, b)$ | 1st-order step (ms) | 2nd-order step (ms) | Speedup vs. earlier engine (1st / 2nd) |
|---|---:|---:|---:|
| `T=2048, C=8, b=1` | 25.00 | 28.07 | 3.53× / 3.18× |
| `T=4096, C=32, b=8` | 22.63 | 25.26 | 2.05× / 2.05× |
| `T=4096, C=32, b=16` | 11.51 | 13.24 | 1.25× / 1.50× |
| `T=2048, C=64, b=64` | 6.28 | 6.23 | 1.63× / 1.87× |

Across 24 full PyTorch updates, the geometric mean speedup is **2.01×** (faster on 24/24 cases).

---

## 3. Structured & Stress Inputs ($T = 2048$)

| Input family | 1st-order (ms) | 2nd-order (ms) | Notes |
|---|---:|---:|---|
| `unary` (constant stream, $b=8$) | 0.88 | 1.67 | Hits single-run difference-array path ($O(Tb)$) |
| `near_fanout` (dense 1-bit neighbors, $b=8$) | 1.34 | 2.21 | **11.7×–19.5×** faster than earlier engine |
| `lagged_copy` (long delayed copy, $b=64$) | 12.49 | 13.96 | Exercises long-row deletion and suffix queries |
| `periodic` (short period, $b=4$) | 16.19 | 19.56 | Longer posting lists make periodic strings relatively slower |

---

## 4. Correctness & Numerical Accuracy

- **Physical bit-flip oracle:** [`tests/cpp/interval_credit.cpp`](../tests/cpp/interval_credit.cpp) verifies **74,896 literal single-bit flips** and 1,812 forward rows against an independent causal DP oracle.
- **Cross-implementation parity:** Across 680 paired native cases, forward outputs, joint pair codes, and $V$/embedding gradients were **byte-identical**. Maximum combined Q/K relative L2 discrepancy was **$9.52 \times 10^{-15}$** (standard `float64` reassociation rounding).
- **Memory safety:** Core interval reducer and C training ABI pass ASan and UBSan with zero errors.

---

## 5. Reproducing Measurements

```bash
# Measure coordinate contraction ratios
c++ -O3 -std=c++17 benchmarks/interval_support.cpp -o build/interval-support
./build/interval-support

# Run full correctness and sanitizer checks
python tests/run.py --exhaustive
python tests/sanitize.py --jobs 2 --no-leaks
```

See [`benchmarks/README.md`](../benchmarks/README.md) for paired checkout comparison commands.
