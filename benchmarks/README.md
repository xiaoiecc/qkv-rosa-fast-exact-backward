# CPU backward measurements

The main results measure this repository's current algorithm on one stream
and one D-bit group. The measured operation is the public five-result
`exact_stream_bit_credits` call: Q/K/V credits plus baseline lengths and
routes. It includes input validation, dtype conversion, all per-call indices,
baseline matching, numeric contraction, output allocation, and conversion
back to the input gradient dtype.

Compilation, imports, input generation, warmup, checks, and report writing
are outside the timer. Dense projections, the autograd layer, and complete
training are also outside this measurement. `tokens/s` means T divided by
the median call duration for this single stream; it is not a Transformer
training-throughput measurement. Returned credit tensor bytes are the sum
of the three actual output tensor sizes, not peak working memory or RSS.

## Reproduce

From the repository root:

```bash
python tests/run.py --exhaustive
python benchmarks/backward.py --repeats 7 --cpu 0 --output build/current.json
```

Choose an available logical CPU for `--cpu`, or omit the affinity option
where unsupported. Stop other CPU-heavy work before measuring. The script
uses one PyTorch thread, one warmup, and seven timed repetitions by default.
JSON contains every raw sample, min/max, input digest, seed, exact source
hashes, work counters, and explicit numerical tolerances. CSV contains the
same main timing rows. `--smoke` runs three tiny setup checks; those timings
should not be compared with the published full matrix.

The case generator is deterministic. It tests random binary, 8-bit, and
16-bit symbols; a four-symbol periodic Q/K pair with one K mutation;
all-zero Q/K; and `shadow` (Q all zero, K initially one and otherwise zero).
Scalar two-class payloads alternate. Wide payload classes are assigned
independently at each position, and unrestricted payloads are random bit
rows. These labels exist only in the benchmark generator; the production
runtime uses one implementation for all of them.

## Current implementation

The checked-in [JSON](results/current.json) and [CSV](results/current.csv)
record the complete matrix. The table below is generated from their median
durations after the source hashes have been checked against the validation
report.

Measured on 2026-10-18 with **INTEL(R) XEON(R) PLATINUM 8573C**, logical CPU 0, Linux x86-64, one PyTorch thread, one warmup, and seven timed calls per row. Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.1+cpu; c++ (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0. The native core uses `-O3 -DNDEBUG -std=c++17 -ffp-contract=off`.

| Input | T | D | Observed payload classes | Median (ms) | Tokens/s | Credit outputs (KiB) |
|---|---:|---:|---:|---:|---:|---:|
| random_binary | 1,024 | 1 | 2 | 3.810 | 268,750 | 12 |
| random_binary | 4,096 | 1 | 2 | 20.432 | 200,465 | 48 |
| random_binary | 8,192 | 1 | 2 | 50.872 | 161,031 | 96 |
| random8 | 1,024 | 8 | 253 | 2.027 | 505,277 | 96 |
| random8 | 4,096 | 8 | 256 | 11.053 | 370,579 | 384 |
| random8 | 8,192 | 8 | 256 | 26.678 | 307,074 | 768 |
| periodic | 1,024 | 4 | 16 | 3.859 | 265,384 | 48 |
| periodic | 4,096 | 4 | 16 | 46.693 | 87,722 | 192 |
| periodic | 8,192 | 4 | 16 | 172.784 | 47,412 | 384 |
| shadow | 1,024 | 1 | 2 | 3.774 | 271,304 | 12 |
| shadow | 4,096 | 1 | 2 | 46.436 | 88,207 | 48 |
| shadow | 8,192 | 1 | 2 | 179.192 | 45,716 | 96 |
| all_same | 1,024 | 1 | 2 | 0.598 | 1,712,057 | 12 |
| all_same | 4,096 | 1 | 2 | 2.004 | 2,043,919 | 48 |
| all_same | 8,192 | 1 | 2 | 3.997 | 2,049,535 | 96 |
| random_binary | 16,384 | 1 | 2 | 121.221 | 135,159 | 192 |
| random_binary | 65,536 | 1 | 2 | 808.774 | 81,031 | 768 |
| random16 | 8,192 | 16 | 7,713 | 13.756 | 595,525 | 1,536 |
| shadow | 2,048 | 64 | 1 | 10.029 | 204,206 | 1,536 |
| shadow | 4,096 | 64 | 2 | 111.249 | 36,818 | 3,072 |
| shadow | 2,048 | 64 | 2,048 | 75.321 | 27,190 | 1,536 |

Credit-output sizes exclude returned route lists and all temporary/native working storage. This run does not measure peak allocation or RSS.

## Measured work

Work counters describe what was actually visited or materialized, rather
than treating a reduction in a single counter as the same reduction in
whole-call latency. In particular, singleton continuations can remove
ordinary SAM state advances while still paying for many numeric interval
visits. Those visits can be quadratic. Periodic inputs can still materialize
many deletion records. No universal subquadratic bound or universal speedup
is inferred from this workload matrix.

| Input | T | D | Stored A records | Ordinary Q advances | Numeric continuation visits |
|---|---:|---:|---:|---:|---:|
| periodic | 8,192 | 4 | 6,289,408 | 16,378 | 2,094,081 |
| shadow | 8,192 | 1 | 16,379 | 0 | 33,550,335 |
| all_same | 8,192 | 1 | 16,381 | 0 | 0 |
| random_binary | 65,536 | 1 | 66,192 | 940,843 | 35,643 |
| shadow | 4,096 | 64 | 8,187 | 0 | 8,386,559 |

## Original upstream comparison

This smaller comparison uses the original repository revision
`32e90e272929942a764d696b639409b783c61bb1`. It does not use an intermediate
development implementation. Every paired call receives identical float32
CPU tensors and packed Q/K values. Timed order alternates between the two
implementations. Float64 checks use the same underlying float32 values and
require identical routes; float32 reduction differences are published
separately and are not described as bitwise equality.

Obtain an external checkout of that revision and run:

```bash
git clone https://github.com/xiaoiecc/qkv-rosa-fast-exact-backward.git /tmp/qkv-rosa-upstream
python benchmarks/backward.py --baseline-dir /tmp/qkv-rosa-upstream \
  --baseline-revision 32e90e272929942a764d696b639409b783c61bb1 \
  --repeats 7 --cpu 0 --output build/upstream.json
```

The setup helper reads tracked sources from that git revision and compiles
them in a temporary cache outside this working tree. The historical runtime
is never bundled with, or selected by, the current library. The historical
build currently supports Linux/WSL and requires PyTorch's C++ headers. It
uses GCC/Clang-compatible optimization flags and adds the single standard
`template` keyword needed for the original dependent `item<scalar_t>()`
expression to compile under GCC. Original and compiled source hashes and
compiler flags are both recorded. No historical arithmetic or backend
selection rule is changed.

| Input | T | D | Original (ms) | Current (ms) | Original/current |
|---|---:|---:|---:|---:|---:|
| random_binary | 256 | 1 | 18.747 | 0.944 | 19.850× |
| random_binary | 1,024 | 1 | 111.258 | 4.050 | 27.469× |
| random8 | 256 | 8 | 11.388 | 0.657 | 17.332× |
| random8 | 1,024 | 8 | 57.685 | 2.143 | 26.918× |
| periodic | 256 | 4 | 36.003 | 0.669 | 53.794× |
| periodic | 1,024 | 4 | 548.974 | 4.197 | 130.795× |
| shadow | 256 | 1 | 76.816 | 0.763 | 100.678× |
| shadow | 1,024 | 1 | 3749.891 | 4.098 | 915.002× |
| all_same | 256 | 1 | 13.701 | 0.403 | 33.961× |
| all_same | 1,024 | 1 | 205.854 | 0.932 | 220.922× |
| random16 | 4,096 | 16 | 61.020 | 6.972 | 8.752× |

All 11 paired median ratios exceed one in this matrix. Maximum absolute differences were 4.68958e-13 in the float64 comparison and 3.24249e-05 in the public float32 comparison. Every checked route was identical; all documented tolerances passed. The paired current timings are from this separate alternating run and can differ from the current-only table above.

All measured rows and all regressions are retained in
[upstream.json](results/upstream.json) and [upstream.csv](results/upstream.csv).
Ratios describe these particular cases and this machine, not a claim of
globally fastest behavior.

## Correctness of the measured sources

The full [validation report](results/validation.json) records the production
source hashes, independent all-bit hard-forward checks, integer equality,
floating tolerances, numerical cancellation regressions, exact dyadic
expansion checks, C++ structural oracles, and PyTorch layer tests. The
implementation hashes in both timing reports are checked against this
report before publishing. See [the test guide](../tests/README.md) for the
scope and limitations of each gate.

Final exhaustive run: **27 test methods, 1 CUDA-device skip, PASS** (16.78 seconds on the recorded machine). The current environment has no CUDA device, so GPU transfer behavior is not claimed as measured here.

- 5,461 binary integer inputs require exact all-bit equality; 160 floating, 25 wide-bit, and 16 longer structured inputs use independent direct hard forwards. Maximum ordinary floating error was 1.06581e-14.
- 542 constant-forward fixtures require exact zero Q/K credits; 24 snapshot/corridor and 9 saved nonconstant target witnesses require their independent exact answers.
- 4,052 Fraction cases check exact binary64-event representation and final rounding. Deliberately extreme inputs reached 61 components; ordinary million-add diagnostics had zero spills.
- Structural oracles check 5,782,250 deletion routes, 7,205,159 deletion intervals, 7,858,789 causal memberships, and 1,267,260 Q plus 1,108,192 K repair routes.
- Layer tests cover batching/grouping, noncontiguous inputs, empty shapes, bfloat16, unsigned bit 63, checkpoint configuration, and training smoke behavior. Input validation rejects complex gradients/embeddings and integer Torch public gradients before lossy conversion.

These verification layers overlap; their counts are not a sum of independent whole-core inputs.
