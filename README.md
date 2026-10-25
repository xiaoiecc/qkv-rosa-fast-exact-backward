# Fast Exact Backpropagation for QKV-ROSA

One CPU algorithm for the **exact single-bit counterfactual VJP of hard
QKV-ROSA**. It computes every Q/K/V bit's projected output change under a
single flip, with full untruncated matching and the latest-endpoint tie-break.
The hard routing is discrete; these credits are finite differences, not
ordinary derivatives of threshold quantization.

**Learn the algorithm:** [English derivation](ALGORITHM.md) ·
[中文推导](ALGORITHM.zh-CN.md). Both start from the `O(DT⁴)` bit-flip brute
force and derive the shared work, invariants, compressed representations, and
remaining complexity. The [readable Python entry point](hard_qkv_rosa_explained.py)
exports the independent definition and the same native runtime used by the
PyTorch layer.

## Current CPU backward performance

The measured operation is one complete public `exact_stream_bit_credits`
call for a stream of length T and width D. It returns Q/K/V credits and the
baseline match lengths/routes. Input checks, float32-to-float64 conversion,
per-call index construction, matching, numerical contraction, allocations,
and float32 outputs are all inside the timer. Compilation and warmup are
outside it. There is no cross-call reuse of a data index.

| Input | T | D | Median backward call | Tokens/s | Returned credit tensors |
|---|---:|---:|---:|---:|---:|
| Random 8-bit symbols | 8,192 | 8 | 26.678 ms | 307,074 | 768 KiB |
| Random 16-bit symbols | 8,192 | 16 | 13.756 ms | 595,525 | 1.5 MiB |
| Random binary symbols | 65,536 | 1 | 808.774 ms | 81,031 | 768 KiB |
| Periodic Q/K | 4,096 | 4 | 46.693 ms | 87,722 | 192 KiB |
| All-same Q/K, 2 payloads | 8,192 | 1 | 3.997 ms | 2,049,535 | 96 KiB |
| Shadow, 2 payloads | 8,192 | 1 | 179.192 ms | 45,716 | 96 KiB |
| Shadow, 1 payload | 2,048 | 64 | 10.029 ms | 204,206 | 1.5 MiB |
| Shadow, 2 payloads | 4,096 | 64 | 111.249 ms | 36,818 | 3 MiB |
| Shadow, random payloads | 2,048 | 64 | 75.321 ms | 27,190 | 1.5 MiB |

Measured on an Intel Xeon Platinum 8573C with GCC 13.3, NumPy 2.3.5,
and PyTorch 2.14.1+cpu (`-O3 -std=c++17 -ffp-contract=off`).
These are medians of seven calls on one pinned CPU core, with one PyTorch
thread. `tokens/s` is T divided by the full stream-call duration, not model
training throughput. The source hashes, machine/compiler metadata, every
sample, and all 21 current-algorithm cases are in the
[benchmark report](benchmarks/README.md),
[JSON](benchmarks/results/current.json), and
[CSV](benchmarks/results/current.csv). Dense projections and full-layer
forward/backward are not included in these stream measurements.

The input names belong to the benchmark generator. All of them execute the
same runtime. `all_same` means all-zero Q/K; `shadow` means all-zero Q with K's
first symbol set to one. Payload diversity is listed separately because it
affects score reuse.

### Resources and scaling

The stream core runs on CPU and uses binary64 arithmetic internally. Returned
credit bytes in the table count the three actual float32 output tensors;
they do not measure peak working memory. The implementation also needs:

- `O(TD + UD + T log(T+1) + P_exp)` working storage, where U is the number
  of distinct V payloads and `P_exp` counts extra floating residual components;
- at most `4TD` doubles for the history score cache, allocated on demand;
- at most `2(T+1)D` doubles for moving-owner gradient buckets, also on demand;
- compact deletion records: 12 bytes per A record and 16 bytes per K segment.

There is no stored `T × T` alignment matrix. Some inputs still perform
quadratic work: unique continuation intervals can cover quadratically many
numeric terms, and periodic inputs can emit many deletion records. The
[complexity derivation](ALGORITHM.md#11-put-the-pieces-together-and-count-the-remaining-work)
accounts for those terms and for variable-size residual arithmetic. Source
level memory bounds and returned tensor sizes are not a substitute for a
peak-RSS measurement.

### Comparison with the original upstream implementation

The smaller paired comparison uses original repository commit
[`32e90e2`](https://github.com/xiaoiecc/qkv-rosa-fast-exact-backward/commit/32e90e272929942a764d696b639409b783c61bb1),
with identical inputs and alternating timed order. Its sources are obtained
from an external checkout for benchmarking and are not bundled into the
runtime.

| Input | T / D | Original upstream | Current | Speedup |
|---|---:|---:|---:|---:|
| Random 16-bit | 4,096 / 16 | 61.020 ms | 6.972 ms | 8.75× |
| Random 8-bit | 1,024 / 8 | 57.685 ms | 2.143 ms | 26.92× |
| Shadow, 2 payloads | 1,024 / 1 | 3749.891 ms | 4.098 ms | 915.00× |

The [complete comparison](benchmarks/README.md#original-upstream-comparison)
includes the original compiler compatibility patch, numerical tolerances,
all paired cases, and raw measurements. Ratios apply to those cases on that
machine; they are not a claim that every input is faster by the same factor.

## What is being shared

For a Q/K flip, a match either avoids the edited position or repairs the one
mismatch at that position. The answer is the lexicographic maximum of a
**common deletion route** and the **best repair route**. The algorithm builds
that decomposition once, shares all D bits' deletion work, and reuses Q repair
discovery for the corresponding K owners.

Consecutive deletion winners become formula intervals. A causal occurrence
list reports only endpoints already permitted by the forward. A complete
edited suffix with one static occurrence has a fixed affine continuation;
its K owner and heap priority remain constant for its lifetime. Payload
scores use a bounded cache, while residual-preserving sums keep interval
cancellation until the final owner merge. V credits are a direct route
scatter. The [illustrated tutorial](ALGORITHM.md) proves each step and shows
where tempting shortcuts fail.

## Quick start

The NumPy stream API needs Python, NumPy, and a C++17 compiler. The layer and
demo additionally need PyTorch. On Linux/macOS/WSL, put `c++` on PATH or set
`CXX`. The native core compiles automatically on first use and caches the
result by source and compiler options. It does not require PyTorch C++
headers, Ninja, or a CUDA toolkit.

```bash
python -m pip install numpy torch
python hard_qkv_rosa_explained.py
```

Use the layer:

```python
import torch
from hard_qkv_rosa_explained import NBitQKVRosa

layer = NBitQKVRosa(channels=64, n_bits=8)
x = torch.randn(2, 128, 64)          # [batch, time, channels]
y = layer(x)
y.square().mean().backward()
```

Q/K/V/O projection parameter names and embedding parameter names are retained.
Each channel group is one D-bit stream. The `bernoulli` and `identity`
credit-to-logit mappings remain explicit layer options; see the
[derivation](ALGORITHM.md#2-the-expensive-definition-and-a-map-of-its-repeated-work).
The layer can transfer routing work from its tensor device to the CPU core;
this is not a GPU matching kernel, and transfer costs are outside the stream
benchmark above.

Or inspect the five-symbol example without constructing a model:

```python
import numpy as np
from conv_rosa_transformer.rosa_core import credits_numpy

q = k = [0, 1, 0, 1, 0]
v = np.array([[0], [0], [0], [0], [1]], dtype=np.uint8)
g = np.array([[0.], [0.], [0.], [0.], [3.]])
cq, ck, cv, lengths, routes, work = credits_numpy(
    q, k, v, g, np.array([0.]), np.array([2.]), D=1
)
assert (lengths[4], routes[4]) == (3, 2)
assert cq[4, 0] == 12.0
```

The declared index domain is `1 <= D <= 64`, `T < 2**28`, and
`T*D <= 2**31-1`; actual sizes must fit available memory. Q/K words must be
nonnegative and fit D bits. V contains only 0/1. Gradients and embeddings
must be real and finite. Empty streams are supported. The full test suite
uses GCC/Clang-compatible compiler flags and is exercised on Linux/WSL;
native Windows build support is provided for an x64 Visual Studio developer
shell but was not measured in the published CPU run.

## Interfaces and repository map

| Entry | Purpose |
|---|---|
| [`hard_qkv_rosa_explained.py`](hard_qkv_rosa_explained.py) | Readable entry point, module aliases, independent oracle exports |
| [`rosa_core.py`](conv_rosa_transformer/rosa_core.py) | `exact_stream_bit_credits` returns five results; `credits_numpy` adds work counters |
| [`rosa_reference.py`](conv_rosa_transformer/rosa_reference.py) | `forward_naive` and `backward_bruteforce`, the independent mathematical oracle |
| [`csrc/`](conv_rosa_transformer/csrc/) | One C++ core and its four shared headers |
| [`rosa_layer.py`](conv_rosa_transformer/rosa_layer.py) | PyTorch autograd integration and compatible layer names |
| [`Conv∥ROSA package`](conv_rosa_transformer/README.md) | Parallel convolution/retrieval model and multiplication demo |
| [`tests/`](tests/README.md) | Independent bit-flip, structure, numerical, and layer checks |
| [`benchmarks/`](benchmarks/README.md) | Current absolute performance, work counts, and upstream comparison |

The five-result API returns `(cq, ck, cv, lengths, routes)`. With a real
floating PyTorch `grad_y`, the credit tensors are CPU tensors of its dtype;
NumPy callers receive float64 arrays. The six-result NumPy interface returns
the same credit arrays plus a counter dictionary. Lengths/routes are Python
lists in both interfaces.

## Correctness and numerical scope

```bash
python tests/run.py
python tests/run.py --exhaustive
```

The independent checks include complete single-bit hard-forward reruns,
binary exhaustive inputs, wide words including bit 63, causal endpoint and
deletion-route oracles, constant-output zero-credit identities, saved
cancellation counterexamples, exact dyadic expansion checks, and full layer
gradient comparisons. The [validation report](benchmarks/results/validation.json)
records the results and the source hashes used for measurement.

Route selection, owner mapping, and bit semantics are exact. Numerical
identities are exact over the reals; implementation arithmetic still rounds.
Expansion preserves generated finite binary64 events under nearest-even
rounding, gradual underflow, and no internal overflow. It cannot recover
information already lost in a row dot product or another ordinary floating
operation. Accordingly, integer cases require equality and floating cases
use stated tolerances; the implementation does not promise whole-VJP bitwise
agreement with a different reduction order. See
[the numerical derivation](ALGORITHM.md#10-a-real-number-identity-is-not-yet-a-safe-floating-point-program).

## Architecture attribution and license

The QKV-ROSA forward architecture comes from **Bo Peng (BlinkDL)** and the
[RWKV-LM RWKV-v8 examples](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8),
including `samx_qkv_slow` / `ROSA_QKV_B_1bit` in
[`251024_rosaQKV_run.py`](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v8/251024_rosaQKV_run.py).
This repository supports n-bit hard symbols and a separate learned empty
payload, and contributes the exact single-bit counterfactual backward.
It is an independent implementation, not an official RWKV distribution.

Copyright 2026 xiaoiecc. Licensed under the [MIT License](LICENSE).
