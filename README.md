# QKV-ROSA: subquadratic counterfactual backpropagation

QKV-ROSA is a hard retrieval mixer: each output takes the payload after one causal longest-suffix match. This repository trains it with the finite output change caused by each Q/K/V bit flip, projected on the incoming gradient. One C++17 engine computes these credits in $\widetilde O(T^{7/4})$ time and $\widetilde O(T^{4/3})$ space for fixed bit width, numerical dimension, and joint-pair budget.

The layer fits an ordinary PyTorch model and optimizer:

```python
import torch
from conv_rosa_transformer.rosa_layer import NBitQKVRosa

layer = NBitQKVRosa(channels=128, n_bits=8)
optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
x = torch.randn(2, 32, 128)
target = torch.randn_like(x)

optimizer.zero_grad(set_to_none=True)
loss = (layer(x) - target).square().mean()
loss.backward()
optimizer.step()
```

[从四次方朴素做法开始的中文教程](ALGORITHM.zh-CN.md) · [English derivation](ALGORITHM.md) · [Formal proof](docs/subquadratic-proof.md)

## One engine, three independent choices

```python
layer = NBitQKVRosa(
    channels=128, n_bits=8,
    credit_order="second",      # diagonal trust + selected joint-bit blocks
    randomized=True,            # verified query cache; same outputs as False
    backward_mode="exact",      # complete credit evaluation, no sampling
)

first_order = NBitQKVRosa(128, 8, credit_order="first")

approximate = NBitQKVRosa(
    128, 8, credit_order="second", backward_mode="approximate",
    approximation_rows=32, approximation_channels=0,
)
```

| Choice | Default | Meaning |
|---|---|---|
| `credit_order` | `"second"` | Original first-order finite credit, or a diagonal local model with selected joint-bit cooperation |
| `randomized` | `True` | Bounded full-key cache; collisions cause exact recomputation and do not change arithmetic order |
| `backward_mode` | `"exact"` | All credit contributions, or an explicit residual-row/channel estimator |

Both credit orders use the same structural algorithm: shared periodic deletion, Q heavy/light repair, compact short K queries, anchored long K repair, and batched interval correlations. No input-family dispatcher selects a different matching algorithm. The repository contains one backward engine; there is no backend selector.

**Migration:** remove the old Python `credit_engine` and `repair_samples` keyword arguments. Select the credit rule with `credit_order`; request statistical sampling through `backward_mode="approximate"` and `approximation_rows` / `approximation_channels`. In the C options struct, `repair_budget` is reserved and must be zero.

**Complete evaluation includes every term, but does not prescribe the old floating-point reduction order.** String comparisons, causality, and longest/latest ties are exact. Binary64 grouping and FFT introduce rounding. Switching the verified cache on and off within this engine leaves numerical outputs bitwise identical; explicitly approximate mode adds sampling error. The [numerical contract](docs/algorithm-reference.md) distinguishes these cases.

## Backward speed, memory, and error

Complete component backward, verified cache enabled, B=G=1, CPU binary64, one thread pinned to CPU 4, `trace=True`. Each figure is the median of nine alternating calls. Timing includes validation, allocations, indexes, trust/joint work and all gradients; forward, loss and optimizer are outside the timer. Rows/s describes this backward component, not whole-model throughput.

| Input | T | D | First ms | First rows/s | Diagonal + joint ms | Diagonal + joint rows/s | Buffer estimate MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 512 | 1 | 1.430 | 358,153 | 1.608 | 318,447 | 0.34 |
| random | 1,024 | 1 | 3.659 | 279,859 | 3.851 | 265,930 | 0.79 |
| random | 1,024 | 8 | 1.572 | 651,375 | 2.207 | 464,025 | 1.47 |
| random | 1,024 | 16 | 0.858 | 1,193,486 | 1.312 | 780,563 | 0.26 |
| random | 2,048 | 1 | 8.223 | 249,071 | 8.541 | 239,775 | 2.24 |
| random | 2,048 | 8 | 4.028 | 508,494 | 4.754 | 430,821 | 2.98 |
| random | 4,096 | 1 | 16.630 | 246,307 | 18.374 | 222,925 | 3.19 |
| random | 4,096 | 8 | 7.469 | 548,418 | 10.496 | 390,260 | 6.10 |
| random | 4,096 | 16 | 3.896 | 1,051,202 | 5.832 | 702,334 | 1.66 |
| random | 4,096 | 64 | 8.575 | 477,670 | 10.204 | 401,399 | 0.84 |
| random | 8,192 | 8 | 19.003 | 431,080 | 25.512 | 321,105 | 12.56 |
| random | 8,192 | 16 | 9.954 | 822,954 | 13.617 | 601,608 | 4.18 |
| random | 8,192 | 64 | 17.197 | 476,372 | 19.855 | 412,584 | 1.69 |
| unary | 1,024 | 8 | 0.784 | 1,305,586 | 0.891 | 1,148,663 | 0.52 |
| unary | 4,096 | 8 | 2.589 | 1,582,345 | 3.052 | 1,342,238 | 2.05 |
| periodic | 1,024 | 4 | 3.933 | 260,373 | 4.200 | 243,803 | 1.20 |
| periodic | 4,096 | 4 | 23.700 | 172,827 | 24.660 | 166,098 | 5.16 |
| shadow | 1,024 | 8 | 4.393 | 233,118 | 4.490 | 228,057 | 1.22 |
| shadow | 4,096 | 8 | 20.282 | 201,954 | 24.685 | 165,930 | 5.23 |
| shadow | 8,192 | 8 | 54.431 | 150,502 | 56.422 | 145,191 | 10.82 |
| skyline | 1,632 | 8 | 13.370 | 122,066 | 13.948 | 117,004 | 2.18 |
| skyline | 6,336 | 8 | 97.403 | 65,050 | 101.853 | 62,207 | 9.47 |

Workspace is a conservative **buffer capacity estimate**, not measured RSS. The [full report](docs/subquadratic-results.md) / [中文实测](docs/subquadratic-results.zh-CN.md) contains all samples, per-mode errors, training losses and exact scope. Raw final-source measurements: [four modes](benchmarks/results/optimized_engine.json), [validation](benchmarks/results/optimized_validation.json), [ASan/UBSan](benchmarks/results/optimized_sanitizers.json), [training](benchmarks/results/optimized_toy.json).

### Comparison with earlier implementations

Against this round's starting implementation, all 20 paired cases improve by **2.23–14.31×**. Against the older historical engine, **28/42 cases with T≥1024 are faster and 14/42 are slower**. The largest measured improvement is 10.99×; the worst regression is random T=2048, D=1, first: 7.386→9.906ms (1.34× the old latency). These are interleaved comparisons on the same inputs: [historical 44-case receipt](benchmarks/results/optimized_comparison.json), [round-baseline 20-case receipt](benchmarks/results/optimized_previous_comparison.json). The full report lists every ratio and regression. Length ≥1024 alone does not guarantee a win.

The complete randomized and deterministic modes are bytewise identical in all 44 measured cases. Approximation remains opt-in: its budget controls work, not an error tolerance, and some cases are slower or have large gradient error. The 48 training runs retain ordinary one-forward/one-backward integration; complete/cache parameter hashes match at every update. Tiny training tasks do not establish a universal quality advantage for diagonal/joint credit or sampling.

## Documentation

| Document | Contents |
|---|---|
| [Algorithm](ALGORITHM.md) / [中文](ALGORITHM.zh-CN.md) | Derivation from the $O(bT^4)$ all-flip implementation to the current engine |
| [Reference map](docs/algorithm-reference.md) / [中文](docs/algorithm-reference.zh-CN.md) | Invariants, boundaries, source files, tests, and proof references |
| [Native training](docs/native-training.md) / [中文](docs/native-training.zh-CN.md) | First order, diagonal trust, joint bits, and ordinary autograd integration |
| [Exact randomization](docs/exact-randomized.md) / [中文](docs/exact-randomized.zh-CN.md) | Full-key cache, collision handling, adaptive probing, and reproducibility |
| [Approximate backward](docs/randomized-backward.md) / [中文](docs/randomized-backward.zh-CN.md) | Complete residual rows, zero certificates, fixed-size channel sampling, and error |
| [Current results](docs/subquadratic-results.md) / [中文](docs/subquadratic-results.zh-CN.md) | Timings, memory, approximation error, and measured historical comparisons |
| [Forward](docs/forward.md) / [中文](docs/forward.zh-CN.md) | Causal suffix-automaton forward |
| [C API](docs/native-api.md) · [Tests](tests/README.md) · [Benchmarks](benchmarks/README.md) | Interfaces and reproduction |

## Training and numerical scope

The default second-order option is a chosen local diagonal model, not the true Hessian of an arbitrary downstream network. Selected joint-bit corners are evaluated inside the component; the outer model and loss still run once. V and embedding gradients retain their complete baseline-route evaluation. Details and the joint budget are in the [training guide](docs/native-training.md).

The trust coefficients and joint proposer depend on the incoming gradient. Combine loss terms before a single backward when you want their combined rule; separate backwards need not produce the same update. Hard-credit double autograd is unsupported. These are properties of the training surrogate, separate from the speed of its credit engine.

This is a CPU implementation. CUDA tensors require transfers, and the reported environment is Linux/GCC. No fast-math, unsafe aliasing, or probabilistic route equality is needed.

## Install and verify

```bash
python -m pip install -r requirements.txt
python tests/run.py --exhaustive
```

The C++17 core compiles on first use from a source-hashed cache with `-ffp-contract=off`. A framework-free example uses the same native implementation:

```bash
c++ -O3 -std=c++17 -ffp-contract=off examples/native_training.cpp \
    conv_rosa_transformer/csrc/unified_core.cpp -o /tmp/rosa-native
/tmp/rosa-native
```

Packed symbols support `1 ≤ D ≤ 64`; empty streams are supported. The detailed shape and allocation limits are specified by [training.h](conv_rosa_transformer/csrc/training.h) and the [C API guide](docs/native-api.md).

## Attribution and license

The QKV-ROSA forward architecture comes from **Bo Peng (BlinkDL)** and the [RWKV-LM RWKV-v8 examples](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8), including `samx_qkv_slow` / `ROSA_QKV_B_1bit` in [251024_rosaQKV_run.py](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v8/251024_rosaQKV_run.py). This independent implementation supports n-bit hard symbols, a learned empty payload, and complete single-bit counterfactual credits.

Copyright 2026 xiaoiecc. Licensed under the [MIT License](LICENSE).
