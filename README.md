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

The [current measured results](docs/subquadratic-results.md) / [中文实测](docs/subquadratic-results.zh-CN.md) report complete backward costs, deterministic versus randomized caching, approximate-mode error, and tracked workspace. The main numbers measure this algorithm directly. Historical commits are external measured baselines, not installed alternative backends.

Complete backward, verified cache enabled, one stream / one bit group. CPU binary64; median of seven calls. Includes indexes, allocations, trust/joint work and all gradients; excludes the model forward, projections, loss and optimizer.

| Input | T | D | First order (ms) | Diagonal + joint (ms) |
|---|---:|---:|---:|---:|
| random | 2,048 | 8 | 51.695 | 40.978 |
| random | 8,192 | 16 | 52.456 | 69.421 |
| random | 4,096 | 64 | 19.990 | 22.553 |
| unary | 4,096 | 8 | 35.362 | 32.817 |
| periodic | 4,096 | 4 | 129.865 | 126.735 |
| shadow | 4,096 | 8 | 84.729 | 87.392 |
| shadow | 8,192 | 8 | 186.739 | 209.366 |
| skyline | 1,632 | 8 | 72.110 | 78.137 |
| skyline | 6,336 | 8 | 445.587 | 439.758 |

The [full report](docs/subquadratic-results.md) / [中文](docs/subquadratic-results.zh-CN.md) includes workspace estimates, all raw samples, approximation errors and training losses. A small historical comparison: the T=8192,D=8 long-boundary second-order case was 576.753 ms in the previous engine and is 209.366 ms here (2.75×); random T=2048,D=8 was 4.962 ms and is 40.978 ms here. These are separate recorded runs. A better asymptotic bound does not guarantee a finite-length win, including for T≥1024.

The exponent alone is not a speed measurement. Suffix indexes, weighted dominance, and numerical batching have substantial finite-size costs, and some workloads remain slower than their historical baseline. Approximation budgets are work budgets rather than guaranteed error tolerances. Measure Q and K error separately, downstream parameter/input-gradient error, and the task's training and validation losses.

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
