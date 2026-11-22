# QKV-ROSA exact counterfactual backpropagation

QKV-ROSA retrieves the payload after the latest causal longest-suffix match. The default C++17 backward evaluates the finite change from every Q/K/V bit flip. One interval algorithm computes Q/K credit for all packed widths from 1 to 64 bits. It contracts inactive time gaps, accumulates one-symbol contributions by class and queries longer suffixes through an exact index. For independent uniform keys and an independent query stream, the expected time is $\widetilde O(T)$ at fixed width and joint budget. The general worst-case time bound is $\widetilde O(T^2)$ at fixed width and joint budget.

[Algorithm and complexity](docs/causal-interval-credit.md) · [Random QKV, full training steps and regressions](docs/causal-interval-results.md)

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
    randomized=True,            # compatible deterministic exact result
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
| `randomized` | `True` | Compatibility option for the complete engine and verified lookup caches in the sampled estimator |
| `backward_mode` | `"exact"` | All credit contributions, or an explicit residual-row/channel estimator |

Both credit orders use the same exact interval sweep. Explicit approximation samples residual rows or numerical channels. Joint-corner evaluation replays the causal forward on the two edited words.

**Migration:** remove the old Python `credit_engine` and `repair_samples` keyword arguments. Select the credit rule with `credit_order`; request statistical sampling through `backward_mode="approximate"` and `approximation_rows` / `approximation_channels`. In the C options struct, `repair_budget` is reserved and must be zero.

**Complete evaluation includes every term, but does not prescribe the old floating-point reduction order.** String comparisons, causality, and longest/latest ties are exact. Binary64 grouping and FFT introduce rounding. Switching the verified cache on and off within this engine leaves numerical outputs bitwise identical; explicitly approximate mode adds sampling error. The [numerical contract](docs/algorithm-reference.md) distinguishes these cases.

## Backward speed and memory

Across 96 independent Gaussian backward cases, the geometric mean speedup was **2.91× over merged main** and **0.82× against the former PR #12**. Across 24 full PyTorch mixer updates it was **2.01× over main** and **0.85× against #12**. The [paired report](docs/causal-interval-results.md) covers both references. It includes independent QKV, skewed bits, repeated inputs and complete CPU PyTorch optimizer steps. The raw samples include memory estimates and numerical differences for every native comparison.

## Documentation

| Document | Contents |
|---|---|
| [Algorithm](ALGORITHM.md) / [中文](ALGORITHM.zh-CN.md) | Finite-credit derivation and the earlier subquadratic construction |
| [Reference map](docs/algorithm-reference.md) / [中文](docs/algorithm-reference.zh-CN.md) | Invariants, boundaries, source files, tests, and proof references |
| [Native training](docs/native-training.md) / [中文](docs/native-training.zh-CN.md) | First order, diagonal trust, joint bits, and ordinary autograd integration |
| [Exact randomization](docs/exact-randomized.md) / [中文](docs/exact-randomized.zh-CN.md) | Full-key cache, collision handling, adaptive probing, and reproducibility |
| [Approximate backward](docs/randomized-backward.md) / [中文](docs/randomized-backward.zh-CN.md) | Complete residual rows, zero certificates, fixed-size channel sampling, and error |
| [Earlier structural-engine results](docs/subquadratic-results.md) / [中文](docs/subquadratic-results.zh-CN.md) | Timings, memory, approximation error, and measured historical comparisons |
| [Forward](docs/forward.md) / [中文](docs/forward.zh-CN.md) | Suffix-automaton reference and causal routing background |
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
