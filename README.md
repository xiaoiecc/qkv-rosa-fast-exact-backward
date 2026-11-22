# QKV-ROSA Fast Exact Backward

[简体中文](README.zh-CN.md) · [Algorithm](docs/algorithm.md) · [Training Guide](docs/training.md) · [C/C++ API](docs/c-api.md) · [Benchmarks](docs/benchmarks.md)

A fast, exact C++17 counterfactual backpropagation engine for hard $n$-bit **QKV-ROSA** (Rapid Online Suffix Automaton), with plug-and-play **PyTorch**, **NumPy**, and standalone **C/C++** interfaces.

## Why this repository exists

In hard QKV-ROSA, each query step $t$ searches strictly earlier keys ($r < t$) for the **longest matching suffix** (breaking ties by the latest position $r$), and outputs the value stored right after that match ($V_{r+1}$), or a learned fallback embedding $e_0$ if no match exists.

Because routing is discrete and non-differentiable, standard autograd cannot pass gradients through `Q` and `K`. Instead, we compute the **exact one-bit counterfactual credit**:

> *If we flip a single bit in $Q$, $K$, or $V$ and re-run the exact hard routing, how much does the output change when projected onto the upstream gradient $g = \partial L / \partial Y$?*

A literal brute-force implementation flips every bit and replays the forward pass, costing $O(b\,T^4)$ time for sequence length $T$ and bit width $b$. This library replaces that with a **single causal interval sweep** in C++17:

- **Exact routing and complete credit:** Every single-bit flip in `Q`, `K`, and `V` is accounted for under the exact longest/latest causal matching rule, for all packed bit widths $1 \le b \le 64$.
- **Near-linear expected speed:** Contracts inactive time gaps, accumulates one-symbol edits by symbol class, and queries longer matches via an exact packed suffix index — achieving **$\widetilde{O}(T)$ expected time** on independent uniform keys (and **$\widetilde{O}(T^2)$ worst-case** at fixed width and joint budget).
- **Drop-in PyTorch training:** Works inside any normal `model(x)` → `loss.backward()` → `optimizer.step()` loop. Includes both first-order linear credit and second-order local trust + joint two-bit corrections.
- **Zero-dependency C++17 core:** Compiles automatically on first import (no PyTorch C++ headers or CUDA toolkit required), or links directly into standalone C/C++ programs.

## Installation

Requires Python 3.10+, a C++17 compiler (`g++`, `clang++`, or MSVC `cl`), NumPy, and PyTorch (for the `nn.Module` layer).

```bash
git clone https://github.com/xiaoiecc/qkv-rosa-fast-exact-backward.git
cd qkv-rosa-fast-exact-backward
python -m pip install -e .
```

The C++17 core is compiled lazily on first use and cached by source hash.

## Quick Start

### PyTorch layer

```python
import torch
from qkv_rosa import NBitQKVRosa

layer = NBitQKVRosa(channels=128, n_bits=8)
optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)

x = torch.randn(2, 32, 128)
target = torch.randn_like(x)

optimizer.zero_grad(set_to_none=True)
loss = (layer(x) - target).square().mean()
loss.backward()
optimizer.step()
```

### Key options

| Argument | Default | What it controls |
|---|---|---|
| `credit_order` | `"second"` | `"first"` for pure linear counterfactual credit; `"second"` adds a local diagonal trust penalty on Q/K plus bounded joint two-bit block corrections. |
| `backward_mode` | `"exact"` | `"exact"` evaluates every single-bit credit term; `"approximate"` opts into stratified residual-row or numerical-channel estimation (`approximation_rows`, `approximation_channels`). |
| `randomized` | `True` | Enables a verified full-key lookup cache inside the suffix index. Never changes routes or numerical results. |

```python
# Pure first-order exact credit
first_order = NBitQKVRosa(128, 8, credit_order="first")

# Opt-in approximate backward (samples 32 residual rows per stream)
approx = NBitQKVRosa(
    128, 8,
    credit_order="second",
    backward_mode="approximate",
    approximation_rows=32,
)
```

### Standalone C++17 (no Python or PyTorch)

```bash
c++ -O3 -std=c++17 -ffp-contract=off \
    examples/native_training.cpp qkv_rosa/csrc/unified_core.cpp \
    -o build/rosa-native
./build/rosa-native
```

## Important notes on scope

- **CPU implementation:** The C++17 matcher runs on CPU. Passing CUDA tensors to `NBitQKVRosa` works transparently via host transfers, but this is not a GPU kernel.
- **Surrogate properties:** The `"second"` diagonal trust term is a chosen local quadratic penalty (not the true Hessian of an arbitrary downstream network). Because trust weights and joint-pair proposals depend on the incoming gradient `grad_y`, combine loss terms before calling `.backward()` if you want their combined update. Double autograd is not supported.
- **Floating-point precision:** Routing decisions (string comparisons, causal boundaries, tie-breaking) are 100% exact. Numeric reductions use IEEE `float64`; reassociated interval sums may differ from naive elementwise summation by ordinary floating-point rounding ($\sim 10^{-14}$ relative L2).

## Repository layout

```text
qkv_rosa/               Core Python package + C++17 engine
├── layer.py            PyTorch NBitQKVRosa layer and autograd bridge
├── transformer.py      Conv ∥ ROSA Transformer blocks (ConvRosaBlock, RosaTransformer)
├── native.py           Batched NumPy bindings to the C ABI
├── core.py             JIT builder + single-stream low-level API
├── reference.py        Literal O(T^3) forward & O(b*T^4) backward oracles for auditing
└── csrc/               Standalone C++17 implementation
    ├── training.h      Versioned C ABI header
    ├── forward.h       Exact causal suffix-automaton matcher
    └── engine/         Causal interval sweep (edit_alphabet.h, interval_credit.h, index.h, delete.h)
docs/                   Bilingual algorithm, training, C API, and benchmark guides
examples/               Quickstart, standalone C++17 example, and multiplication demo
tests/                  Python unit tests + standalone C++ physical-flip oracles
benchmarks/             CPU throughput, layer step, and coordinate-contraction benchmarks
```

## Documentation

| Topic | English | 简体中文 |
|---|---|---|
| Core algorithm & complexity | [docs/algorithm.md](docs/algorithm.md) | [docs/algorithm.zh-CN.md](docs/algorithm.zh-CN.md) |
| Training rules & PyTorch guide | [docs/training.md](docs/training.md) | [docs/training.zh-CN.md](docs/training.zh-CN.md) |
| Standalone C/C++ API & ABI | [docs/c-api.md](docs/c-api.md) | [docs/c-api.zh-CN.md](docs/c-api.zh-CN.md) |
| Benchmarks & accuracy | [docs/benchmarks.md](docs/benchmarks.md) | [docs/benchmarks.zh-CN.md](docs/benchmarks.zh-CN.md) |
| Tests & benchmarks usage | [tests/README.md](tests/README.md) · [benchmarks/README.md](benchmarks/README.md) | — |

## Testing

```bash
python tests/run.py               # Fast Python unit tests
python tests/run.py --exhaustive  # Full CI gate including standalone C++ oracles
```

## Attribution & License

The QKV-ROSA forward retrieval architecture originates from **Bo Peng (BlinkDL)** and the [RWKV-LM RWKV-v8 examples](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8) (`ROSA_QKV_B_1bit` in `251024_rosaQKV_run.py`). This repository is an independent C++17 implementation adding $n$-bit packed symbols, learned fallback embeddings, and fast exact counterfactual backpropagation.

Copyright 2026 xiaoiecc. Released under the [MIT License](LICENSE).
