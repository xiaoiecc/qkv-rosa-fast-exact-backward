# Fast Exact Backpropagation for QKV-ROSA

QKV-ROSA (from the RWKV-8 architecture) is a *hard* retrieval mixer: every
output is the payload of one longest-suffix match, so the ordinary derivative
is zero almost everywhere. This repository trains it anyway — by computing,
for every bit of Q/K/V, the **exact finite change in the hard output** caused
by flipping that bit, projected onto the upstream gradient. A single C++
implementation delivers that counterfactual credit fast enough for ordinary
training, bitwise-validated against the brute-force definition.

```python
import torch
from conv_rosa_transformer.rosa_layer import NBitQKVRosa

layer = NBitQKVRosa(channels=128, n_bits=8)  # second-order credit, exact search
optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
x = torch.randn(2, 32, 128)

loss = (layer(x) - torch.randn_like(x)).square().mean()
loss.backward()
optimizer.step()        # ordinary model, ordinary optimizer
```

**Learn the algorithm:** the [principle-first guide](ALGORITHM.md) /
[中文推导](ALGORITHM.zh-CN.md) explains the forward, the credit definition and
the three insights in one sitting; the [exhaustive reference](docs/algorithm-reference.md)
/ [中文](docs/algorithm-reference.zh-CN.md) keeps every invariant, certificate
and cost parameter.

## Headline numbers

Complete native backward costs (validation, matching, all credits, logit
mapping; medians of nine interleaved samples, one pinned CPU core):

| Input | T / D | Exact backward | A0 (approximate) |
|---|---:|---:|---:|
| Random words | 8,192 / 16 | 10.916 ms | 12.932 ms |
| Captured long-prefix adjoint | 4,096 / 8 | 264.940 ms | **12.432 ms** |
| Captured long-prefix adjoint | 2,048 / 64 | 302.257 ms | **34.084 ms** |

Against the original upstream backward (first-order, historical paired
experiment): **8.75×–915×** depending on the input family. On a structured
prefix-copy training task, 20 complete forward/backward/Adam updates at
length 4096 take **1.53 s** with the approximate mode vs 19.50 s exact
(**12.74×**), with all hard forwards matching exactly.

The [full reports](#documentation-map) contain all 26 benchmark cases, both
credit orders, error measurements, and the seeds where approximation loses.

## Documentation map

| Document | What it answers |
|---|---|
| [ALGORITHM.md](ALGORITHM.md) / [中文](ALGORITHM.zh-CN.md) | How the exact backward works — start here |
| [Exhaustive reference](docs/algorithm-reference.md) / [中文](docs/algorithm-reference.zh-CN.md) | Complete derivation from the all-flip brute force |
| [Native training](docs/native-training.md) / [中文](docs/native-training.zh-CN.md) | The layer's credit rule, second order, integration contracts |
| [Exact randomization](docs/exact-randomized.md) / [中文](docs/exact-randomized.zh-CN.md) | The randomized root join that stays exact |
| [Randomized backward](docs/randomized-backward.md) / [中文](docs/randomized-backward.zh-CN.md) | The approximate estimator, certificates, FP boundary |
| [Results](docs/randomized-results.md) / [中文实测](docs/randomized-results.zh-CN.md) · [native results](docs/native-results.md) | All measurements, including failures |
| [Forward](docs/forward.md) / [中文](docs/forward.zh-CN.md) | The causal SAM forward scan and its timings |
| [C API](docs/native-api.md) · [test guide](tests/README.md) · [benchmarks](benchmarks/README.md) | Interfaces and reproduction |

## Scope and limitations

Everything below is preserved from the full reports — read them before
drawing conclusions:

- **Approximation can fail badly.** At 32 residual rows, raw Q/K gradient
  errors exceed 100% on some inputs, and A8 has a severe first-order
  cancellation counterexample. These cases are kept as reported failures;
  `backward_mode="exact"` is the default
  ([details](docs/randomized-results.md)).
- **Better credit ≠ better loss on every task.** Native integration improved
  training loss 0.068→0.009 while *worsening* held-out loss 0.279→0.528 in
  one MSE-retrieval study ([details](docs/native-results.md)).
- **No bitwise compatibility with old float32 Python trajectories** (C++ uses
  binary64; divergence trace documented). Randomization on/off *is* bitwise
  identical.
- **CPU only.** CUDA tensors incur transfers; Linux/GCC is the measured
  environment; this is not a GPU implementation.
- **Surrogate-credit contracts.** Second-order proposals depend on incoming
  `g`: no double autograd, no additivity across separate loss backwards
  ([integration guide](docs/native-training.md)).
- The structural worst case remains quadratic up to log factors; this is not
  a near-linear theorem ([cost model](docs/algorithm-reference.md#11-put-the-pieces-together-and-count-the-remaining-work)).

## Install, verify, standalone C++

```bash
python -m pip install -r requirements.txt
python tests/run.py --exhaustive   # independent all-flip oracles, structural checks
```

The C++17 core compiles on first use, source-hashed, `-ffp-contract=off`, no
fast-math. A framework-free [C ABI](conv_rosa_transformer/csrc/training.h)
([API details](docs/native-api.md)) builds standalone:

```bash
c++ -O3 -std=c++17 -ffp-contract=off examples/native_training.cpp \
    conv_rosa_transformer/csrc/unified_core.cpp -o /tmp/rosa-native
```

Words support `1≤D≤64`, `T<2²⁸`, `T·D≤2³¹−1`; empty streams are supported.

## Attribution and license

The QKV-ROSA forward architecture comes from **Bo Peng (BlinkDL)** and the
[RWKV-LM RWKV-v8 examples](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8)
(`samx_qkv_slow` / `ROSA_QKV_B_1bit` in
[`251024_rosaQKV_run.py`](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v8/251024_rosaQKV_run.py)).
This independent implementation supports n-bit hard symbols, a learned empty
payload and exact single-bit counterfactual backward.

Copyright 2026 xiaoiecc. Licensed under the [MIT License](LICENSE).
