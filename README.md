# Fast Exact Backpropagation for QKV-ROSA

One C++ implementation for hard QKV-ROSA matching, exact single-bit
counterfactual credit, and ordinary model training. It preserves untruncated
longest-match/latest-endpoint routing. Bit credit is a finite hard-output
change projected onto an upstream vector, not the ordinary derivative of a
threshold.

**Learn the algorithm:** [English](ALGORITHM.md) · [中文推导](ALGORITHM.zh-CN.md).
Both start from the literal O(DT⁴) all-flip brute force and derive the shared
work, invariants and costs that remain. Further guides cover
[native training](docs/native-training.md) / [中文](docs/native-training.zh-CN.md),
[exact randomization](docs/exact-randomized.md) / [中文](docs/exact-randomized.zh-CN.md),
and [forward acceleration](docs/forward.md) / [中文](docs/forward.zh-CN.md).

## Use it as an ordinary model component

```python
import torch
from conv_rosa_transformer.rosa_layer import NBitQKVRosa

layer = NBitQKVRosa(
    channels=128,
    n_bits=8,
    credit_order="second",  # "first" selects original singleton credit
    randomized=True,       # exact search acceleration; default is False
)
optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
x = torch.randn(2, 32, 128)
target = torch.randn_like(x)

optimizer.zero_grad(set_to_none=True)
y = layer(x)
loss = (y - target).square().mean()
loss.backward()
optimizer.step()
```

`first` uses the original complete single-bit credit. `second` adds a
component-defined diagonal trust penalty and at most one independent joint-bit
block per stream by default. It needs no supplied Hessian, whitening,
additional full-model evaluation or special optimizer. Each selected joint
corner does perform one internal matcher replay, included in the measurements.
The diagonal is a local geometric model; it does not recover an arbitrary
network's true curvature or guarantee loss reduction.

Packing, matching, metric preparation, proposals, singleton credits, joint
replays, logit mapping and embedding gradients run in batched C++. The Torch
adapter makes one native forward call and one native backward call, with no
Python loop over batch/group streams. Q/K/V/O parameter names and shapes are
preserved. Both modes use the same hard evaluation.

`randomized=True` selects an exact randomized root join. Every seed retains
all legal one-bit neighbors, with full-word verification. It never turns on
gradient thinning. On/off outputs, gradients, proposal streams and tested
training updates are bitwise identical. The default stays off because some
inputs do not benefit. The separate advanced `repair_samples` research option
has approximation error and is not selected by this switch.

## Current complete-layer speed

These are **absolute current CPU costs**, including dense Q/K/V/O projections
and their gradients. They exclude the downstream model, task loss and
optimizer. Both columns below enable exact randomization. B is batch size,
G is the number of D-bit groups. Medians of seven alternating samples, one
pinned Intel Xeon Platinum 8573C core, one Torch thread, GCC 13.3,
PyTorch 2.14.1+cpu, NumPy 2.3.5:

| T | D | B / G | First-order backward | Second-order backward | Second-order forward + backward |
|---:|---:|---:|---:|---:|---:|
| 128 | 8 | 4 / 2 | 2.094 ms | 3.373 ms | 4.032 ms |
| 1,024 | 8 | 1 / 1 | 2.272 ms | 2.935 ms | 4.573 ms |
| 1,024 | 16 | 2 / 4 | 15.415 ms | 21.679 ms | 26.326 ms |
| 8,192 | 8 | 1 / 1 | 23.940 ms | 27.678 ms | 31.318 ms |
| 8,192 | 16 | 1 / 1 | 17.482 ms | 26.858 ms | 30.693 ms |
| 8,192 | 64 | 1 / 1 | 66.069 ms | 70.939 ms | 83.044 ms |

The [full results](docs/native-results.md) / [中文实测](docs/native-results.zh-CN.md)
include both toggle states, the hard-component scope, every sample and the
[source-hashed raw report](benchmarks/results/native_component.json).
Native second-order forward+backward is 1.13–4.48× faster than the frozen
previous research implementation over these six shapes. Randomization alone
has mixed full-layer results: 0.723–1.349× across both modes. It is not a
universal speedup.

### What the exact randomized join saves

For a T8192 D64 input with planted one-bit repairs, the complete native
credit engine takes **12.033 ms**. Switching randomization on reduces the
same current engine from 20.094 ms: **1.670×**. Root probes fall from 524,288
to 16,384 bucket queries and 2,048 exact member checks. Both settings use a
sparse root table and produce bitwise-identical Q/K/V credits.

Across all 15 cases, current on versus the preceding research baseline ranges
from 0.972–1.814×, retaining the periodic regression. This credit-core scope
excludes projections and the extra second-order training work; it must not
be confused with the complete-layer table.
[Measurements](benchmarks/results/exact_randomized.json) ·
[coverage proof and bounded buckets](docs/exact-randomized.md).

### Forward matching

The completed forward scan uses a causal suffix automaton and a lazy
latest-endpoint index, with O(T) auxiliary memory. At T8192, native route-only
matching takes 0.367 ms for all-equal words, 1.583 ms for random 8-bit words,
and 0.475 ms for wide words with planted historical matches. Across its
21-case saved matrix, public `matching_stats` is 1.48–9.26× faster than the
preceding forward. The general bound remains O(T log T), assuming constant
average full-key hash probe cost. Payload gathering and neural projections
are outside these route-only times. The unchanged measured forward header
and historical whole-library scope are documented in the
[forward report](docs/forward.md).

### Small comparison with the original upstream backward

The repository's earlier paired experiment against original commit
[`32e90e2`](https://github.com/xiaoiecc/qkv-rosa-fast-exact-backward/commit/32e90e272929942a764d696b639409b783c61bb1)
is retained below. These are historical first-order public stream-call
measurements, not freshly measured native-layer results from this patch.

| Input | T / D | Original | Published unified backward | Speedup |
|---|---:|---:|---:|---:|
| Random 16-bit | 4,096 / 16 | 61.020 ms | 6.972 ms | 8.75× |
| Random 8-bit | 1,024 / 8 | 57.685 ms | 2.143 ms | 26.92× |
| Shadow, two payloads | 1,024 / 1 | 3749.891 ms | 4.098 ms | 915.00× |

All original samples, tolerances and build details remain in the
[benchmark history](benchmarks/README.md#original-upstream-comparison).

## Training evidence and numerical migration

The selected second-order rule is available as a normal component, but better
credit modeling does not imply better results on every task. In three fresh
100-update MSE-retrieval runs, native integration reduces average training
loss from 0.068074 to 0.009362 while **held-out loss worsens from 0.278856 to
0.528012** against the frozen prior Python implementation. Fresh-data CE also
has slightly worse loss, with both at 100% accuracy. These negative results
are preserved.

The C++ mapping and reductions use binary64 intermediates, where the old Torch
path performed some float32 operations. A targeted trace finds an initial
maximum gradient difference of 1.49e-8, amplified by Adam near zero; hard bits
first diverge after five updates at logits near 1e-13. Thus old Python training
trajectories are not promised bitwise compatibility. Native randomization
on/off itself remains bitwise identical through every measured update.
The [results report](docs/native-results.md#4-ordinary-training-an-exact-toggle-and-a-numerical-migration-regression)
contains every seed, timing, negative result and the first-divergence trace.

The roughly 17× speedup in tiny, many-stream toy training mainly removes
Python orchestration. It is not a randomization speed claim and does not
establish larger-model accuracy or throughput.

## Standalone C++

The public [C ABI](conv_rosa_transformer/csrc/training.h) uses caller-owned
contiguous arrays and has no Torch headers. [API details](docs/native-api.md)
cover layouts, options, status codes and diagnostics.

```bash
c++ -O3 -std=c++17 -ffp-contract=off examples/native_training.cpp \
    conv_rosa_transformer/csrc/unified_core.cpp -o /tmp/rosa-native
/tmp/rosa-native
/tmp/rosa-native --first
```

Initialize options with `rosa_training_defaults`. To select original first
order from those second-order defaults, set `credit_order=1`, `joint_pairs=0`
and `trust_diagonal=0.0`; `randomized` is independent. The Python constructor
derives the related defaults automatically.

## Install, inspect and verify

```bash
python -m pip install -r requirements.txt
python tests/run.py --exhaustive
```

The CPU core is source-hashed and compiled on first use. The measured build
uses C++17 and `-ffp-contract=off`, without fast-math. CUDA tensors currently
incur transfers and synchronization; this is not a native GPU implementation.
Linux/GCC is the measured environment. The existing Windows developer-shell
build helper is retained, but the new native training path was not measured
there.

The [test guide](tests/README.md) describes independent all-flip oracles,
structural checks, bit63, floating cancellation, ordinary optimizers,
checkpoint, GradScaler and RNG parity. The [research scripts](research/README.md)
reproduce the finite training study. Runtime code does not import research.

| Entry | Role |
|---|---|
| [Algorithm guide](ALGORITHM.md), [中文](ALGORITHM.zh-CN.md) | Derivation from the literal all-flip algorithm |
| [training.h](conv_rosa_transformer/csrc/training.h) / [training_impl.h](conv_rosa_transformer/csrc/training_impl.h) | Standalone batched native training |
| [rosa_layer.py](conv_rosa_transformer/rosa_layer.py) / [native_training.py](conv_rosa_transformer/native_training.py) | Thin Torch / NumPy adapters |
| [rosa_core.py](conv_rosa_transformer/rosa_core.py) | Validated stream credits and source-hashed build |
| [rosa_reference.py](conv_rosa_transformer/rosa_reference.py) | Independent hard-forward and all-flip oracle |
| [csrc](conv_rosa_transformer/csrc/) | Shared forward, deletion, repair, scores and exact-event accumulation |
| [Conv∥ROSA package](conv_rosa_transformer/README.md) | Model integration and historical multiplication demo |

The stream API retains `exact_stream_bit_credits` with five returns and
`credits_numpy` with an additional work-counter dictionary. Supported words
have 1≤D≤64, T<2²⁸, T·D≤2³¹−1 and must fit available memory. Empty streams
are supported. Full-key route comparisons are exact; floating sums still
round under documented conditions. Sparse roots use O(UQ+R) storage, while
complete output gradients still require O(TD). Some continuation and periodic
deletion work remains quadratic; this is not a general near-linear backward.

The second-order metric/proposals depend on incoming g, so separately
backpropagating distinct losses need not equal a single backward on their sum.
Positive scaling of the same loss is supported; double autograd is explicitly
unsupported. These are surrogate-credit contracts, documented in the
[integration guide](docs/native-training.md).

## Architecture attribution and license

The QKV-ROSA forward architecture comes from **Bo Peng (BlinkDL)** and the
[RWKV-LM RWKV-v8 examples](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8),
including `samx_qkv_slow` / `ROSA_QKV_B_1bit` in
[`251024_rosaQKV_run.py`](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v8/251024_rosaQKV_run.py).
This independent implementation supports n-bit hard symbols, a learned empty
payload and exact single-bit counterfactual backward. Exact block indexing
builds on the multi-index Hamming-search idea, with attribution in its guide.

Copyright 2026 xiaoiecc. Licensed under the [MIT License](LICENSE).
