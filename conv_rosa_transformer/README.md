# Conv∥ROSA with one native credit engine

[English algorithm](../ALGORITHM.md) · [中文推导](../ALGORITHM.zh-CN.md) · [Training guide](../docs/native-training.md) / [中文](../docs/native-training.zh-CN.md)

`NBitQKVRosa` uses one batched C++17 forward and one batched C++17 backward. The backward is the shared subquadratic construction for every input. Its options select the credit rule, exact query caching, and explicit numerical estimation—not a different backend.

```python
import torch
from conv_rosa_transformer.rosa_layer import NBitQKVRosa

layer = NBitQKVRosa(
    channels=128, n_bits=8,
    credit_order="second", randomized=True, backward_mode="exact",
)
x = torch.randn(2, 32, 128)
optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)

optimizer.zero_grad(set_to_none=True)
loss = layer(x).square().mean()
loss.backward()
optimizer.step()
```

Inputs have shape `[B,T,C]`, with positive `C`, `C % n_bits == 0`, and `1 <= n_bits <= 64`. Each channel group is an independent stream. Hard bits use `logit > 0`; rows select the longest causal suffix match, breaking ties by the latest endpoint, and retrieve the payload after that endpoint or the learned empty output.

## Training options

| Option | Meaning |
|---|---|
| `credit_order="first"` | Original linear single-bit output-change credit |
| `credit_order="second"` | Default diagonal local trust model plus a bounded number of joint-bit blocks |
| `randomized=True` | Default bounded cache with full-query-key verification; same numerical answers as disabling it |
| `backward_mode="approximate"` | Explicit residual-row or centered numerical-channel estimation |
| `approximation_rows=32` | Residual-row work budget when the channel budget is zero |
| `approximation_channels=0` | Default keeps all channels in the row estimator; a positive budget selects channel estimation instead |

The ordinary model, loss, and optimizer loop is unchanged. Selected joint corners are evaluated inside the component. Complete mode includes every mathematical credit term; grouped binary64 reductions and FFT have their stated rounding contract. The exact-cache toggle preserves this engine's arithmetic schedule. Sampling has a separate statistical error. See the [integration guide](../docs/native-training.md), [exact-cache guide](../docs/exact-randomized.md), and [approximation guide](../docs/randomized-backward.md).

The default local metric and proposals depend on the incoming gradient. Separate backwards on different losses need not equal one backward on their sum; combine losses when that is the intended rule. Hard-credit double autograd is unsupported. The native matcher is CPU code, so CUDA tensors incur transfers and synchronization. Dense projections and ordinary outputs preserve the tensor device.

## Build and verify

Install from the repository root and run the independent checks:

```bash
python -m pip install -r requirements.txt
python tests/run.py
python tests/run.py --exhaustive
```

The first call compiles a source-hashed C++17 library through `c++` or `CXX`, without PyTorch C++ headers or a CUDA toolkit. The NumPy binding does not require importing PyTorch. The [standalone example](../examples/native_training.cpp) uses the same [C ABI](csrc/training.h) without a framework. See [native-api.md](../docs/native-api.md) for arrays, options, ownership, and diagnostics.

Current complete-backward times, memory, and approximation errors are in [the measured results](../docs/subquadratic-results.md) / [中文](../docs/subquadratic-results.zh-CN.md); reproduction is documented in [benchmarks/README.md](../benchmarks/README.md). Historical JSONs and example artifacts are evidence of their recorded builds, not a current runtime or training-quality guarantee.

## Source map

| File | Role |
|---|---|
| [rosa_layer.py](rosa_layer.py) | Ordinary PyTorch layer, custom autograd adapter, dense projections |
| [native_training.py](native_training.py) | Validated batched NumPy binding and owned forward cache |
| [rosa_core.py](rosa_core.py) | Source-hashed build, packed-word helpers, thin current-engine credit calls |
| [csrc/training.h](csrc/training.h) | Versioned C ABI and caller-owned array contract |
| [csrc/training_impl.h](csrc/training_impl.h) | Packing, forward, trust model, proposals, complete/approximate credit, joint replay and gradient mapping |
| [csrc/forward.h](csrc/forward.h) | Exact causal suffix-automaton forward scan |
| [csrc/subquadratic/index.h](csrc/subquadratic/index.h) | Shared suffix/LCP and wavelet indexes, verified query cache |
| [csrc/subquadratic/delete.h](csrc/subquadratic/delete.h) | AP deletion profiles, streamed periodic sums, arbitrary owner/time projections |
| [csrc/subquadratic/repair.h](csrc/subquadratic/repair.h) | Q heavy/light repair and anchored long K repair |
| [csrc/subquadratic/short_k.h](csrc/subquadratic/short_k.h) | Compact virtual-word dictionaries and weighted short K queries |
| [csrc/subquadratic/correlation.h](csrc/subquadratic/correlation.h) | Streamed arbitrary shifted interval correlations |
| [csrc/subquadratic/engine.h](csrc/subquadratic/engine.h) | Complete composition, centered diagonal features, explicit estimators |
| [csrc/pair_candidates.h](csrc/pair_candidates.h) | Bounded causal joint-bit proposal policy |
| [rosa_reference.py](rosa_reference.py) | Independent literal definitions used by tests |
| [rosa_transformer.py](rosa_transformer.py) | Model blocks combining ROSA with causal convolution and feed-forward layers |

The derivation belongs in [ALGORITHM.md](../ALGORITHM.md) and [ALGORITHM.zh-CN.md](../ALGORITHM.zh-CN.md). The [reference map](../docs/algorithm-reference.md) connects invariants to proofs and tests; the [test guide](../tests/README.md) distinguishes route equality, exact integer checks, floating tolerances, and training integration.
