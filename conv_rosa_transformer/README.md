# Conv∥ROSA with batched native training

`NBitQKVRosa(channels, n_bits, credit_order="second", randomized=False)` uses
one batched C++ forward and one batched C++ backward. Select `credit_order="first"`
for original full singleton credit. Second order adds a diameter-scaled trust
metric and one independent joint block per stream by default. `randomized=True`
is an exact search accelerator and does not enable repair-lane sampling.

```python
from conv_rosa_transformer.rosa_layer import NBitQKVRosa
layer = NBitQKVRosa(128, 8, credit_order="second", randomized=True)
y = layer(x)
loss = criterion(y, target)
loss.backward()
```

The normal optimizer and outer training loop are unchanged. Each selected
joint corner adds one internal component match, not another whole-model pass.
The [integration guide](../docs/native-training.md) / [中文](../docs/native-training.zh-CN.md)
explains the learning rule and its restrictions. The standalone
[C ABI](csrc/training.h) and [example](../examples/native_training.cpp) need no
framework headers. [Results](../docs/native-results.md) include absolute layer
cost and the negative held-out result from the numerical migration.

The main layer is CPU native; CUDA tensors incur transfers. Native arithmetic
is binary64 before casting gradients back to input dtype. Old float32 Torch
training is not bitwise preserved; the new random on/off toggle is. Second-order
updates depend on incoming gradients and do not support double autograd or
additivity between separate loss backwards. First/second modes share hard
evaluation semantics and parameter names/shapes.

| New source | Role |
|---|---|
| [native_training.py](native_training.py) | Thin validated contiguous-array binding |
| [csrc/training.h](csrc/training.h) | Versioned C options and caller-owned arrays |
| [csrc/training_impl.h](csrc/training_impl.h) | Batched packing, forward, proposals, metric, credits and gradient mapping |
| [csrc/forward.h](csrc/forward.h) | Lean exact causal forward scan |
| [csrc/sam_q.h](csrc/sam_q.h) | Sparse exact roots and optional random omitted-block join |
| [csrc/pair_candidates.h](csrc/pair_candidates.h) | Shared bounded causal joint-bit proposals |

## Original model and example documentation

The following model layout and historical demonstration files are retained.
Descriptions of the original singleton layer below refer to `credit_order="first"`.
The current `RosaTransformer` uses the native layer's second-order default;
retaining the model layout does not retain the old training rule by default.


This package combines a hard QKV-ROSA retrieval mixer with a depthwise causal
convolution and a SwiGLU feed-forward block. Its retrieval layer uses the
repository's single unified CPU algorithm for hard forward matching and the
exact single-bit counterfactual Q/K/V credits.

For the algorithm itself, read the [English derivation](../ALGORITHM.md) or
[中文推导](../ALGORITHM.zh-CN.md). Both derive the implementation from the
bit-flip brute-force definition. The root
[`hard_qkv_rosa_explained.py`](../hard_qkv_rosa_explained.py) is a readable
entry point into the same runtime; it does not carry a second optimized
implementation.

## Build and use

Python, NumPy, PyTorch, and a C++17 compiler are required for the layer. The
NumPy-only stream API does not import PyTorch. On Linux/macOS/WSL, the first
call compiles the core through `c++` (or `CXX`) and caches it by source and
compiler options. PyTorch C++ headers, Ninja, and a CUDA toolkit are not
required by the current native core.

Run from the repository root:

```bash
python -m pip install numpy torch
python -m conv_rosa_transformer.self_test
python -m conv_rosa_transformer.multiply_demo --steps 1500
```

Native Windows can use `build.bat` from an x64 Visual Studio developer command
prompt, for example `build.bat self_test.py`. The full structural test harness
uses GCC/Clang-compatible commands; use Linux/WSL for that harness. Published
CPU benchmarks and validation do not constitute native Windows or CUDA
performance measurements.

```python
import torch
from conv_rosa_transformer.rosa_layer import NBitQKVRosaCpp
from conv_rosa_transformer.rosa_transformer import RosaTransformer

layer = NBitQKVRosaCpp(channels=64, n_bits=8, credit_order="first")
x = torch.randn(2, 32, 64)
layer(x).square().mean().backward()

model = RosaTransformer(  # uses the current second-order layer default
    vocab_size=16, channels=64, n_layers=2, n_bits=8,
    conv_kernel=4, max_seq_len=64,
)
logits = model(torch.randint(0, 16, (2, 32)))
```

The historical `NBitQKVRosaCpp`, `NBitQKVRosaForwardCpp`, and
`NBitQKVRosa` names refer to the same implementation. Projection and embedding
parameter names are retained for checkpoint compatibility. Inputs have shape
`[B,T,C]`, with `C % n_bits == 0` and `1 <= n_bits <= 64`. Each group of D
channels is processed as one stream.

The layer preserves its tensor device for ordinary projections and outputs,
but transfers discrete stream work to the CPU core when necessary. It is not
a GPU suffix-matching implementation. Device transfers, dense projections,
and model training are distinct from the
[stream backward measurements](../benchmarks/README.md).

## Forward and backward semantics

Hard codes use `bit = (logit > 0)`. At output t, select the longest Q suffix
matching a K endpoint `r < t`, breaking ties by the latest r. A nonempty route
reads the signed payload at `V[r+1]`; an empty route reads `emb0`.

The core returns the projected output difference after each individual Q/K/V
bit flip. This is a finite counterfactual quantity. It is not the ordinary
derivative of a threshold. The layer explicitly maps that credit to logits:

- `credit_to_logit="bernoulli"` orients the flip as output-at-1 minus
  output-at-0, then multiplies by `p*(1-p)/tau`, with `p=sigmoid(z/tau)`.
- `credit_to_logit="identity"` uses the same oriented difference directly.

Both retain the hard forward. The embedding gradients use the actual selected
payloads and upstream vectors. Full derivations and floating-point guarantee
boundaries are in [the algorithm guide](../ALGORITHM.md).

## Model layout

Each pre-normalized block computes

```text
h = RMSNorm(x)
x = x + ROSA(h) + ShortConv(h)
x = x + SwiGLU(RMSNorm(x))
```

The two mixers share their input. `ShortConv` is a left-padded depthwise
convolution; its default kernel length is 4. `tie_qk=True` shares the Q and K
projection module, making identical input vectors receive identical Q/K
codes. It is a model design option, not a correctness condition of the
stream algorithm. Untied Q/K codes are supported and covered by the stream
tests. `RosaTransformer` also supports tied input/output token embeddings.

## Multiplication demo and historical artifacts

The demonstration task predicts decimal products, least-significant digit
first, then reverses the result for display. Its default convolution kernel
is 32, distinct from the model's generic default of 4. Evaluation reports
per-digit accuracy and exact product-string accuracy.

The bundled [`multiply_ckpt.pt`](multiply_ckpt.pt),
[`multiply_metrics.json`](multiply_metrics.json), and
[`train_log.txt`](train_log.txt) are historical artifacts from the original
upstream implementation at commit
[`32e90e2`](https://github.com/xiaoiecc/qkv-rosa-fast-exact-backward/commit/32e90e272929942a764d696b639409b783c61bb1).
They have not been rerun as a training-quality evaluation of the current core.
The saved metrics record 1,500 steps, digit accuracy 0.1730, and exact accuracy
0.0000. Retaining these files preserves the example's provenance; it does not
establish a new model accuracy or training-speed result.

`overfit_test.py` remains a runnable small training diagnostic. Successful
backward smoke tests check integration and gradients; they are not evidence
of multiplication generalization.

## Source map

| File | Role |
|---|---|
| [`rosa_core.py`](rosa_core.py) | Validated NumPy/PyTorch stream APIs and source-hashed C++ build |
| [`csrc/unified_core.cpp`](csrc/unified_core.cpp) | Shared row loop, common deletion, Q/K repair, bounded scores, numerical contraction |
| [`csrc/sam_q.h`](csrc/sam_q.h) | Static K SAM, causal occurrence list, ordinary edited states and uniqueness certificates |
| [`csrc/sam_routes.h`](csrc/sam_routes.h) | Shared suffix, predecessor/successor, and route queries |
| [`csrc/delete_indexed.h`](csrc/delete_indexed.h) | R staircases and compressed A/H deletion envelopes |
| [`csrc/exact_expansion.h`](csrc/exact_expansion.h) | Residual-preserving interval and snapshot aggregation |
| [`rosa_reference.py`](rosa_reference.py) | Independent hard-forward and all-flip definitions for tests |
| [`rosa_layer.py`](rosa_layer.py) | Custom autograd function, logit mapping, compatible layer aliases |
| [`rosa_transformer.py`](rosa_transformer.py) | RMSNorm, ShortConv, SwiGLU, parallel block, token model |
| [`self_test.py`](self_test.py) | Independent stream checks, also exposed by the root entry point |
| [`build.py`](build.py), [`build.bat`](build.bat) | Convenience build/launch entry points |

## Verification and measurement

From the repository root:

```bash
python tests/run.py
python tests/run.py --exhaustive
python benchmarks/backward.py --repeats 7 --cpu 0 --output build/current.json
```

Choose a valid logical CPU or omit `--cpu` where unsupported. The
[test guide](../tests/README.md) distinguishes route equality, independent
integer credit checks, floating tolerances, targeted cancellation regressions,
and layer gradients. The [measurement guide](../benchmarks/README.md) provides
current absolute latency and throughput before its smaller comparison with
the original upstream implementation. No old backend or runtime selector is
needed to use this package.
