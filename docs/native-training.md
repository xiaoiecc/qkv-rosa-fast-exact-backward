# One forward, one backward: the C++ training component

[中文](native-training.zh-CN.md) · [Exact randomized search](exact-randomized.md)

The implementation now has two layers. C++ takes contiguous arrays and owns
packing, hard matching, finite credits, joint corners, and input gradients.
PyTorch owns dense projections, device/layout conversion, and attachment to
the normal autograd graph. A standalone C++ program calls the same interface
without Python, NumPy, LibTorch, or PyTorch C++ headers.

## Two independent choices

```python
from conv_rosa_transformer.rosa_layer import NBitQKVRosa

first = NBitQKVRosa(128, 8, credit_order="first", randomized=False)
second = NBitQKVRosa(128, 8, credit_order="second", randomized=True)
```

`credit_order` selects the learning rule. `randomized` selects an exact search
strategy for that same rule: it does not thin credit coordinates, approximate
candidate maxima, or change joint-proposal seeds. The switch changes lookup
work, while preserving the numerical answer.

The training loop remains ordinary:

```python
optimizer.zero_grad(set_to_none=True)
prediction = model(x)       # model contains ROSA
loss = criterion(prediction, target)
loss.backward()
optimizer.step()
```

Both rules use hard Q/K/V forward and the same causal longest-then-latest
matching. Each batch/group is an independent stream. Second-order mode selects
at most one joint block by default, adding at most one internal ROSA matcher
replay per stream during backward. It does not rerun the downstream network
or task loss. First-order mode does not perform this joint replay.

`randomized=False` disables the new exact-search accelerator. The second-order
rule already has a bounded randomized joint proposer, whose random numbers are
identical in both accelerator modes. With no first-order joint proposals, the
accelerator also consumes no additional Torch RNG, preserving downstream
Dropout and other framework randomness. Evaluation consumes no training RNG.

## What first order computes

For the complete output change $`\delta`$ caused by one hard-bit edit,

```math
C^{(1)}=g^\top\delta,
\qquad g=\partial L/\partial Y.
```

The engine computes this finite-difference object exactly for Q/K/V singleton
edits. It is not the classical derivative of threshold quantization or a full
rerun of an arbitrary nonlinear downstream loss. The default mapping orients
the credit as bit-1 minus bit-0 and multiplies by the sigmoid probability's
derivative with respect to its logit.

## Second order: diagonal trust and joint cooperation

A normal backward gives the component an upstream gradient. At the same
output $`Y_0`$, the losses

```math
L_a(Y)=g_0^\top(Y-Y_0)+\tfrac12a\|Y-Y_0\|^2
```

have the same gradient and different curvature. This API therefore does not
claim to recover the true Hessian of an arbitrary downstream model. “Second
order” names an explicit local quadratic credit model:

```math
C^{(2)}(\delta)=g^\top\delta+
\tfrac12\sum_{t,j}h_{tj}\delta_{tj}^2.
```

One group's output set contains the empty output $`e_0`$ and all signed payload
vertices. Its diameter determines the scale:

```math
\Delta_Y=\max\left(2\|e_1\|_2,\ \||e_0|+|e_1|\|_2\right),
\qquad h_{tj}=\frac{\|g_t\|_2}{\Delta_Y}.
```

The implementation floors zero amplitude. A row shares one diagonal
coefficient, constructed in $`O(TD)`$ work. Q/K routing edits use this penalty;
V retains first-order credit. An edit aligned with the negative gradient, of
length $`a\le\Delta_Y`$, has model cost

```math
-a\|g_t\|_2+\frac{a^2\|g_t\|_2}{2\Delta_Y}
\le-\tfrac12a\|g_t\|_2\lt0.
```

This avoids canceling a useful jump between antipodal payloads, and includes
the learned empty output's scale. The guarantee concerns the chosen local
model, not every possible true task loss.

Joint blocks address a different obstacle: both singleton edits can have zero
signal while the joint edit helps. For corner costs $`C_i,C_j,C_{ij}`$, define
$`\kappa=C_{ij}-C_i-C_j`$. With the corners held fixed for the current local
calculation, the independent flip-probability objective is

```math
F=\alpha_iC_i+\alpha_jC_j+\alpha_i\alpha_j\kappa,
\qquad
\partial_{z_i}F=\alpha_i'(C_i+\alpha_j\kappa).
```

Singleton costs already come from the full engine; one internal matcher replay
gives the joint cost. Selected coordinate gradients are replaced before
accumulation into shared projection parameters. Default J=1 bounds replay
cost. Cross-block interactions are not solved in full.

![A real ROSA input with no singleton signal and useful joint credit](../images/diameter_barrier.png)

The second-order metric and active-row proposal selection depend on g, so the
rule is generally not additive across different incoming loss gradients.
Combining loss terms and calling backward once is supported; separate backward
calls followed by accumulation need not equal that result. Positive rescaling
of the same loss is supported. Hard-credit double autograd is explicitly
unsupported rather than returning an incomplete higher-order derivative.

## C++ interface and layouts

The public header is [`csrc/training.h`](../conv_rosa_transformer/csrc/training.h).
Call `rosa_training_defaults(&options)` to initialize the versioned, sized
options struct. The default is the second-order surrogate. For first-order
credit, set all three related fields consistently:

```cpp
RosaTrainingOptions options;
rosa_training_defaults(&options);
options.credit_order = 1;
options.joint_pairs = 0;
options.trust_diagonal = 0.0;
options.randomized = 1;  // independent exact search option
```

The Python constructor derives these settings from `credit_order` automatically.

| Array | Contiguous row-major layout | Type |
|---|---|---|
| Q/K/V logits, output, incoming and input gradients | `[B,G,T,D]` | `double` |
| Empty output, payload scale, and their gradients | `[G,D]` | `double` |
| Packed Q/K | `[B,G,T]` | `uint64_t` |
| V bits | `[B,G,T,D]` | `uint8_t` |
| Match lengths and endpoints | `[B,G,T]` | `int32_t` |

`rosa_training_forward` processes the entire batch in one call: packing,
the completed accelerated exact matcher, payload selection, and the hard
state needed by backward. `rosa_training_backward` constructs the metric,
proposes blocks, calls the shared credit engine, replays selected joint corners,
maps Q/K/V logits, and accumulates embedding gradients for all streams.
The caller allocates memory and keeps inputs alive during each call. Status
codes and optional diagnostics are specified in the header.

Compile and run the standalone example:

```bash
c++ -O3 -std=c++17 -ffp-contract=off examples/native_training.cpp \
    conv_rosa_transformer/csrc/unified_core.cpp -o /tmp/rosa-native
/tmp/rosa-native
```

All production computation uses this one core. The old Python arithmetic in
the tests is a migration oracle, never a runtime backend choice.

## Numerical and model compatibility

Q/K/V/O parameter names, shapes, and layer aliases are preserved. Existing
weights load strictly; training configuration must accompany them. Forward
uses hard values, and both credit modes have identical evaluation semantics.

Native credit and mapping arithmetic uses binary64 intermediates. Compared
with earlier Python/Torch float32 operations, reduction order and rounding can
differ. Migration checks use declared tolerances rather than claiming all old
float32 training trajectories remain bitwise identical. The native accelerator
toggle does not change arithmetic order and is checked for bitwise equality.

When active rows exceed `joint_rows`, the native weighted row sampler uses
`mt19937_64`; the frozen Python implementation used NumPy `PCG64`. The sampling
rule is preserved, but the same integer seed need not select the same rows
across those implementations. Within the native implementation, switching
`randomized` does preserve the proposal stream and framework RNG state.

The framework adapter preserves normal GradScaler overflow skipping and supports
ordinary optimizers, finite loss scaling, and checkpoint use. The C++ engine is
CPU code; CUDA tensors still incur CPU transfer and synchronization. No GPU
matching kernel is claimed.

Accuracy, timing, identical-randomness training, and negative results are in
the [results report](native-results.md). Rule selection should depend on task
results per unit time, not on the complexity of the credit formula or an
isolated search speedup.
