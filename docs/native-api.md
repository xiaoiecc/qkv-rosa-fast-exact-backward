# Standalone C++ training API

The selected production rule is implemented in C++17. The Python layer only
performs dense projections, layout/device conversion, and ordinary autograd
bookkeeping. There is one batched native forward call and one batched native
backward call; there is no Python loop over streams in the layer.

Public declarations are in
[`training.h`](../conv_rosa_transformer/csrc/training.h), implementation in
[`training_impl.h`](../conv_rosa_transformer/csrc/training_impl.h), and thin
NumPy bindings in [`native_training.py`](../conv_rosa_transformer/native_training.py).
The implementation is included by the same `unified_core.cpp` that supplies
the single-bit engine. It reuses the existing lean `SamForward` and unified
deletion/repair algorithm; it does not embed a second optimized kernel.

## Build without a framework

From the repository root:

```bash
c++ -O3 -std=c++17 -ffp-contract=off examples/native_training.cpp \
    conv_rosa_transformer/csrc/unified_core.cpp -o /tmp/rosa-native
/tmp/rosa-native
```

The [complete example](../examples/native_training.cpp) constructs options,
packs hard Q/K/V through the native forward, computes one output gradient, and
calls the native backward. It needs no Python, NumPy, Torch headers, or CUDA.
`training.h` itself is C-compatible; its implementation is compiled as C++17.

## Options and seeds

Call `rosa_training_defaults(&options)` before setting individual fields. The
struct carries `abi_version` and `struct_size`; unknown versions/sizes fail
instead of silently interpreting incompatible memory.

| Field | Meaning |
|---|---|
| `credit_order` | 1 = original first-order singleton credits; 2 = diagonal trust plus independent joint blocks |
| `randomized` | Answer-exact randomized root preparation; it never requests repair thinning |
| `logit_mode` | 0 = oriented finite credit times sigmoid derivative; 1 = oriented identity mapping |
| `joint_pairs` | Maximum disjoint two-bit blocks per stream; default 1 |
| `joint_rows` | Maximum proposal query rows; default 32 |
| `active_rows` | 1 = nonzero upstream-gradient rows, 0 = evenly spaced input rows |
| `tau` | Positive sigmoid temperature; default 1 |
| `trust_diagonal` | Default 0.5, equivalent to gradient norm divided by output-set diameter |
| `proposal_seed` | Joint proposal seed, frozen for this forward/backward |
| `repair_seed` | Independent seed for exact root layout and optional explicit repair sampling |
| `repair_budget` | 0 = complete exact repair computation; positive = legacy research estimator, independently selected |

The C struct is explicit: `rosa_training_defaults` initializes the complete
second-order configuration (`credit_order=2`, `joint_pairs=1`,
`trust_diagonal=0.5`). To select first order, change all three related fields:

```cpp
RosaTrainingOptions options;
rosa_training_defaults(&options);
options.credit_order = 1;
options.joint_pairs = 0;
options.trust_diagonal = 0.0;
```

Setting only `credit_order=1` leaves inconsistent second-order settings and
returns status 1. The Python `credit_order="first"` constructor sets coherent
defaults automatically. The standalone example accepts `--first` to exercise
this configuration. For every stream s, seeds receive the same fixed uint64
offset `0x9e3779b97f4a7c15 * (s+1)` with wraparound. Turning exact acceleration
on or off does not change proposal seeds or framework RNG consumption.

The Torch adapter consumes two CPU Torch RNG draws only when joint proposals
or explicit repair sampling are enabled, identically for exact acceleration
on/off. First-order/no-MC training consumes no such RNG. Evaluation consumes
no proposal RNG. Checkpoint recomputation can therefore use the normal
framework RNG-preservation contract.

Active-row sampling uses weighted exponential races with `mt19937_64`. For
more than `joint_rows` nonzero rows, this changes the old NumPy `PCG64` seed
trajectory, while preserving the stated sampling law. When all active rows
fit the budget, rows and native joint proposals are unchanged. All selected
joint corners are evaluated exactly under the frozen local trust model.

## Arrays and ownership

All arrays are contiguous, row-major, host memory. Floating inputs/outputs use
binary64, packed words uint64, V bits uint8, and routes/lengths int32.

| Arrays | Shape |
|---|---|
| Q/K/V logits, output, upstream gradient, Q/K/V logit gradients | `[B,G,T,D]` |
| Q/K packed words, lengths, routes | `[B,G,T]` |
| V bits | `[B,G,T,D]` |
| Embeddings and their gradients | `[G,D]` |
| Forward counters | `[B*G,4]` |
| Backward counters | `[B*G,35]` |
| Optional pair codes | `[B*G,joint_pairs,2]` |
| Optional corner costs | `[B*G,joint_pairs,4]` |

The caller owns every buffer and keeps it alive during the call. Forward emits
packed Q/K/V to reuse in backward. Backward must receive those same hard codes,
logits, and embeddings; replacing them independently invalidates the cache
contract. The NumPy `TrainingCache` independently copies the logits, embeddings
and options, so later changes to caller storage do not alter its backward.
Do not mutate the cache itself between forward and backward. These snapshots
require an additional copy of the five floating input arrays. Torch additionally saves the original tensors
so its version checks detect ordinary in-place modification.

Only optional pair codes/costs may be null for nonempty outputs. A null pointer
for a zero-sized array is allowed. Domains are `B>=0`, `G>0`, `0<=T<2**28`,
`1<=D<=64`, `T*D<=2**31-1`, and `B*G<=2**31-1`. Actual allocations must fit
available memory. CPU caches retain double logits and packed hard codes; CUDA
inputs require transfers and synchronization, not a GPU execution path.

## Operations and costs

`rosa_training_forward` packs hard bits using `logit > 0`, runs the lean exact
causal matcher, and gathers the signed payload or empty embedding. Forward
semantics are identical for first/second order and exact acceleration on/off.

`rosa_training_backward` constructs the local metric, chooses bounded disjoint
blocks, calls the shared full credit engine, reruns one exact joint route per
selected block, maps credits to logits, and scatters V/embedding gradients.
There is no downstream model replay. A zero upstream adjoint returns zero
gradients without constructing credit indices. Other streams use the same
general engine. No cross-call routing index or model state is silently cached.

The diagonal penalty protects Q/K routing edits; V keeps its first-order
credit. The default is a geometric trust model, not an inferred Hessian. The
metric and active-row proposals depend on g, so separate loss backward calls
need not add to the result of one combined loss backward. Positive scaling is
supported. The Torch hard-credit path explicitly rejects double autograd.

Joint cost accumulation uses the existing residual-preserving expansion on
baseline-relative terms. Native sigmoid and accumulation use binary64 before
the adapter casts to tensor dtype. Consequently, comparison to the previous
Python/Torch float32 implementation uses a floating tolerance rather than a
false bitwise claim. Exact acceleration on/off preserves the same numerical
order and is checked bitwise, including stochastic downstream models.

## Return codes and counters

| Status | Meaning |
|---|---|
| 0 | Success |
| 1 | Invalid shape, option, ABI, or required pointer |
| 2 | Allocation failure |
| 3 | Core/numerical execution failure |
| 4 | Nonfinite required input |

The Torch adapter catches nonfinite incoming gradients before this strict
native interface and returns nonfinite tensor gradients. This permits an
ordinary GradScaler to detect overflow and skip its optimizer step.

Forward counters are baseline passes, transition probes, hash probe steps,
and endpoint queries. Backward indices 0–22 are the existing singleton-engine
work counters. Remaining indices are:

| Index | Counter |
|---:|---|
| 23 | Selected pairs |
| 24 | Joint matcher replays |
| 25 | Active/input proposal rows |
| 26 | Candidate events |
| 27 | Supported repair lanes |
| 28 | Selected repair lanes |
| 29 | Whether this stream's repairs were exact |
| 30 | Logit-gradient elements per Q/K/V array |
| 31 | Whether exact randomized root preparation was requested |
| 32 | Proposal rows retained |
| 33 | Nonzero-cost joint contraction components visited |
| 34 | Singleton-engine calls (zero for zero-adjoint short circuit) |

Optional pair codes below `T*D` refer to K; codes at least `T*D` refer to Q
after subtracting `T*D`. Divide the remaining code by D for its position and
take modulo D for its bit. Corner costs are `C_i,C_j,C_ij,kappa` in that order.
Only the first `selected_pairs` trace entries are valid.

## Validation scope

`tests/test_native_training.py` compares the native complete rule with the
frozen previous Python calculation, including both mappings, empty inputs,
multiple groups, bit 63, first/second order, and the real zero-signal joint
barrier. Double comparisons use `2e-11` absolute/relative tolerance; float32
mapping uses `2e-6`. On/off acceleration comparisons require bitwise equality.
Normal optimizer, GradScaler, checkpoint, and serialization tests exercise
the production layer. Historical Python-mocking checks explicitly load a
test-only snapshot; production never selects it as a backend.
