# Standalone C++ training API

[Training guide](native-training.md) · [Algorithm](../ALGORITHM.md) · [Current measurements](subquadratic-results.md)

One C++17 implementation provides the hard forward and subquadratic credit computation. The C complete and approximate backward functions are entry points into that same engine. There is no engine selector. Python performs projections, array/device conversion, and normal autograd bookkeeping; a batch/group collection is processed in one native call.

Public declarations: [training.h](../conv_rosa_transformer/csrc/training.h). Implementation: [training_impl.h](../conv_rosa_transformer/csrc/training_impl.h). NumPy adapter: [native_training.py](../conv_rosa_transformer/native_training.py).

## Build and call

```bash
c++ -O3 -std=c++17 -ffp-contract=off examples/native_training.cpp \
    conv_rosa_transformer/csrc/unified_core.cpp -o /tmp/rosa-native
/tmp/rosa-native
/tmp/rosa-native --first --deterministic
/tmp/rosa-native --approximate
```

The example needs no Python, NumPy, PyTorch headers, or CUDA. `training.h` is C-compatible; compile its implementation as C++17. Initialize options through `rosa_training_defaults` and `rosa_approximation_defaults`, rather than hard-coding layouts or uninitialized fields.

| Function | Purpose |
|---|---|
| `rosa_training_forward` | Pack hard Q/K/V, run the causal matcher, return output and saved routes |
| `rosa_training_backward` | Complete finite-credit computation, logit mapping, embedding gradients and bounded joint corrections |
| `rosa_training_backward_approx` | Same local rule with explicit residual-row or numerical-channel estimation |
| `rosa_training_backward_subquadratic` | Detailed alias for the same engine, accepting an optional approximation description and a structural diagnostic buffer |

The detailed name exists for diagnostics and the versioned C interface; it is not a second backend. A null approximation pointer requests complete evaluation. All entries use the same hard routing rule and fixed local score.

## Options and ABI

`RosaTrainingOptions` retains ABI version 1 and its established layout. Every options struct carries `abi_version` and `struct_size`; initialize these from the header's defaults function. Unsupported sizes or versions return status 1.

| Training field | Default | Meaning |
|---|---:|---|
| `credit_order` | 2 | 1 = first-order singleton credit; 2 = diagonal trust and joint-bit cooperation |
| `randomized` | 1 | Verified full-query-key cache; never requests numerical sampling |
| `logit_mode` | 0 | 0 = oriented credit times sigmoid derivative; 1 = oriented identity |
| `joint_pairs` | 1 | Maximum disjoint two-bit blocks per stream |
| `joint_rows` | 32 | Maximum proposal rows inspected |
| `active_rows` | 1 | 1 = nonzero upstream-gradient rows; 0 = evenly spaced input rows |
| `tau` | 1 | Positive sigmoid temperature |
| `trust_diagonal` | 0.5 | Local trust scale; default gives gradient norm divided by output-set diameter |
| `proposal_seed` | 0 | Joint proposal seed |
| `repair_seed` | 0 | Independent query-cache layout seed and approximation sampling salt |
| `repair_budget` | 0 | Reserved for layout compatibility; must remain zero |
| `reserved` | 0 | Leave at the initialized value |

To choose first order in C, set the related fields consistently:

```cpp
RosaTrainingOptions options;
rosa_training_defaults(&options);
options.credit_order = 1;
options.joint_pairs = 0;
options.trust_diagonal = 0.0;
```

Changing only `credit_order` leaves the default joint/trust settings inconsistent and returns status 1. Python's `credit_order="first"` derives coherent defaults automatically. There is no public repair-lane thinning option.

`RosaApproximationOptions` has ABI version 1:

```c
typedef struct RosaApproximationOptions {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t sample_rows;
    int32_t importance;
    int32_t sample_channels;
    int32_t reserved;
    uint64_t seed;
} RosaApproximationOptions;
```

| Approximation field | Default | Meaning |
|---|---:|---|
| `sample_rows` | 32 | Positive int32 complete-residual-row budget per stream |
| `importance` | 1 | 1 = gradient-norm probabilities with a uniform floor; 0 = uniform temporal strata |
| `sample_channels` | 0 | 0 = row estimator with all channels; 1–64 = fixed-size PPS of eligible centered channels |
| `reserved` | 0 | Leave at the initialized value |
| `seed` | 0 | Additional sampling salt, separate from joint proposal randomness |

For `sample_channels > 0`, the complete structural algorithm runs on the weighted feature sample and `sample_rows` is unused. The empty feature remains complete. If every eligible channel fits the budget, no channel sampling is needed. In the row estimator, complete deletion and one-symbol control precede stratified residual evaluation; large/full budgets use complete structural evaluation. A budget is not a universal error tolerance.

The Python names are `backward_mode="approximate"`, `approximation_rows`, and `approximation_channels`. Complete mode is `backward_mode="exact"`; no statistical sample is requested in that mode. See [the estimator derivation](randomized-backward.md).

## Arrays, bounds, and ownership

All arrays are contiguous and row-major. Floating arrays use IEEE binary64.

| Array | Shape | Type |
|---|---|---|
| Q/K/V logits, output, upstream gradient, Q/K/V logit gradients | `[B,G,T,D]` | `double` |
| Empty embedding, signed-payload scale, their gradients | `[G,D]` | `double` |
| Packed Q/K | `[B,G,T]` | `uint64_t` |
| V bits | `[B,G,T,D]` | `uint8_t` |
| Match lengths and endpoints | `[B,G,T]` | `int32_t` |
| Forward counters | `[B*G, ROSA_FORWARD_COUNTERS]` | `uint64_t` |
| Training counters | `[B*G, ROSA_TRAINING_COUNTERS]` | `uint64_t` |
| Approximation counters | `[B*G, ROSA_APPROXIMATION_COUNTERS]` | `uint64_t` |
| Detailed structural counters | `[B*G, ROSA_SUBQUADRATIC_COUNTERS]` | `uint64_t` |

Supported shapes have `B ≥ 0`, `G > 0`, `0 ≤ T < 2^28`, `1 ≤ D ≤ 64`, `T*D ≤ INT32_MAX`, and `B*G ≤ INT32_MAX`. Empty streams are supported. Bit 63 is a normal packed coordinate. The caller allocates outputs and keeps all input buffers alive during the call; use distinct writable output storage.

Backward must use packed codes, logits, embeddings, and options from the same forward. Approximate calls also receive that forward's `saved_lengths` and `saved_routes`; do not substitute or mutate a cache from another model evaluation. The NumPy adapter owns snapshots of retained floating inputs, and the Torch adapter saves the state automatically. Embedding gradients sum across batch streams for each group.

Optional `pair_codes` and `pair_costs` may be null. Their shapes are `[B*G,joint_pairs,2]` and `[B*G,joint_pairs,4]`. Only the first `selected_pairs` records of each stream are valid. Codes below `T*D` identify K bits; subtract `T*D` for Q codes, then divide by D for position and use modulo D for bit index. Costs are `C_i,C_j,C_ij,kappa`.

## Complete and approximate calls

After ordinary forward, complete backward uses the saved hard state:

```cpp
int status = rosa_training_backward(
    &options, B, G, T, D, q, k, v_bits,
    q_logits, k_logits, v_logits, grad_y, emb0, emb1,
    grad_q, grad_k, grad_v, grad_emb0, grad_emb1,
    training_counts, nullptr, nullptr);
```

To request residual-row estimation explicitly:

```cpp
RosaApproximationOptions approximation;
rosa_approximation_defaults(&approximation);
approximation.sample_rows = 32;
approximation.sample_channels = 0;
int status = rosa_training_backward_approx(
    &options, &approximation, B, G, T, D, q, k, v_bits,
    q_logits, k_logits, v_logits, grad_y, emb0, emb1,
    grad_q, grad_k, grad_v, grad_emb0, grad_emb1,
    training_counts, nullptr, nullptr,
    lengths, routes, approximation_counts);
```

These sketches assume the named arrays have already been allocated with the layouts above. The [standalone example](../examples/native_training.cpp) is a complete runnable program. For NumPy, call `training_forward_numpy(...)` and pass its owned cache unchanged to `training_backward_numpy(cache, grad_y)`.

## Accuracy, seeds, and training integration

Complete mode includes every finite-credit term. String comparisons, causal cutoffs and longest/latest ties are exact. Batched binary64 sums and FFT can round differently from a direct historical reduction; FFT is part of the complete construction, not a statistical omission. For the same inputs, rule and other seeds, turning the verified query cache on/off preserves this engine's numerical result bitwise.

Approximate mode adds conditional sampling error. It settles each evaluated route maximum completely, retains V/embedding gradients, and restores chosen joint corners with complete internal evaluations when necessary. Those edited-route replays do not run the surrounding model or loss. The Q/K diagonal is an explicit local model, not the true downstream Hessian.

The adapter freezes options and seeds during forward. Streams receive deterministic uint64 seed salts; cache on/off preserves proposal randomness and framework RNG consumption. Row/channel sampling uses a local generator separate from the proposal generator. Fixed seeds give reproducibility within the same implementation, not identical trajectories across numerical environments.

Trust and proposal decisions depend on the incoming gradient, so separate backwards on different losses need not add to one backward on their combined loss. Positive loss scaling is supported; hard-credit double autograd is not. The Torch adapter propagates nonfinite incoming gradients so ordinary GradScaler overflow skipping works. CPU matching entails transfer/synchronization for CUDA tensors.

## Return codes and diagnostics

| Status | Meaning |
|---|---|
| 0 | Success |
| 1 | Invalid shape, option, ABI, or required pointer |
| 2 | Allocation failure |
| 3 | Numerical/core execution failure |
| 4 | Nonfinite required input |

Use the header constants rather than literals when allocating counters: forward has 4 entries, training 35, approximation 40, detailed structure 22. The first 22 training entries are the structural-engine counters, followed by one reserved entry; indices 23–34 retain training-level diagnostics such as selected pairs, joint replay count, proposal work, gradient elements, and engine calls. The detailed alias exposes the same structural counters separately.

The authoritative names are `TRAINING_COUNTER_NAMES`, `SUBQUADRATIC_COUNTER_NAMES`, and `APPROXIMATION_COUNTER_NAMES` in [native_training.py](../conv_rosa_transformer/native_training.py). Structural counters cover AP pieces/events, Q heavy/light work, long anchors, short dictionaries, correlations, range blocks/fringes, cache probes/hits, and sampled channels. Workspace accounting is diagnostic allocated-buffer accounting, not a measurement of process peak RSS.

Approximation diagnostics distinguish the residual population, selected rows, complete residual evaluation, extra joint replays, and channel sampling. A complete-residual flag is not a bitwise-compatibility flag: complete numerical grouping still has the stated floating contract. Counters should accompany [measured time and error](subquadratic-results.md), not substitute for them.
