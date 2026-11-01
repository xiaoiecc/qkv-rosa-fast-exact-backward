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

The default is **exact backward with randomized search**: C options use
`randomized=1`, and the Python layer uses `backward_mode="exact",
randomized=True`. Randomized search checks complete keys, preserves numerical
accumulation order, and falls back to direct lookup when appropriate. With
the same inputs, credit rule and seeds, turning this exact acceleration on or
off preserves the native gradient outputs bit for bit. Search-work counters
may differ. This contract does not describe the optional approximate API,
which may reorder floating-point sums or sample numerical corrections.

See the [randomized-backward tutorial](randomized-backward.md) for the
derivation and the [measurements and training results](randomized-results.md)
for speed, error and counterexamples. Both have linked Chinese editions.

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

Call `rosa_training_defaults(&options)` before setting individual fields.
**`RosaTrainingOptions` remains ABI version 1 with its original layout.** The
original `rosa_training_forward` and `rosa_training_backward` signatures are
unchanged. Approximation has a separate options struct and backward entry
point; no approximation fields were appended to `RosaTrainingOptions`.
Both option structs carry `abi_version` and `struct_size`; unsupported
versions or sizes return status 1.

| Field | Meaning |
|---|---|
| `credit_order` | 1 = original first-order singleton credits; 2 = diagonal trust plus independent joint blocks |
| `randomized` | 0 or 1; default 1. Answer-exact randomized root preparation; it never requests repair thinning or numerical approximation |
| `logit_mode` | 0 = oriented finite credit times sigmoid derivative; 1 = oriented identity mapping |
| `joint_pairs` | Maximum disjoint two-bit blocks per stream; default 1 |
| `joint_rows` | Maximum proposal query rows; default 32 |
| `active_rows` | 1 = nonzero upstream-gradient rows, 0 = evenly spaced input rows |
| `tau` | Positive sigmoid temperature; default 1 |
| `trust_diagonal` | Default 0.5, equivalent to gradient norm divided by output-set diameter |
| `proposal_seed` | Joint proposal seed, frozen for this forward/backward |
| `repair_seed` | Independent seed for exact root layout and optional explicit repair sampling |
| `repair_budget` | Default 0 = complete exact repair computation; positive = legacy research estimator, independently selected. Must be 0 with `rosa_training_backward_approx` |

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

The Torch adapter consumes two CPU Torch RNG draws when joint proposals,
explicit legacy repair sampling, or `backward_mode="approximate"` is enabled,
identically for exact acceleration on/off. First-order exact training without
joint proposals or repair sampling consumes no such RNG. Evaluation consumes
no proposal RNG. Checkpoint recomputation can therefore use the normal
framework RNG-preservation contract. Native row and channel sampling use a
local generator seeded from the stream's salted `repair_seed` XOR the
approximation option's `seed`; they do not consume the joint-proposal generator.

Joint-proposal row selection uses weighted exponential races with `mt19937_64`. For
more than `joint_rows` nonzero rows, this changes the old NumPy `PCG64` seed
trajectory, while preserving the stated sampling law. When all active rows
fit the budget, rows and native joint proposals are unchanged. All selected
joint corners are evaluated exactly under the frozen local trust model.

### Separate approximation options

Initialize the additional struct with
`rosa_approximation_defaults(&approximation)`. Its public declaration is:

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

`ROSA_APPROXIMATION_ABI_VERSION` is **1**, and this layout has
`sizeof(RosaApproximationOptions) == 32`. Initialize it through the defaults
function so `struct_size` uses the actual C declaration, rather than
hard-coding the size in a binding.

| Field | Default | Valid values and meaning |
|---|---:|---|
| `abi_version` | 1 | Must equal `ROSA_APPROXIMATION_ABI_VERSION` |
| `struct_size` | 32 | Must equal `sizeof(RosaApproximationOptions)` |
| `sample_rows` | 32 | Positive int32 budget for complete correction rows per stream; capped by the certified population size |
| `importance` | 1 | 0 = uniform temporal strata; 1 = gradient-norm weighting with a uniform probability floor inside each stratum. Controls row sampling |
| `sample_channels` | 0 | 0 = retain all FFT channels; 1–64 = explicit PPS budget for eligible nonzero signed channels, capped by their count |
| `reserved` | 0 | Leave at the initialized value |
| `seed` | 0 | Additional sampling salt, independent of joint-proposal randomness |

**A zero channel budget means all channels, not zero work.** It does not turn
off row sampling: a cost planner chooses shared-boundary numerical contraction
or the controlled residual-row calculation. `sample_rows` applies to the row
path; `sample_channels` applies to the FFT path. The full constant control and
sparse one-hot channels are retained even with a positive channel budget.
Channel PPS has its own probability construction; `importance=0` does not
change it to uniform channel sampling.

The Python names are `approximation_rows` and `approximation_channels`, with
the same defaults of 32 and 0. Approximate mode is explicitly selected through
`backward_mode="approximate"`. Exact mode continues to call the original C
entry point. Increasing a budget spends more work on the corresponding
correction, but neither budget is a universal error tolerance; see the
[first-order cancellation cases](randomized-results.md).

### Calling approximate backward

Use the ordinary forward once, retain its packed codes **and its
`lengths`/`routes` arrays**, then call `rosa_training_backward_approx` instead
of `rosa_training_backward`. The following sketch assumes the dimensions and
named pointer buffers have already been allocated as listed below:

```cpp
RosaTrainingOptions options;
rosa_training_defaults(&options);  // Exact randomized search; repair_budget=0.

RosaApproximationOptions approximation;
rosa_approximation_defaults(&approximation);  // 32 rows, all FFT channels.
// approximation.sample_channels = 8;        // Explicitly more aggressive.

std::vector<uint64_t> approximation_counts(
    size_t(B) * G * ROSA_APPROXIMATION_COUNTERS);

int status = rosa_training_backward_approx(
    &options, &approximation, B, G, T, D,
    q, k, v_bits, q_logits, k_logits, v_logits,
    grad_y, emb0, emb1,
    grad_q, grad_k, grad_v, grad_emb0, grad_emb1, counters,
    nullptr, nullptr,  // Optional pair-code and corner-cost traces.
    saved_lengths, saved_routes, approximation_counts.data());
```

`saved_lengths` and `saved_routes` are the unmodified outputs of that same
forward, with layout `[B,G,T]`. They are inputs to this backward call, not
scratch outputs. Their causal bounds are checked by the native API. For
`B*G>0`, the approximation counter buffer is required, including when `T=0`, and has
**40 uint64 values per stream**. The ordinary 35-value backward counter buffer
is still required separately. Passing a null approximation-options pointer,
invalid approximation fields, or a nonzero legacy `repair_budget` returns
status 1.

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
| Approximation counters, only for `rosa_training_backward_approx` | `[B*G,40]` |
| Optional pair codes | `[B*G,joint_pairs,2]` |
| Optional corner costs | `[B*G,joint_pairs,4]` |

The caller owns every buffer and keeps it alive during the call. Forward emits
packed Q/K/V to reuse in backward. Backward must receive those same hard codes,
logits, and embeddings; replacing them independently invalidates the cache
contract. The NumPy `TrainingCache` independently copies the logits, embeddings
and options, including the separate approximation options when present, so
later changes to caller storage do not alter its backward.
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

`rosa_training_backward_approx` uses the same hard forward, frozen local
credit model and logit mapping. Its planner may combine shared corridor sums
with FFT, whose floating reduction order differs from the exact engine, or
sample complete residual rows around a fully computed control. A positive
channel budget adds fixed-size PPS sampling of signed numerical channels.
Route maxima are resolved before these numerical samples. V and embedding
gradients remain exact, and selected joint blocks retain complete internal
matcher evaluations of their corners. These internal matcher passes do not
replay the surrounding model or require an extra user backward call.

This approximate entry point does **not** promise bitwise Q/K gradients or a
uniform relative-error bound. Conversely, `randomized=1` on the original exact
entry point enables none of FFT reordering, row sampling or channel sampling.

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

### Approximation counters

`ROSA_APPROXIMATION_COUNTERS` is **40**. The separate buffer uses the following
indices; names match `APPROXIMATION_COUNTER_NAMES` in the NumPy binding:

| Index | Counter |
|---:|---|
| 0 | `sampled_rows` |
| 1 | `row_endpoints` |
| 2 | `row_lce_queries` |
| 3 | `row_onebit_pairs` |
| 4 | `row_q_touched` |
| 5 | `row_k_touched` |
| 6 | `row_score_components` |
| 7 | `row_deletion_updates` |
| 8 | `rebuilt_baseline_rows` |
| 9 | `row_index_materializations` |
| 10 | `evaluated_rows` |
| 11 | `control_neighbor_probes` |
| 12 | `control_score_components` |
| 13 | `control_intervals` |
| 14 | `repair_births` |
| 15 | `lce_history_rows` |
| 16 | `active_rows` |
| 17 | `uncertified_rows` |
| 18 | `certificate_bucket_queries` |
| 19 | `certificate_member_checks` |
| 20 | `certificate_full_pair_probes` |
| 21 | `certificate_hash_probes` |
| 22 | `certificate_distinct_pairs` |
| 23 | `certificate_indexes` |
| 24 | `population_rows` |
| 25 | `selected_rows` |
| 26 | `row_budget` |
| 27 | `complete_residual_streams` |
| 28 | `extra_joint_replays` |
| 29 | `certified_zero_rows` |
| 30 | `importance_streams` |
| 31 | `full_deletion_calls` |
| 32 | `spectral_groups` |
| 33 | `spectral_queries` |
| 34 | `spectral_rows` |
| 35 | `spectral_transforms` |
| 36 | `spectral_columns` |
| 37 | `spectral_channels` |
| 38 | `spectral_exact_fallbacks` |
| 39 | `spectral_fallback_rows` |

Indices 0–15 describe selected-row work and the one-symbol control; 16–23
describe exact support certificates; 24–31 describe the resulting population,
sampling and additional replays. The FFT branch also sets index 27, so it is
**not** a bitwise-exactness flag. Index 31 counts separate full-deletion-only
calls in the residual-row path.

For the spectral counters, `spectral_rows` counts represented active interval
rows and can exceed T because intervals overlap. `spectral_channels` counts
available numerical channels before optional channel sampling, whereas
`spectral_transforms` counts the transforms actually executed. Exact recovery
may use either direct expansion or a proven zero-payload identity; therefore
`spectral_exact_fallbacks` can be nonzero while `spectral_fallback_rows` is zero.

## Validation scope

`tests/test_native_training.py` compares the native complete rule with the
frozen previous Python calculation, including both mappings, empty inputs,
multiple groups, bit 63, first/second order, and the real zero-signal joint
barrier. Double comparisons use `2e-11` absolute/relative tolerance; float32
mapping uses `2e-6`. On/off acceleration comparisons require bitwise equality.
Normal optimizer, GradScaler, checkpoint, and serialization tests exercise
the production layer. Historical Python-mocking checks explicitly load a
test-only snapshot; production never selects it as a backend.
