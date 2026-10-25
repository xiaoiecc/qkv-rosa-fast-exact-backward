# Native training and exact randomization: results and limits

[中文](native-results.zh-CN.md) · [Integration and derivation](native-training.md) · [Randomization proof](exact-randomized.md)

This round freezes batched C++ training and an exact randomized root join.
The learning rule retains the previous diameter-scaled diagonal trust model
and one independent joint-bit block; original first-order credit remains
selectable. `randomized` changes discovery, never drops credit terms, and
defaults to `False` because several small or direct-query cases do not benefit.

## Three timing scopes

| Scope | Included | Excluded |
|---|---|---|
| Credit core | All per-call indexes, Q/K/V credits, routes, accumulation; reused native buffers | Python validation, logit mapping, joint proposals, projections, optimizer |
| Hard component | Packing, forward, metric, proposals, singleton credits, joint replay, logit/embedding gradients, adapter conversions | Dense Q/K/V/O projections, downstream model, optimizer |
| Full layer | Hard component plus dense Q/K/V/O projections and their gradients | Downstream model, task loss, optimizer |

The environment was Intel Xeon Platinum 8573C, GCC 13.3, Linux, PyTorch
2.14.1+cpu and NumPy 2.3.5, with one pinned CPU core and one Torch thread.
Compilation and warmup are excluded. Credit-core rows have 11 alternating
samples; component rows have seven. Forward, backward and their sum each
have their own median, so medians need not add. Raw samples show substantial
variation; small ratios are not automatically algorithmic improvements.

The previous research baseline is `f6d7c4b73d7040b1b33860f5a2649c3cd3036cc5`,
preserved in the downloadable delivery. It is not an online repository
commit. The patch application base is online `4248b46`. Earlier online
repository measurements remain separate in benchmarks.

## 1. Current full-layer cost

These rows include all dense projections. All configurations have the same
hard-forward semantics. The comparison column uses the separately measured
median of the complete **forward plus backward**.

| T | D | B / G | First backward off / on (ms) | Second backward off / on (ms) | Second on forward+backward (ms) | f6 / second on |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 8 | 4 / 2 | 2.198 / 2.094 | 3.039 / 3.373 | 4.032 | 4.48× |
| 1,024 | 8 | 1 / 1 | 2.556 / 2.272 | 2.714 / 2.935 | 4.573 | 1.48× |
| 1,024 | 16 | 2 / 4 | 17.100 / 15.415 | 19.910 / 21.679 | 26.326 | 1.96× |
| 8,192 | 8 | 1 / 1 | 23.084 / 23.940 | 30.719 / 27.678 | 31.318 | 1.57× |
| 8,192 | 16 | 1 / 1 | 21.016 / 17.482 | 23.503 / 26.858 | 30.693 | 1.48× |
| 8,192 | 64 | 1 / 1 | 73.476 / 66.069 | 77.193 / 70.939 | 83.044 | 1.13× |

Native second-order on is **1.13–4.48×** faster than f6 over these six shapes.
The random toggle alone, across both orders, has full-layer off/on ratios of
**0.723–1.349×**, including regressions. Native integration and exact-search
improvements must not be conflated. The D64 second-order layer takes
83.044 ms for both passes, including 70.939 ms backward; it is a different
input and scope from the 12.033 ms root-join credit case below.

The hard-component scope isolates cost outside dense projections:

| T | D | B / G | First backward off / on (ms) | Second backward off / on (ms) | Second on forward+backward (ms) | f6 / second on |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 8 | 4 / 2 | 3.263 / 2.612 | 3.376 / 3.896 | 4.461 | 4.00× |
| 1,024 | 8 | 1 / 1 | 3.185 / 2.230 | 2.868 / 3.409 | 3.873 | 1.85× |
| 1,024 | 16 | 2 / 4 | 15.810 / 16.387 | 20.640 / 19.981 | 22.609 | 2.11× |
| 8,192 | 8 | 1 / 1 | 26.599 / 27.891 | 31.165 / 32.325 | 34.590 | 1.37× |
| 8,192 | 16 | 1 / 1 | 19.880 / 20.343 | 27.900 / 24.236 | 26.537 | 1.90× |
| 8,192 | 64 | 1 / 1 | 65.322 / 56.938 | 68.022 / 62.598 | 69.851 | 1.18× |

[Complete report](../benchmarks/results/native_component.json). Every shape
checks current on/off outputs and all parameter gradients for bitwise equality,
and baseline forward outputs are bitwise equal. Both current and frozen
source hashes remain unchanged throughout timing.

## 2. What exact randomization removes

Wide words previously required a root lookup for every bit of every distinct
Q word. Random omitted-block indexes find all possible Hamming-one neighbors,
full XOR checks verify them, and large buckets fall back to exact bit probes.
Both toggle settings use the new sparse mask/CSR root representation.

For T8192 D64 with planted one-bit repairs, 524,288 full root probes become
16,384 bucket queries and 2,048 member checks. SAM preparation falls from
7.960 to 1.713 ms. The current complete credit core falls from 20.094 ms off
to 12.033 ms on: **1.670× from the toggle alone**, or 1.814× against f6's
21.824 ms. Only 2,048 root targets are stored, versus 524,288 logical dense
entries. Entry counts are not peak allocation or RSS; vector capacity and
O(TD) returned gradients remain.

All 15 rows follow. `wide` contains planted repairs; `narrow` varies only eight
bits. Input-family names belong to the generator, not runtime dispatch.

| Input | T | D | Credit model | f6 (ms) | Native off (ms) | Native on (ms) | f6 / on | off / on |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| random | 1,024 | 8 | linear | 1.416 | 1.263 | 1.306 | 1.085× | 0.967× |
| random | 8,192 | 8 | linear | 26.656 | 26.835 | 23.974 | 1.112× | 1.119× |
| random | 2,048 | 16 | linear | 1.756 | 1.605 | 1.518 | 1.157× | 1.057× |
| random | 2,048 | 16 | diagonal | 1.573 | 1.495 | 1.533 | 1.026× | 0.975× |
| random | 8,192 | 16 | linear | 12.039 | 10.217 | 10.608 | 1.135× | 0.963× |
| random | 8,192 | 32 | linear | 11.624 | 11.351 | 7.121 | 1.632× | 1.594× |
| wide | 8,192 | 64 | linear | 21.824 | 20.094 | 12.033 | 1.814× | 1.670× |
| wide | 8,192 | 64 | diagonal | 18.437 | 16.659 | 10.435 | 1.767× | 1.596× |
| narrow | 8,192 | 64 | linear | 44.476 | 39.129 | 43.731 | 1.017× | 0.895× |
| correlated | 8,192 | 64 | linear | 10.221 | 9.608 | 8.330 | 1.227× | 1.153× |
| asymmetric | 8,192 | 16 | linear | 8.508 | 7.818 | 7.805 | 1.090× | 1.002× |
| shadow | 2,048 | 64 | linear | 14.977 | 14.670 | 14.754 | 1.015× | 0.994× |
| shadow | 2,048 | 64 | diagonal | 16.038 | 15.615 | 15.510 | 1.034× | 1.007× |
| periodic | 4,096 | 4 | linear | 47.999 | 45.635 | 49.376 | 0.972× | 0.924× |
| all_same | 8,192 | 8 | linear | 3.774 | 3.618 | 3.692 | 1.022× | 0.980× |

The complete f6/on range is **0.972–1.814×**, including the periodic regression.
The planner uses direct queries for D8 and some periodic cases; unchanged
work and small timing differences are not new algorithmic gains. Every row
has bitwise-identical Q/K/V credits, lengths, routes and the original 23
structural counters against f6.

[Raw data](../benchmarks/results/exact_randomized.json). Later test-only
coverage changed the bridge module, so the actual timed C++ probe is retained
as a [snapshot](../benchmarks/snapshots/exact_randomized_probe.cpp) with a
[provenance record](../benchmarks/results/exact_randomized_provenance.json).
Production hashes and raw timings are unchanged.

## 3. Why residual thinning was rejected

Common deletion plus repair residual gives an unbiased whole-lane estimator.
We gave it an unavailable practical advantage: knowledge of every residual's
energy, allowing optimal importance probabilities at a fixed budget. At half
budget, relative RMS error still reached 84.7% for random D8, 70.3% for random
D16 and 97.2% for wide D64. A 10% error target required all 8/8 and 16/16 lanes,
and at least 63/64 in the wide case.

This is evidence against low-noise, large savings from whole-lane thinning on
these inputs, not a lower bound against all randomized algorithms. Exact
randomization changes discovery instead. See the [derivation](exact-randomized.md)
and [energies/results](../research/results/native_random_noise_ceiling.json).

## 4. Ordinary training: an exact toggle and a numerical migration regression

Two composed models were trained for 100 updates with fresh seeds 901–903:
fixed-data MSE retrieval and fresh-data cross-entropy retrieval. Both include
projections, a task head, loss and Adam updates, using one model forward and
one backward per update. Hyperparameters were not retuned after observing
migration results.

| Task, mean of 3 seeds | f6 | Native second off | Native second on |
|---|---:|---:|---:|
| MSE training loss | 0.068074 | 0.009362 | 0.009362 |
| MSE held-out loss | **0.278856** | **0.528012** | **0.528012** |
| MSE 100 updates | 5.902 s | 0.360 s | 0.331 s |
| Fresh CE training loss | 0.000641 | 0.000879 | 0.000879 |
| Fresh CE held-out loss | 0.000558 | 0.000788 | 0.000788 |
| Fresh CE 100 updates | 5.415 s | 0.312 s | 0.310 s |

For all six native on/off pairs, every update's parameters and loss, and final
framework RNG state, are bitwise equal. An additional contract test covers
both credit orders, two sizes and downstream Dropout: outputs, gradients,
weights and RNG remain bitwise equal. The roughly 17× small-toy speedup chiefly
removes Python loops over many short batch/group streams. D4 root queries
fall back to direct probing; this is not a 17× randomization result.

**Native MSE held-out loss regressed.** Lower training loss did not imply
better generalization. Fresh CE accuracy is 100% in both implementations,
but the native cross-entropy values are also slightly worse. Three seeds
provide limited evidence, and binary64 arithmetic must not be advertised as
an automatic improvement in learning quality.

### Tracing the first divergence

For the worst held-out MSE seed903, initial forward outputs and selected pair
identities are identical; joint-corner cost differences are at most 6.94e-18,
and the largest parameter-gradient difference is 1.49e-8.

Near-zero gradients matter. One emb0 coordinate receives −2.794e-9 in the old
code and −9.313e-10 in native code. With Adam epsilon 1e-8 and learning rate
0.03, the first updates are approximately 0.006551 and 0.002556: a parameter
difference of 0.003996. After five updates, before the sixth update, 31 Q bits first disagree, at old
logits exactly zero versus native logits around 7.44e-14 or 1.49e-13. Since
the threshold is `z > 0`, these become different hard bits. Pair identities
and framework RNG were identical up to that point.

The old singleton mapping and some reductions used Torch float32; native
code uses binary64 intermediates before casting to tensor dtype. This trace
shows small numerical differences amplified by near-zero Adam updates and
hard thresholds. It is not evidence of missed randomized candidates, and old
training trajectories are not promised to be bitwise reproducible. When
active rows exceed the proposal-row cap, NumPy PCG64 versus native
mt19937_64 can also change the old fixed-seed proposal sequence; that did not
cause this trace's divergence.

Evidence:

- [Summary and every seed](../research/results/native_random_training_summary.json)
- [Frozen runs](../research/results/native_random_training_baseline.json) and [all native runs](../research/results/native_random_training_final.json)
- [Mode/RNG contract](../research/results/native_random_mode_contract.json)
- [First-divergence trace](../research/results/native_random_roundoff_trace.json)

## 5. Correctness and unmeasured scope

The [native validation report](../research/results/native_cpp_validation.json)
records 88 passing/skip-counted tests with one CUDA skip, including independent
brute force and structural oracles, tolerance-based migration checks, a real
two-bit barrier, ordinary optimizers, checkpoint, GradScaler and state_dict.
The added four-block and sampled/forced-trajectory tests also passed separately;
counts overlap. Production sources remained frozen during all formal timing
and validation. The standalone C++ example needs no Python or Torch headers.

These are CPU measurements, not native GPU results. The diagonal is a chosen
local trust model, not the unknown downstream Hessian. Second-order surrogate
updates depend on g, are not additive for arbitrary incoming gradients, and
do not support double autograd. Ordinary single-backward training needs no
optimizer changes. Larger-model generalization, peak RSS and distributed
throughput remain unmeasured. Those limits do not weaken the exact-search
coverage proof, but they constrain the training claims supported here.
