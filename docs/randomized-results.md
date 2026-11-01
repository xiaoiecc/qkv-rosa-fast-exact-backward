# Randomized backward: measurements and limits

[中文](randomized-results.zh-CN.md) · [Algorithm tutorial](randomized-backward.md) · [Training API](native-training.md)

The final implementation produces large savings when many repair corridors share numerical work. It also has inputs where approximation is slower or too noisy. This report keeps both. The default remains **exact backward with randomized search**; approximate backward is an explicit choice.

## Settings and provenance

| Name here | Setting | Meaning |
|---|---|---|
| Exact | `backward_mode="exact", randomized=True` | Complete finite credits; randomness changes search work, not the answer |
| A0 | `backward_mode="approximate", approximation_rows=32, approximation_channels=0` | Sample certified row residuals, or use all spectral channels when the cost planner selects contraction |
| A8 | As A0, with `approximation_channels=8` | Additionally sample spectral channels; an advanced opt-in |

A0's zero channel budget means **all channels**, not zero work. It does not disable row sampling. Neither approximate setting promises a universal relative-error bound or bitwise equality. Both credit orders use the same API. “Exact” refers to the selected finite-credit rule; the diagonal second-order rule is a surrogate, not the true Hessian of an arbitrary surrounding model.

**Online** is commit `b510fd9479946f10e46ced1a6265695cbc131d0a`. **P1** is the frozen intermediate row sampler at `5e3e9f81965242492ae5a3870efd3cfbbca4bf9e`. Current exact, A0 and A8 all use final native SHA-256 `a732d50fd3f12342320f19306b6a5292b3d6aa829ff2c3a8b43d64e28dfc9cb5`. The [manifest](../research/results/randomized_final_manifest.json) records the evidence and source identities; earlier experiments are labeled separately below.

## What each timer measures

| Scope | Included | How to interpret it |
|---|---|---|
| Credit core | Route/credit computation inside a private probe | Used during development; not a complete backward or training-step speedup |
| Native backward | Public NumPy/native call, validation/allocation, metric/proposals, credits, joint replays, logit mapping, V/embedding accumulation and trace reporting | The complete 26-case tables below |
| Ordinary training step | Zeroing gradients, model forward and projections, loss, backward, Adam | The separate training table below |

Native timings exclude the model forward, loss, optimizer and compilation. They are medians of **nine interleaved repetitions**, on Linux x86-64 with affinity to logical CPU 8. Training used CPU 2, with OpenBLAS, OMP, MKL and Torch thread counts explicitly limited to one. It excludes data generation, evaluation, compilation and diagnostic logging. These CPU measurements make no GPU throughput claim.

## Complete native backward results

All times are milliseconds. Error is the median relative L2 error of the **concatenated raw Q/K logit gradients**, expressed as a percentage. `≈0` means relative error below `1e-10`; it does not assert bitwise equality. Current exact matched online Q/K/V/embedding outputs **byte for byte in every case**. Approximate V and embedding gradients also matched byte for byte; selected joint codes matched and their costs passed the recorded `1e-12` tolerance. Every timing sample and error draw is in the [raw native report](../benchmarks/results/random_final_native.json).

Rows marked † use actual upstream model adjoints: context has B×G = 4×4; both prefix captures have 1×2. Other cases have 1×1 and synthetic adjoints. `wide` uses full 64-bit words with planted causal one-bit neighbors; `narrow` stores an eight-bit vocabulary in 64-bit words. `sparse_periodic` has a nonzero adjoint only at the last row. The same capture is evaluated under both credit orders.

### Diagonal second order

| Input: T / D | Current exact ms | A0 ms | A8 ms | Online ms | P1 ms | QK error %: A0 / A8 |
|---|---:|---:|---:|---:|---:|---:|
| random 128 / 8 | 0.252 | 0.291 | 0.287 | 0.270 | 0.299 | ≈0 / ≈0 |
| random 8192 / 8 | 21.613 | 18.192 | 18.342 | 22.784 | 19.134 | 229.10 / 229.10 |
| random 8192 / 16 | 10.916 | 12.932 | 13.032 | 14.171 | 13.070 | ≈0 / ≈0 |
| wide 8192 / 64 | 25.079 | 25.236 | 25.820 | 35.011 | 26.117 | ≈0 / ≈0 |
| narrow 8192 / 64 | 51.041 | 58.666 | 59.771 | 66.064 | 67.376 | 89.95 / 89.95 |
| all_same 8192 / 8 | 4.028 | 4.388 | 4.405 | 5.621 | 5.470 | 0 / 0 |
| periodic 4096 / 4 | 43.488 | 41.056 | 38.954 | 47.570 | 40.705 | 5.73 / 5.73 |
| shadow 8192 / 1 | 509.272 | 8.758 | 9.052 | 439.798 | 8.709 | ≈0 / ≈0 |
| shadow 2048 / 64 | 21.687 | 7.241 | 6.727 | 22.010 | 12.010 | ≈0 / ≈0 |
| sparse_periodic 4096 / 4 | 4.093 | 2.948 | 2.852 | 36.748 | 3.167 | ≈0 / ≈0 |
| context† 512 / 2 | 17.044 | 13.509 | 13.503 | 17.904 | 15.006 | 102.63 / 102.63 |
| prefix† 4096 / 8 | 264.940 | 12.432 | 12.456 | 272.466 | 12.898 | ≈0 / ≈0 |
| prefix64† 2048 / 64 | 302.257 | 34.084 | 25.005 | 304.820 | 31.255 | ≈0 / 0.81 |

### First order

| Input: T / D | Current exact ms | A0 ms | A8 ms | Online ms | P1 ms | QK error %: A0 / A8 |
|---|---:|---:|---:|---:|---:|---:|
| random 128 / 8 | 0.157 | 0.191 | 0.172 | 0.161 | 0.184 | ≈0 / ≈0 |
| random 8192 / 8 | 19.508 | 13.239 | 13.536 | 19.763 | 13.961 | 337.23 / 337.23 |
| random 8192 / 16 | 8.382 | 9.550 | 9.412 | 11.049 | 9.445 | ≈0 / ≈0 |
| wide 8192 / 64 | 22.592 | 18.372 | 18.487 | 32.906 | 18.185 | ≈0 / ≈0 |
| narrow 8192 / 64 | 47.556 | 42.935 | 41.120 | 56.268 | 51.836 | 166.62 / 166.62 |
| all_same 8192 / 8 | 3.256 | 3.605 | 3.501 | 4.566 | 4.613 | 0 / 0 |
| periodic 4096 / 4 | 40.998 | 35.077 | 34.850 | 41.470 | 35.697 | 246.99 / 246.99 |
| shadow 8192 / 1 | 476.118 | 7.915 | 7.746 | 414.910 | 7.635 | ≈0 / ≈0 |
| shadow 2048 / 64 | 21.525 | 6.122 | 6.029 | 21.355 | 10.799 | ≈0 / ≈0 |
| sparse_periodic 4096 / 4 | 3.543 | 2.191 | 2.200 | 36.178 | 2.399 | ≈0 / ≈0 |
| context† 512 / 2 | 15.333 | 9.785 | 9.541 | 16.824 | 10.794 | 194.63 / 194.63 |
| prefix† 4096 / 8 | 310.518 | 12.288 | 12.469 | 304.121 | 12.331 | ≈0 / ≈0 |
| prefix64† 2048 / 64 | 233.646 | 25.249 | 16.939 | 280.638 | 23.919 | ≈0 / 0.98 |

The large gains are concentrated. For the D8 prefix capture, A0's second-order backward takes **12.432 ms** versus online's 272.466 ms. For D64 prefix, it takes **34.084 ms**, or **25.005 ms** with A8, versus 304.820 ms online. A8 introduces 0.811% raw QK error there, and **3.341% Q-only error**.

The counterexamples matter just as much. Random D8 has 229% second-order raw QK error at 32 rows; context has 103%. First-order periodic input has 247% combined error and **713% Q-only error**. Those are not accurate gradient approximations. A0 is slower than current exact on random D16, narrow second order and all-same inputs. Current exact itself regresses on shadow D1: 509.272 ms versus 439.798 ms online in second order. Across all cases, exact/online speedups range from 0.864× to 10.212×; A0/online ranges from 0.847× to 52.422×. The maximum is not a general speed guarantee.

When the planner does not use spectral contraction, A8 and A0 use the same row estimator. Small timing differences on those rows do not show a channel-sampling benefit. Full-channel FFT is itself a change in floating-point evaluation order, not an output-exact randomized algorithm.

## Why eight channels is not the default

A separate final-public-API probe uses 16 draws per setting on a fixed D64 prefix capture. `balanced` keeps its original adjoint. `tangent` removes its component along the current output, weakening the exact constant control; `nine_tangent` also concentrates energy in nine channels. These are local gradient diagnostics, not training curves. The table measures actual shared projection parameters after float32 accumulation and the input gradient sent to earlier layers.

| Adjoint / order | Q parameter error % | Q cosine | QK parameter error % | Input error % |
|---|---:|---:|---:|---:|
| balanced / first | 0.813 | 1.000 | 0.198 | 0.086 |
| balanced / second | 0.704 | 1.000 | 0.171 | 0.072 |
| tangent / first | 286.434 | 0.370 | 17.549 | 5.882 |
| tangent / second | 0.434 | 1.000 | 0.106 | 0.044 |
| nine_tangent / first | 55.740 | 0.871 | 3.323 | 0.996 |
| nine_tangent / second | 0.278 | 1.000 | 0.068 | 0.031 |

The first-order tangent failure is severe: Q parameter error is **286%**, with cosine **0.370**. Concatenating the unchanged K contribution reduces the reported error to 17.5% and raises cosine to 0.985, hiding much of the damage. Balanced second order, by contrast, has 0.171% shared QK parameter error and 0.072% input error. Both findings belong in the decision about a channel budget.

The [full A8 controls](../research/results/training_round_final_channel_controls.json) also retain one- and nine-dominant-channel cases. V and embedding outputs remain bytewise equal; joint checks pass. Four [A0 controls](../research/results/training_round_final_channel_controls_full.json)—balanced/tangent, first/second—have **zero float32 parameter and input-gradient error** on this capture. This supports keeping all spectral channels by default; it does not make the separate row-sampling branch noise-free.

## Ordinary training, with the final runtime

This deliberately structured prefix-copy task uses B1, G2, D64, Adam at learning rate 0.003, and fresh data at every update. Each run performs **20 ordinary updates, each with one model forward and one backward**. Loss is the row mean of `0.5 * sum(error**2)`, not the per-coordinate mean squared error. Each cell gives total training-step seconds and final held-out loss. Seed 1101 was used during discovery; 2401 and 2402 were held out from it.

| T / order / seed | Initial loss | Exact: seconds / loss | A0: seconds / loss | A8: seconds / loss |
|---|---:|---:|---:|---:|
| 2048 / second / 1101 | 27.06270 | 5.379 / 14.83426 | 0.830 / 14.75660 | 0.652 / 14.59788 |
| 2048 / second / 2401 | 27.94375 | 5.346 / 9.35802 | 0.867 / 9.35802 | 0.761 / 9.35389 |
| 2048 / second / 2402 | 27.22209 | 5.358 / 26.39198 | 0.793 / 26.39198 | 0.884 / 26.75267 |
| 2048 / first / 1101 | 27.06270 | 4.009 / 43.10059 | 0.903 / 43.51060 | 0.843 / 43.91100 |
| 4096 / second / 1101 | 27.07714 | 19.501 / 15.54690 | not measured | 1.530 / 15.54690 |

At T4096, A8 takes **1.530 seconds versus 19.501 seconds: 12.74× faster for the complete measured training step**. All 20 batch losses and hard-forward hashes match exact on that run, but Q/K parameters differ from the first update. The final held-out losses are equal. Only exact and A8 were measured at that length. This is a structural stress result over 20 updates, not evidence of general convergence or a universal model speedup.

At T2048 on discovery seed 1101, A8 gives 8.25× whole-step acceleration. P1 was also rerun: 0.830 seconds, final loss 13.77414. That historical method has its own frozen binary, not the final one. The suite contains 15 runs and 300 total updates, including that comparison.

There are two clear failures. On fresh second-order seed 2402, A8 is **slower than A0 and slightly worse in held-out loss**; progress is weak for all methods. Under the unretuned first-order configuration, every method raises loss from 27.06 to 43–44. Fast execution does not turn that into successful learning.

Sources: [discovery second order](../research/results/training_round_final_channels_prefix1101_second.json), [fresh second-order seeds](../research/results/training_round_final_channels_prefix_fresh_second.json), [first order](../research/results/training_round_final_channels_prefix1101_first.json), [T4096 second order](../research/results/training_round_final_channels_prefix4096_second.json). Each report records source/binary hashes and per-update evidence.

## Earlier training evidence stays earlier

The frozen P1 study ran 59 experiments and 17,916 updates. Three fresh seeds of context recall reached 100% nonempty-payload accuracy with exact and row32, but dense retrieval had an exact-baseline failure, and certified weighted-periodic parameter trajectories were identical, but approximation did not consistently run faster. Adding a trainable residual tanh MLP stem caused both exact and row32 to fail; removing it in a labeled isolation control let the exact baseline learn. Arbitrary deep-model stability therefore remains unverified.

An intermediate indexed-row implementation also matched P1's parameter updates and hard forwards across **600 paired context updates**. This was equality to P1, not to exact, and it predates the final frozen runtime. Its aggregate report is marked `SOURCE_CHANGED` because validation/bookkeeping changed during the first exact run. Its timings must not be presented as final frozen results. The manifest identifies these historical reports; full files and rejected experiments remain in the study bundle under `research/training_round_frozen_results.md` and `research/results/training_round_phase2b_context.json`.

## Reproduction and use

The [native benchmark script](../benchmarks/random_final_native.py) reruns all five methods from the recorded online/P1 checkouts and captured inputs supplied in the study bundle. The raw report includes nine samples, per-draw errors, work counters, source hashes, loaded-library hashes and the unchanged-source check. Use the [training guide](native-training.md) for installation and the unchanged `loss.backward(); optimizer.step()` loop.

Start with the default exact mode. If choosing approximation, use `backward_mode="approximate", approximation_rows=32, approximation_channels=0`; assess both Q-only and QK errors and the task's actual loss curve. A positive channel budget such as 8 is an explicit additional tradeoff. The [algorithm tutorial](randomized-backward.md) explains why the estimators have the stated expectation, and why that alone cannot guarantee a small error or a good optimizer trajectory.
