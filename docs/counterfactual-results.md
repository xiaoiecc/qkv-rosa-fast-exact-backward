# Exact backward compared with merged PR #8

Both checkouts ran on the same Intel Xeon Platinum 8573C, pinned to one logical CPU. The measured revisions were `881ab23` (reference) and `4c86a5a` (current). Each measurement is the median of 11 interleaved calls per revision after warmup. The comparison uses single-stream CPU data and verifies the source snapshots, inputs and proposal seeds. See the [algorithm and conditions](counterfactual-sweeps.md).

## Complete native backward

The 44 existing comparisons had a geometric-mean speedup of **1.67×** with 36 faster medians. The 28 structural comparisons had a geometric-mean speedup of **2.77×**, all faster. The largest observed slowdown was a **10.1% increase in time** on periodic input at length 1,024, width 4, second order. Speedup here divides the baseline median by the current median. All times are milliseconds.

| Input | T | Bits | First order, old → new | Speedup | Second order, old → new | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Many one-bit neighbors, few equal pairs | 8,192 | 8 | 156.65 → 7.87 | 19.92× | 156.01 → 11.67 | 13.37× |
| Interleaved runs (skyline) | 14,112 | 8 | 272.16 → 51.58 | 5.28× | 273.52 → 55.86 | 4.90× |
| Random binary | 4,096 | 1 | 23.62 → 6.95 | 3.40× | 25.02 → 8.45 | 2.96× |
| Delayed copy with three damaged blocks | 4,096 | 64 | 39.74 → 15.91 | 2.50× | 40.88 → 19.91 | 2.05× |
| Two nonconstant query positions | 8,192 | 8 | 18.97 → 9.50 | 2.00× | 21.42 → 12.62 | 1.70× |
| Random 16-bit | 8,192 | 16 | 12.31 → 8.38 | 1.47× | 18.47 → 12.26 | 1.51× |
| Periodic | 1,024 | 4 | 4.71 → 4.76 | 0.99× | 5.41 → 5.96 | 0.91× |

On the first row, 25,154,779 pairs have last-symbol Hamming distance at most one, of which just **6,179** are equal. At lengths 1,024, 4,096 and 8,192 this family's first-order speedups were 10.47×, 16.38× and 19.92×. The delayed-copy row has **4,088** equal pairs but **2,066,679** total baseline match characters. The skyline row has **98,554,632** equal pairs and benefits from its constant query. These constructions exercise different sufficient conditions.

The first-order native workspace estimates for the 8,192-symbol one-bit-neighbor case were 13,688,203 → 2,049,579 bytes. On the 4,096-symbol delayed copy they were 17,599,144 → 942,439 bytes. These are the native counter's tracked buffer-capacity estimates. Process RSS and GPU memory were not measured.

Across all 72 native comparisons the largest combined Q/K relative L2 error was **2.90e-14 or less**. Forward output, V and embedding gradients, and selected joint pair codes agreed byte for byte. Floating-point accumulation order differs between the two implementations. The native timer includes the training backward interface, index construction, gradient arrays, V and embedding credit and the configured trust/joint work. Compilation, forward and dense projections fall outside this timer.

[All 44 existing comparisons](../benchmarks/results/counterfactual_comparison.json) · [All 28 structured comparisons](../benchmarks/results/counterfactual_scaling.json)

## One full PyTorch layer update

A separate benchmark exercises the public float32 `NBitQKVRosa` layer on Gaussian input and target tensors. Each timed update includes zeroing gradients, Q/K/V/output projections, hard forward, mean-squared error, autograd backward and SGD. Each pair starts from identical parameters, data, and framework RNG. Batch size is one. These are synthetic single-layer training steps.

| T | Channels | Bits per group | First order, old → new (ms) | Speedup | Second order, old → new (ms) | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 8 | 1 | 90.93 → 31.43 | 2.89× | 97.33 → 39.10 | 2.49× |
| 4,096 | 32 | 8 | 41.76 → 42.08 | 0.99× | 56.01 → 52.47 | 1.07× |
| 4,096 | 32 | 16 | 15.37 → 12.95 | 1.19× | 19.55 → 15.83 | 1.24× |
| 2,048 | 64 | 64 | 9.81 → 9.39 | 1.04× | 10.89 → 10.34 | 1.05× |

The maximum measured absolute differences in output, parameter gradients and updated parameters were all zero on these eight float32 fixtures. The geometric-mean training-step speedup was **1.37×**. [All eight paired layer updates](../benchmarks/results/counterfactual_layer_steps.json)

## Reproduction

Use a CPU in the host's allowed affinity set. The receipts used CPU 7 and one numerical-library thread.

```bash
git worktree add --detach ../rosa-v8 881ab23
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python benchmarks/compare_checkout.py --reference-checkout ../rosa-v8 --cpu 7 --repeats 11 --expanded --output build/comparison.json
python benchmarks/compare_checkout.py --reference-checkout ../rosa-v8 --cpu 7 --repeats 11 --counterfactuals --output build/scaling.json
python benchmarks/compare_layer_step.py --reference-checkout ../rosa-v8 --cpu 7 --repeats 11 --output build/layer_steps.json
python tests/run.py --exhaustive
python tests/sanitize.py --tests subquadratic_counterfactual subquadratic_training --jobs 2
```

The native counterfactual oracle checked 246,477 scalar results across 2,322 sequence pairs, including direct recomputation after actual bit flips. The full suite reported 69 passes and one skipped CUDA-only test on this CPU host. ASan and UBSan passed for that oracle and the full C training interface. Leak detection was disabled for the recorded sanitizer run. [Validation receipt](../benchmarks/results/counterfactual_validation.json) · [Sanitizer receipt](../benchmarks/results/counterfactual_sanitizers.json)
