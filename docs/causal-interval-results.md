# Paired CPU results

All comparisons use one pinned Intel Xeon Platinum 8573C thread, one warm-up, five interleaved timed repetitions and identical inputs for each checkout. The measured code is 0541f48. The references are merged main at 881ab23 and former PR #12 at 7059000. Speedup is reference time divided by candidate time. Each cell gives the geometric mean of per-case median speedups and the number of faster cases.

| Inputs | Against main | Against #12 |
|---|---:|---:|
| Independent Gaussian, lengths 2048 and 8192, 8 widths, 3 seeds | 2.91× (96/96) | 0.82× (8/96) |
| Independent random signs, length 8192, 8 widths, 2 seeds | 3.23× (32/32) | 0.84× (5/32) |
| Independent Gaussian signs with P(bit=1)=0.8, same grid | 2.90× (29/32) | 1.02× (16/32) |
| Gaussian, all 64 widths, length 4096, one fresh seed | 3.43× (128/128) | 0.91× (25/128) |
| Gaussian and biased, length 32768, 5 widths, one fresh seed | 3.53× (19/20) | 1.04× (9/20) |
| Complete PyTorch mixer SGD step, 4 shapes, 3 seeds | 2.01× (24/24) | 0.85× (6/24) |
| Structured inputs, 4 families and 4 lengths | 2.10× (29/32) | 0.90× (15/32) |

## Independent Gaussian at length 8192

First and second order appear in that order. Absolute times are medians across seed medians in the main comparison. The independently interleaved #12 comparison uses the same inputs.

| Case | Main ms | Candidate ms | Speedup over main | Speedup over #12 |
|---|---:|---:|---:|---:|
| 1 bit | 57.24 / 58.39 | 12.68 / 15.24 | 4.49× / 3.92× | 0.61× / 0.70× |
| 2 bits | 41.33 / 42.89 | 11.73 / 13.48 | 3.53× / 3.21× | 0.68× / 0.78× |
| 4 bits | 32.46 / 35.60 | 9.94 / 10.93 | 3.28× / 3.16× | 0.86× / 0.92× |
| 8 bits | 23.81 / 28.56 | 7.79 / 9.68 | 2.93× / 3.00× | 0.96× / 0.97× |
| 12 bits | 21.05 / 28.69 | 12.79 / 15.41 | 1.62× / 1.89× | 0.90× / 0.96× |
| 16 bits | 11.21 / 15.75 | 9.23 / 11.07 | 1.21× / 1.42× | 0.67× / 0.72× |
| 32 bits | 11.95 / 15.88 | 2.00 / 2.69 | 5.82× / 5.86× | 0.98× / 1.00× |
| 64 bits | 21.42 / 27.31 | 3.75 / 5.24 | 5.77× / 4.96× | 1.02× / 0.96× |

## Complete PyTorch mixer updates

First and second order appear in that order. Absolute times are medians across seed medians in the main comparison. The independently interleaved #12 comparison uses the same inputs.

| Case | Main ms | Candidate ms | Speedup over main | Speedup over #12 |
|---|---:|---:|---:|---:|
| T=2048, C=8, b=1 | 88.60 / 92.30 | 25.00 / 28.07 | 3.53× / 3.18× | 0.67× / 0.71× |
| T=4096, C=32, b=8 | 46.00 / 51.28 | 22.63 / 25.26 | 2.05× / 2.05× | 0.86× / 0.80× |
| T=4096, C=32, b=16 | 14.08 / 19.78 | 11.51 / 13.24 | 1.25× / 1.50× | 0.94× / 0.84× |
| T=2048, C=64, b=64 | 9.79 / 12.08 | 6.28 / 6.23 | 1.63× / 1.87× | 1.05× / 1.02× |

## Structured inputs at length 2048

First and second order appear in that order. Absolute times are medians across seed medians in the main comparison. The independently interleaved #12 comparison uses the same inputs.

| Case | Main ms | Candidate ms | Speedup over main | Speedup over #12 |
|---|---:|---:|---:|---:|
| unary | 1.50 / 1.78 | 0.88 / 1.67 | 1.70× / 1.07× | 1.44× / 1.25× |
| periodic | 10.10 / 10.78 | 16.19 / 19.56 | 0.62× / 0.55× | 0.64× / 0.55× |
| lagged_copy | 15.65 / 17.33 | 12.49 / 13.96 | 1.25× / 1.24× | 0.44× / 0.43× |
| near_fanout | 26.22 / 25.98 | 1.34 / 2.21 | 19.54× / 11.74× | 1.08× / 0.73× |

At length 32768 the five-width mixed Gaussian/biased grid averaged 1.04× against #12. On the biased 8-bit stream the two orders were 1.75× and 1.41× faster. Both length-8192 uniform 1-bit cases and the structured periodic and delayed-copy cases favor #12. The longer grid uses one seed per distribution.

## Accuracy, workspace and scope

Native timings include the NumPy wrapper, allocation and complete native backward. The PyTorch test includes the linear projections, causal forward, MSE loss, backward and SGD. It restores identical parameters for each repetition. Periodic and delayed-copy inputs expose the cost of longer posting lists.

680 paired native cases had byte-identical forward outputs, joint codes and V/embedding gradients. Joint costs and Q/K gradients passed tolerance. Maximum combined Q/K relative L2 discrepancy was 9.52e-15. All 48 paired full steps passed output, gradient and post-SGD parameter checks. The largest absolute parameter-gradient difference was 1.49e-08.

Primary Gaussian workspace estimates ranged from 0.19× to 4.12× the main estimate. The counters track selected native buffer capacities. They omit allocator overhead and framework tensors. Process RSS was not measured.

The exhaustive CPU suite ran 70 tests with 1 CUDA-only skip. The independent causal DP oracle checked 74,896 physical flips and 1,812 forward rows. It and the C training ABI passed ASan and UBSan with leak detection disabled.

At T=8192 the support probe retained 361, 284 and 307 coordinates at 24 bits, then 9, 0 and 0 at 32 bits. At 64 bits it retained zero coordinates for all three seeds. At 8 and 16 bits all 8192 positions were retained. This measures the singleton reducer. Input validation and dense gradient output still span the full stream.

[All timings, source and binary hashes, numerical errors, support counts, validation, sanitizers and narrow-loop ablation](../benchmarks/results/causal-interval-measurements.zip)

## Reproduction

Use a separate reference checkout and select a CPU allowed on the benchmark machine.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python benchmarks/random_qkv.py --reference-checkout ../reference --cpu 1 --repeats 5 --families gaussian --lengths 2048 8192 --seeds 923153 503369 181093 --output /tmp/native.json
python benchmarks/random_qkv.py --reference-checkout ../reference --cpu 1 --repeats 5 --families biased bernoulli --lengths 8192 --seeds 923153 503369 --output /tmp/holdout.json
python benchmarks/random_qkv.py --reference-checkout ../reference --cpu 1 --repeats 5 --families gaussian --lengths 4096 --widths $(seq 1 64) --seeds 437989 --output /tmp/all-widths.json
python benchmarks/random_qkv.py --reference-checkout ../reference --cpu 1 --repeats 5 --families gaussian biased --lengths 32768 --widths 1 8 16 32 64 --seeds 891443 --output /tmp/long.json
python benchmarks/compare_layer_step.py --reference-checkout ../reference --cpu 1 --repeats 5 --seed-offsets 186 4297 7011 --output /tmp/layer.json
python benchmarks/frontier_structured.py --reference-checkout ../reference --cpu 1 --repeats 5 --output /tmp/structured.json
c++ -O3 -std=c++17 benchmarks/interval_support.cpp -o /tmp/interval-support && /tmp/interval-support
python tests/run.py --exhaustive
python tests/sanitize.py --tests interval_credit subquadratic_training --jobs 2 --no-leaks
```
