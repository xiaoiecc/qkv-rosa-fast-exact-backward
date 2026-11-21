# Random QKV performance and numerical checks

The reference is PR #9 at `44f6e23`. The measured implementation is `25f49a8`. Both use the same CPU exact credit rule and seeded joint proposals. Each native timing includes the NumPy wrapper, allocations and complete backward. Forward, compilation and trace-based correctness checks are outside the timer. The layer timings below include forward and the optimizer.

Each checkout ran once for warm-up, then five or seven times per input in alternating order on the same pinned CPU thread (Intel Xeon Platinum 8573C, Linux). A row is one input and credit order. Its speedup is the reference median time divided by the new median. Aggregates use the geometric mean of those ratios, giving each row equal weight. They describe these inputs on this machine. All samples, seeds and input digests are in the linked JSON files.

## Complete native backward

| Distribution and grid | Faster / measured | Geometric mean | Slowest | Fastest | Data |
| --- | ---: | ---: | ---: | ---: | --- |
| Independent Gaussian, 8 widths, 3 lengths, 3 seeds | 142 / 144 | 2.15× | 0.81× | 6.42× | [JSON](../benchmarks/results/random_qkv_native.json) |
| Independent random signs, 8 widths, length 4096, 3 seeds | 48 / 48 | 2.32× | 1.34× | 5.14× | [JSON](../benchmarks/results/random_qkv_native.json) |
| Biased independent bits, 8 widths, length 4096, 3 seeds | 48 / 48 | 2.80× | 1.43× | 6.31× | [JSON](../benchmarks/results/random_qkv_native.json) |
| Biased independent bits, widths 14 and 16, 2 lengths, 3 fresh seeds | 24 / 24 | 1.85× | 1.41× | 2.36× | [JSON](../benchmarks/results/random_qkv_skew_validation.json) |
| Gaussian, every width from 1 to 64, length 4096, 1 seed | 127 / 128 | 3.11× | 0.98× | 6.32× | [JSON](../benchmarks/results/random_qkv_all_widths.json) |
| Gaussian, 6 widths, lengths 16384 and 32768, 1 seed | 24 / 24 | 2.25× | 1.07× | 5.44× | [JSON](../benchmarks/results/random_qkv_long.json) |
| Gaussian, width 3, lengths 1024 to 32768, 1 seed | 8 / 8 | 3.77× | 2.03× | 5.29× | [JSON](../benchmarks/results/random_qkv_odd_scaling.json) |

Gaussian logits are i.i.d. standard normal for Q, K and V. The random signs have magnitude 0.35. The skewed Gaussian has bit probability 0.8. The narrow fresh-seed skew check follows a change to the 13–16-bit path, prompted by a 16-bit slowdown in the initial implementation. The breadth and odd-width sweeps share their width-3, length-4096 input.

### Gaussian by length and bit width

Each cell gives first-order / second-order speedup. The three shorter columns average three independently seeded ratios per order. Length 32768 uses one different seed and five repetitions per checkout.

| Bits | 1024 | 4096 | 8192 | 32768 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.67× / 1.55× | 1.76× / 1.69× | 1.80× / 1.73× | 2.12× / 2.03× |
| 2 | 1.52× / 1.45× | 1.62× / 1.70× | 1.57× / 1.66× | — |
| 4 | 1.86× / 1.28× | 1.49× / 1.48× | 1.72× / 1.67× | 1.52× / 1.59× |
| 8 | 2.22× / 1.53× | 2.75× / 2.54× | 2.96× / 3.13× | 4.93× / 4.81× |
| 12 | 1.00× / 1.40× | 1.41× / 1.63× | 1.62× / 1.67× | 1.72× / 1.58× |
| 16 | 1.69× / 1.48× | 1.28× / 1.49× | 1.19× / 1.37× | 1.07× / 1.16× |
| 32 | 3.73× / 3.61× | 5.13× / 5.21× | 4.63× / 2.83× | — |
| 64 | 5.81× / 5.35× | 5.45× / 5.33× | 4.99× / 5.39× | 4.55× / 4.76× |

The 8-bit, length-4096 case took 15.91 → 5.74 ms for first order and 20.33 → 8.09 ms for second order (median of the three per-seed medians). At width 3, increasing the length from 1024 to 32768 gave speedups of 2.88× / 2.03× and 5.29× / 4.34×, respectively.

Individual regressions in the independent Gaussian sweeps are listed below. Millisecond timings include system variability.

| Length | Bits | Order | Seed | Reference ms | New ms | Extra time |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1024 | 12 | first | 597353 | 1.282 | 1.577 | 23.0% |
| 4096 | 13 | first | 240119 | 7.536 | 7.663 | 1.7% |
| 1024 | 2 | second | 597353 | 4.083 | 4.122 | 1.0% |

The recorded workspace estimate varies by width. At 8 bits and length 4096 the three-seed median dropped from 6.10 to 1.71 MiB. The largest proportional increase in the measured random rows was a 12-bit Gaussian first-order case at length 4096, from 0.41 to 2.75 MiB. These counters estimate native workspace. Process RSS and GPU memory were outside this benchmark.

## PyTorch training step

Each repetition starts from identical model parameters, inputs and targets in both checkouts. The timer covers gradient clearing, a full `NBitQKVRosa` forward with projections, mean-squared loss, backward and one SGD update. Each shape uses three seeds for its inputs and weights. Each seeded setup has seven timed calls per checkout. Batch size is one, tensors are float32 and PyTorch uses one CPU thread. This measures one mixer update on synthetic data. A complete Transformer and multi-step training convergence are outside the scope of the measurement.

| Length | Channels | Bits | First order | Second order |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 8 | 1 | 1.86× | 1.74× |
| 4096 | 32 | 8 | 2.21× | 2.24× |
| 4096 | 32 | 16 | 1.22× | 1.52× |
| 2048 | 64 | 64 | 1.54× | 1.78× |

All 24 layer comparisons accelerated. Their geometric mean was 1.73×. The maximum absolute difference was zero for forward outputs and updated parameters and $1.49\times10^{-8}$ for parameter gradients. [Layer timings and parity](../benchmarks/results/random_qkv_layer.json)

## Correctness

Across the five random native sweeps, forward outputs and V/embedding gradients were byte-identical, sampled joint pair codes matched exactly, and numerical joint costs and Q/K gradients passed the paired tolerance. The maximum combined Q/K relative L2 difference was $1.57\times10^{-14}$ (rounded up). The separate 28-row structured control also passed; 27 of its 28 timings accelerated, with the remaining shadow case taking 1.2% longer. [Structured control](../benchmarks/results/random_qkv_structured.json)

The exhaustive CPU suite ran 73 tests, including one skipped CUDA transfer test on this CPU-only host. AddressSanitizer and UndefinedBehaviorSanitizer passed the three standalone native oracles with leak detection disabled. The counterfactual oracle compared 526,751 scalars over 2,413 stream pairs, the two-symbol oracle checked 154,760 values, and the training ABI oracle compared 2,166,640 gradients. [Full validation](../benchmarks/results/random_qkv_validation.json) · [Sanitizer reports](../benchmarks/results/random_qkv_sanitizers.json)

## Reproduction

Use separate checkouts of the two measured revisions. Run from the new checkout with the repository dependencies and a C++17 compiler. Pick an allowed CPU index on your host.

```bash
git worktree add --detach ../rosa-reference 44f6e231afe891db3049eea32bac945b1e33a93b
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
core=7
common=(--reference-checkout ../rosa-reference --cpu "$core")
python benchmarks/random_qkv.py "${common[@]}" --output build/random_qkv_native.json
python benchmarks/random_qkv.py "${common[@]}" --families biased --lengths 4096 8192 --widths 14 16 --seeds 148239 529633 971041 --output build/random_qkv_skew_validation.json
python benchmarks/random_qkv.py "${common[@]}" --families gaussian --lengths 16384 32768 --widths 1 4 8 12 16 64 --seeds 860281 --repeats 5 --output build/random_qkv_long.json
python benchmarks/random_qkv.py "${common[@]}" --families gaussian --lengths 4096 --widths {1..64} --seeds 240119 --repeats 5 --output build/random_qkv_all_widths.json
python benchmarks/random_qkv.py "${common[@]}" --families gaussian --lengths 1024 4096 16384 32768 --widths 3 --seeds 240119 --repeats 5 --output build/random_qkv_odd_scaling.json
python benchmarks/compare_layer_step.py "${common[@]}" --repeats 7 --seed-offsets 0 7919 104729 --output build/random_qkv_layer.json
python benchmarks/compare_checkout.py "${common[@]}" --counterfactuals --repeats 5 --output build/random_qkv_structured.json
python tests/run.py --exhaustive --output build/random_qkv_validation.json
python tests/sanitize.py --tests subquadratic_counterfactual subquadratic_bigram subquadratic_training --no-leaks --output build/random_qkv_sanitizers.json
```
