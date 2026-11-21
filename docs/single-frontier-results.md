# Single-algorithm timings against PR #10

Reference `c3a9368`, measured experiment `31a066f`. This draft is a research comparison. The independent Gaussian and full PyTorch tests favor retaining the #10 release path. One exception is the long skewed 8-bit case, where the single algorithm wins.

Native trials use one pinned Intel Xeon Platinum 8573C thread and the public exact backward with identical seeds, traces disabled during timing. After one warm-up, the runner alternates checkouts. Each row divides the #10 median time by the single-algorithm median time. A value below one means a slowdown. Group results are geometric means of per-row ratios. The PyTorch measurement includes projections, forward, loss, backward and SGD at batch size one on CPU.

| Grid | Faster cases | Geometric mean | Min | Max | Samples |
| --- | ---: | ---: | ---: | ---: | --- |
| Gaussian, lengths 512 and 2048, 7 widths, 2 seeds | 4 / 56 | 0.53× | 0.20× | 1.25× | [JSON](../benchmarks/results/frontier_gaussian.json) |
| Gaussian, lengths 4096 and 8192, 5 widths, 2 fresh seeds | 3 / 40 | 0.47× | 0.21× | 1.08× | [JSON](../benchmarks/results/frontier_holdout.json) |
| Biased 0.8, lengths 4096 and 8192, 5 widths, 2 fresh seeds | 4 / 40 | 0.30× | 0.10× | 1.58× | [JSON](../benchmarks/results/frontier_holdout.json) |
| Four structured families, lengths 256 through 2048 | 1 / 32 | 0.16× | 0.01× | 1.07× | [JSON](../benchmarks/results/frontier_structured.json) |
| Full PyTorch mixer step, 4 shapes, 3 seeds | 0 / 24 | 0.57× | 0.23× | 0.97× | [JSON](../benchmarks/results/frontier_layer.json) |

## Cases that constrain the design

Numbers are first-order / second-order speedups. Timings in the last column give first-order #10 → single algorithm in milliseconds, using the median of the per-seed medians.

| Input | Length | Bits | Speedups | First order ms |
| --- | ---: | ---: | ---: | ---: |
| gaussian | 2048 | 1 | 0.24× / 0.29× | 3.13 → 13.04 |
| gaussian | 2048 | 8 | 0.55× / 0.56× | 2.78 → 5.02 |
| gaussian | 2048 | 12 | 1.18× / 1.17× | 4.04 → 3.43 |
| gaussian | 8192 | 8 | 0.46× / 0.48× | 10.60 → 23.18 |
| biased | 4096 | 8 | 0.27× / 0.31× | 8.44 → 31.26 |
| biased | 8192 | 8 | 1.50× / 1.52× | 142.58 → 95.35 |
| unary | 2048 | 8 | 0.01× / 0.01× | 1.77 → 126.09 |
| periodic | 2048 | 4 | 0.18× / 0.17× | 18.94 → 103.15 |
| lagged_copy | 2048 | 64 | 0.93× / 0.94× | 9.34 → 10.03 |
| near_fanout | 2048 | 8 | 0.01× / 0.01× | 1.45 → 142.09 |

For skewed 8-bit inputs the two measured length-8192 instances have 587,987 and 563,169 causal two-symbol neighbors. The #10 reduction's per-stream limit is 524,288 visits. Separate diagnostic runs of the reference observed failed two-symbol, packed and trie attempts on both seeds. At length 4096 the reference completed on its first two-symbol attempt. The native samples in the tables use the original uninstrumented binaries.

For constant inputs the frontier explicitly visits a quadratic set of equal windows. The [derivation](single-suffix-frontier.md#why-the-worst-case-matters) gives an interval formula for the same changes and explains this disparity. At long random 8-bit widths, the #10 reducer also batches contributions from many length-one repairs by shared symbol.

## Numerical checks

Across the 168 native rows, forward outputs and V/embedding gradients were byte-identical, joint proposal codes were exact and numerical costs passed tolerance. The maximum combined Q/K relative L2 difference was 2.255e-14. The 24 PyTorch steps had maximum absolute differences of 0 for outputs, 0 for parameter gradients and 0 after SGD.

The exhaustive suite passed 72 tests and skipped the one CUDA transfer test on this CPU host. A literal flip oracle checked 115,506 scalars from this exact Q/K implementation over 2,413 sequence pairs. It includes immutable symbol tags, the highest possible editable bit, word-boundary suffixes and overlapping surviving windows. Both the flip oracle and the public native training oracle passed ASan and UBSan with leak detection disabled. [Exhaustive suite](../benchmarks/results/frontier_validation.json) · [Sanitizers](../benchmarks/results/frontier_sanitizers.json)

## Reproduction

Run from this branch and substitute an available CPU index.

```bash
git worktree add --detach ../rosa-reference c3a9368202912dd8b6903801e8b52e49e897fd8d
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
common=(--reference-checkout ../rosa-reference --cpu 7)
python benchmarks/random_qkv.py "${common[@]}" --families gaussian --widths 1 4 8 12 16 32 64 --lengths 512 2048 --seeds 192631 566819 --repeats 7 --output build/frontier_gaussian.json
python benchmarks/random_qkv.py "${common[@]}" --families gaussian biased --widths 1 8 12 16 64 --lengths 4096 8192 --seeds 160309 771317 --repeats 5 --output build/frontier_holdout.json
python benchmarks/frontier_structured.py "${common[@]}" --repeats 7 --output build/frontier_structured.json
python benchmarks/compare_layer_step.py "${common[@]}" --seed-offsets 0 7919 104729 --repeats 7 --output build/frontier_layer.json
python tests/run.py --exhaustive --output build/frontier_validation.json
python tests/sanitize.py --no-leaks --tests subquadratic_counterfactual subquadratic_training --output build/frontier_sanitizers.json
```
