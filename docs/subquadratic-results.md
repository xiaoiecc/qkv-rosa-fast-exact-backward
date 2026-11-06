# Current C++ engine: measurements and limits

[中文](subquadratic-results.zh-CN.md) · [Algorithm](../ALGORITHM.md) · [Reproduce](../benchmarks/README.md)

The current repository contains only the subquadratic engine. The previous
optimized engine, its selectable paths, and its executable snapshots have
been deleted. Historical JSON receipts remain identifiable evidence.

For fixed packed width, numerical dimension and joint budget, the structural
bound is $\widetilde O(T^{7/4})$ time and $\widetilde O(T^{4/3})$ space.
The measurements below describe finite-size execution, which does not follow
from that bound alone. Complete mode uses every contribution; its route
choices are exact, while FFT and reassociation still round in binary64.

## Complete backward: absolute costs

Measured 2026-11-06 on a Linux x86-64 CPU, affinity 4, NumPy 2.3.5, C++17
`-O3 -DNDEBUG -ffp-contract=off`. Each entry is the median of seven interleaved
calls after compilation/warmup. All rows use the same engine with the verified
cache enabled. A stream has one bit group, B=G=1. Second credit uses the default
trust coefficient 0.5 and at most one selected joint pair.

The timer includes input validation, allocations, all indexes, singleton
credits, diagonal/joint work, V/embedding gradients and logit mapping. It
excludes the model forward, dense projections, loss and optimizer. “rows/s”
means T divided by this component-backward time; it is not model throughput.

| Input | T | D | First ms | Diagonal + joint ms | Diagonal + joint rows/s | Estimated workspace MiB |
|---|---:|---:|---:|---:|---:|---:|
| random | 512 | 1 | 12.718 | 12.499 | 40,962 | 0.47 |
| random | 2,048 | 8 | 51.695 | 40.978 | 49,978 | 2.11 |
| random | 8,192 | 16 | 52.456 | 69.421 | 118,004 | 9.29 |
| random | 4,096 | 64 | 19.990 | 22.553 | 181,620 | 7.83 |
| unary | 4,096 | 8 | 35.362 | 32.817 | 124,815 | 4.43 |
| periodic | 4,096 | 4 | 129.865 | 126.735 | 32,319 | 4.43 |
| shadow | 4,096 | 8 | 84.729 | 87.392 | 46,869 | 5.83 |
| shadow | 8,192 | 8 | 186.739 | 209.366 | 39,128 | 11.87 |
| skyline | 1,632 | 8 | 72.110 | 78.137 | 20,886 | 3.11 |
| skyline | 6,336 | 8 | 445.587 | 439.758 | 14,408 | 15.70 |

Workspace is the engine's diagnostic estimate, **not measured peak RSS**. It
does not account for every allocator overhead, saved framework tensor or
caller-owned array; use the space proof for asymptotics, not this estimate as
a process-memory guarantee. The host can have concurrent work even when the
process is pinned. Small timing differences should not be treated as stable
wins. Every sample, counter and source hash is in
[subquadratic_engine.json](../benchmarks/results/subquadratic_engine.json).

`unary` uses all-zero Q/K. `periodic` repeats four symbols with one middle
mutation. `shadow` uses all-zero Q and K with a single initial 1. `skyline`
concatenates unequal zero runs separated by 1. The source generator specifies
every input; these are benchmark families, not runtime dispatch keys.

## Exact randomized acceleration

The cache verifies the entire `(row,length,cutoff)` key. Collisions evict or
recompute. It does not select credit terms or reorder arithmetic. Across all
20 timing cases the cached and uncached outputs were bytewise identical.
Observed uncached/cached speed ratios ranged from 0.79× to
1.36×. Low-reuse inputs can pay a net lookup cost despite adaptive
bypass. The default follows the answer-preserving randomized policy; disable
`randomized` if measurements on the application favor doing so.

## Explicit approximation: speed and error belong together

The default approximate method retains full deletion and the one-symbol
repair control, then estimates complete residual rows. In this table the
budget is 32. The relative error is the mean across the seven recorded draws
of $\|\hat g_{QK}-g_{QK}\|_2/\|g_{QK}\|_2$, measured after logit and
joint mapping against the current complete mode. Q-only and K-only errors,
cosines and every draw remain in the raw receipt.

| Input | T | D | Credit | Complete ms | Row 32 ms | Speedup | Q/K relative error |
|---|---:|---:|---|---:|---:|---:|---:|
| random | 2,048 | 8 | second | 40.978 | 14.212 | 2.88× | 0.412 |
| random | 8,192 | 16 | second | 69.421 | 47.239 | 1.47× | 1.6e-16 |
| periodic | 4,096 | 4 | second | 126.735 | 72.963 | 1.74× | 0.0679 |
| shadow | 8,192 | 8 | second | 209.366 | 101.247 | 2.07× | 0.155 |
| skyline | 6,336 | 8 | second | 439.758 | 237.588 | 1.85× | 0.155 |
| shadow | 8,192 | 8 | first | 186.739 | 92.461 | 2.02× | 7.06 |

A residual certificate removes only rows proved to contribute zero. This
explains the near-zero error for the random T=8192,D=16 case. It does not make
the row estimator uniformly accurate: first-order long-boundary cases show
large errors. Approximation stays opt-in, with no claim of a universal
relative-error or training-loss bound.

A positive channel budget instead samples centered numerical features of the
full structure. The empty-route indicator is always complete. It is an
alternative to row sampling, not a second sampler stacked on it. With D≤8,
`channels8` often retains every relevant feature, so those timings do not
demonstrate channel sampling speed. On random T=8192,D=16, second credit,
channel-8 averaged 0.285 Q/K relative error and took 93.618 ms, slower than the
69.421 ms complete call. That negative result is retained.

V and embedding gradients remain fully evaluated. Approximate singleton
corners needed by a selected joint pair are restored with complete internal
route evaluations. No extra model or loss evaluation is requested from the
caller.

## A small historical comparison

These old times come from the frozen pre-removal
[comparison receipt](../benchmarks/results/subquadratic_comparison.json).
Current times come from the final-source receipt above. Inputs and machine
are shared, but these are separate timing runs, not simultaneous paired
measurements. The old code is not included in the repository.

| Input | T | D | Credit | Historical ms | Current complete ms | Old / current |
|---|---:|---:|---|---:|---:|---:|
| shadow | 8,192 | 8 | second | 576.753 | 209.366 | 2.75× |
| random | 2,048 | 8 | second | 4.962 | 40.978 | 0.12× |
| periodic | 4,096 | 4 | second | 57.453 | 126.735 | 0.45× |
| skyline | 6,336 | 8 | second | 143.487 | 439.758 | 0.33× |

The long-boundary example improves substantially; random, periodic and
skyline examples can be slower. **T≥1024 is not sufficient to guarantee a
speedup.** The retained comparison does not support a general tenfold claim.
The old receipt also contains the pre-removal algebra/rounding comparison;
it must not be relabeled as a final-source bitwise-parity claim.

## Ordinary training: 48 small runs, 1,440 updates

Two structured tasks, three seeds (1101–1103), two credit orders, four methods,
30 Adam updates, common learning rate 0.005, two training examples and eight
held-out examples per run. Every update asserts one model forward and one
component backward. Evaluation is outside the step timer. The learning rate
was fixed from an earlier exact-only pilot; neither approximate method gets
its own tuning. Values below are final held-out losses, mean ± population
standard deviation across three seeds.

| Task | T | Credit | Complete / verified cache | Row 8 | Channel 4 |
|---|---:|---|---:|---:|---:|
| periodic_copy | 64 | first | 3.536 ± 0.092 | 3.546 ± 0.091 | 3.226 ± 0.245 |
| periodic_copy | 64 | second | 3.253 ± 0.162 | 3.268 ± 0.100 | 3.235 ± 0.057 |
| prefix_copy | 128 | first | 2.812 ± 0.895 | 2.141 ± 0.127 | 2.205 ± 0.310 |
| prefix_copy | 128 | second | 3.339 ± 0.187 | 3.133 ± 0.186 | 4.038 ± 0.216 |

The complete and verified-cache parameter hashes match at **every update**.
For these seeds, the initial mean test losses were 3.654 (periodic) and 3.361
(prefix). Second credit helps the periodic task, while first credit finishes
lower on the prefix task. Channel sampling hurts second-order prefix loss
substantially. This is an integration and local-quality check, not evidence
that one rule or sampler dominates on arbitrary models. Short T=64/128 step
times should not be used to advertise long-sequence speed. Full trajectories
and per-step samples are in [subquadratic_toy.json](../benchmarks/results/subquadratic_toy.json).

## Correctness and numerical conditioning

The final [validation gate](../benchmarks/results/subquadratic_validation.json)
ran 64 tests: 63 passed, one CUDA-only test was skipped, and production
source hashes stayed unchanged. It includes literal all-bit rerouting, supplied diagonal costs,
64-bit symbols, the five structural C++ oracles, normal autograd, checkpoint,
CPU autocast, saved configuration and framework-RNG parity.

The [ASan/UBSan run](../benchmarks/results/subquadratic_sanitizers.json) passed
635 C ABI calls, 2,166,640 gradient comparisons, 2,016 joint-cost comparisons
and 52 independent literal fixtures. Maximum absolute discrepancy in that
integration driver was 6.58e-14. Empty data pointers at T=0 and the strict
zero coordinates Q[0], K[T−1], V[0] are covered.

The numerical correlation oracle includes 217,157 interval queries. Maximum
error divided by the sum of absolute products was 1.67e-16. Its deliberately
ill-conditioned case, with products around 2^50 cancelling toward a unit
residue, had **0.505 absolute error**. Expansion accumulation cannot recover
bits already lost in FFT coefficients. These two observations are compatible:
relative error in a tiny cancelled answer can be large even when backward
error relative to the products is small. The complete mode is algebraically
complete and route-exact; it is not arbitrary-precision arithmetic.
