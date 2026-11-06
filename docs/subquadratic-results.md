# Current C++ engine: measurements and limits

[中文](subquadratic-results.zh-CN.md) · [Algorithm](../ALGORITHM.md) · [Reproduce](../benchmarks/README.md)

The repository ships one subquadratic backward engine. The final measurements show a substantial improvement over this round's starting implementation: **all 20 paired cases improve, by 2.23–14.31×**. Against the older historical engine, **28 of 42 cases with T≥1024 improve and 14 regress**. The worst measured regression is 1.34× the old latency. This is progress toward broad superiority, not a result that establishes it.

For fixed packed width, numerical dimension and joint budget, the structural bound remains $\widetilde O(T^{7/4})$ time and $\widetilde O(T^{4/3})$ space. Certified sparse work, short-context fingerprint reports and deletion summaries reduce finite-size overhead; bounded rejection returns to the same complete structural calculation. Input-family labels below are benchmark descriptions, never runtime dispatch keys. Route selection is exact in complete mode; binary64 reduction and FFT still round.

## Complete backward: absolute costs

Measured 2026-11-06 on Linux x86-64, NumPy 2.3.5, one CPU thread pinned to CPU 4, C++17 `-O3 -DNDEBUG -ffp-contract=off`. The four modes run in alternating order, with **nine samples per case**; the table uses their medians. One batch, one group (B=G=1), `trace=True`, verified cache enabled. Second credit includes the default trust coefficient 0.5 and at most one selected joint pair.

Timing includes input validation, allocation, index construction, all singleton credits, trust/joint work, V/embedding gradients and logit mapping. Compilation/warmup, forward, dense projections, loss and optimizer are outside the timer. Rows/s is T divided by the component-backward time; it is not end-to-end model throughput.

| Input | T | D | First ms | First rows/s | Diagonal + joint ms | Diagonal + joint rows/s | Buffer estimate MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 512 | 1 | 1.430 | 358,153 | 1.608 | 318,447 | 0.34 |
| random | 1,024 | 1 | 3.659 | 279,859 | 3.851 | 265,930 | 0.79 |
| random | 1,024 | 8 | 1.572 | 651,375 | 2.207 | 464,025 | 1.47 |
| random | 1,024 | 16 | 0.858 | 1,193,486 | 1.312 | 780,563 | 0.26 |
| random | 2,048 | 1 | 8.223 | 249,071 | 8.541 | 239,775 | 2.24 |
| random | 2,048 | 8 | 4.028 | 508,494 | 4.754 | 430,821 | 2.98 |
| random | 4,096 | 1 | 16.630 | 246,307 | 18.374 | 222,925 | 3.19 |
| random | 4,096 | 8 | 7.469 | 548,418 | 10.496 | 390,260 | 6.10 |
| random | 4,096 | 16 | 3.896 | 1,051,202 | 5.832 | 702,334 | 1.66 |
| random | 4,096 | 64 | 8.575 | 477,670 | 10.204 | 401,399 | 0.84 |
| random | 8,192 | 8 | 19.003 | 431,080 | 25.512 | 321,105 | 12.56 |
| random | 8,192 | 16 | 9.954 | 822,954 | 13.617 | 601,608 | 4.18 |
| random | 8,192 | 64 | 17.197 | 476,372 | 19.855 | 412,584 | 1.69 |
| unary | 1,024 | 8 | 0.784 | 1,305,586 | 0.891 | 1,148,663 | 0.52 |
| unary | 4,096 | 8 | 2.589 | 1,582,345 | 3.052 | 1,342,238 | 2.05 |
| periodic | 1,024 | 4 | 3.933 | 260,373 | 4.200 | 243,803 | 1.20 |
| periodic | 4,096 | 4 | 23.700 | 172,827 | 24.660 | 166,098 | 5.16 |
| shadow | 1,024 | 8 | 4.393 | 233,118 | 4.490 | 228,057 | 1.22 |
| shadow | 4,096 | 8 | 20.282 | 201,954 | 24.685 | 165,930 | 5.23 |
| shadow | 8,192 | 8 | 54.431 | 150,502 | 56.422 | 145,191 | 10.82 |
| skyline | 1,632 | 8 | 13.370 | 122,066 | 13.948 | 117,004 | 2.18 |
| skyline | 6,336 | 8 | 97.403 | 65,050 | 101.853 | 62,207 | 9.47 |

The last column is the larger of the two credit orders' **buffer capacity estimates**, not measured peak RSS. It conservatively counts retained buffers and component storage, may double count shared structures, and excludes allocator metadata, caller arrays and saved framework tensors. It must not be read as a process-memory guarantee. Every raw timing and counter is retained in [the final receipt](../benchmarks/results/optimized_engine.json).

`random` draws independent packed Q/K words; D=1 is binary. `unary` uses all-zero words. `periodic` repeats four symbols with one middle mutation. `shadow` uses all-zero Q with a single initial 1 in K. `skyline` concatenates unequal zero runs separated by 1. These deliberately cover sparse and dense matches, long contexts and frequent winner changes; they do not represent a distribution over trained models.

## What changed relative to the two baselines?

The paired comparisons alternate complete backward calls on identical inputs, options and proposal seeds, on CPU 4. The historical comparison uses nine repetitions; the round-baseline comparison uses seven. Their current-side times are measured independently of the absolute table above. Ratios below are calculated within each paired run, never by mixing measurements from separate runs.

The historical runtime is the code from `b5663b9`, loaded from the frozen `7a0df4f` checkout. Runtime source comparison confirms the intervening changes were documentation/demo/artifact changes, not a different matching engine. This round's baseline is checkout `4bb512c`, whose tree matches online `91bc`. Both checkouts are external benchmark inputs; neither is an installed alternative backend.

### Historical engine: all 44 paired cases

A ratio greater than one means the current implementation is faster. For T≥1024: **28 wins, 14 losses, no ties**. The worst is random T=2048, D=1, first: 7.386 → 9.906 ms, old/current=0.746×. The largest win is shadow T=8192, D=8, second, 10.99×.

| Input | T | D | Old / current: first | Old / current: second |
| --- | ---: | ---: | ---: | ---: |
| random | 512 | 1 | 0.811× | 0.820× |
| random | 1,024 | 1 | 0.819× | 0.836× |
| random | 1,024 | 8 | 1.022× | 0.924× |
| random | 1,024 | 16 | 1.082× | 1.039× |
| random | 2,048 | 1 | 0.746× | 0.881× |
| random | 2,048 | 8 | 1.040× | 0.825× |
| random | 4,096 | 1 | 1.044× | 0.960× |
| random | 4,096 | 8 | 1.243× | 1.063× |
| random | 4,096 | 16 | 1.286× | 1.040× |
| random | 4,096 | 64 | 1.296× | 1.084× |
| random | 8,192 | 8 | 1.177× | 1.149× |
| random | 8,192 | 16 | 1.294× | 1.095× |
| random | 8,192 | 64 | 1.301× | 1.069× |
| unary | 1,024 | 8 | 0.888× | 0.872× |
| unary | 4,096 | 8 | 0.851× | 0.878× |
| periodic | 1,024 | 4 | 0.835× | 0.850× |
| periodic | 4,096 | 4 | 2.131× | 2.117× |
| shadow | 1,024 | 8 | 2.161× | 2.574× |
| shadow | 4,096 | 8 | 6.009× | 6.340× |
| shadow | 8,192 | 8 | 9.990× | 10.989× |
| skyline | 1,632 | 8 | 0.914× | 1.033× |
| skyline | 6,336 | 8 | 1.515× | 1.665× |

The regressions occur in binary/random cases, small unary/periodic cases, and some skyline/second-credit cases. **Length ≥1024 is insufficient to guarantee a win, and these results do not establish a universal tenfold improvement.** The complete 14-case regression list is: random T=2048, D=8, second (0.825×); unary T=4096, D=8, first (0.851×); unary T=4096, D=8, second (0.878×); skyline T=1632, D=8, first (0.914×); random T=1024, D=1, first (0.819×); random T=1024, D=1, second (0.836×); random T=2048, D=1, first (0.746×); random T=2048, D=1, second (0.881×); random T=4096, D=1, second (0.960×); random T=1024, D=8, second (0.924×); unary T=1024, D=8, first (0.888×); unary T=1024, D=8, second (0.872×); periodic T=1024, D=4, first (0.835×); periodic T=1024, D=4, second (0.850×).

All 44 pairs passed numerical comparison; maximum Q/K relative L2 difference was 1.07e-14. Forward bytes, V/embedding-gradient bytes and selected pair codes matched in every pair. Q/K arithmetic is not claimed bitwise identical to the historical implementation.

### Starting implementation of this round

All 20 paired cases improve. Ratios use the same definition, starting-implementation time divided by current time.

| Input | T | D | Round baseline / current: first | Round baseline / current: second |
| --- | ---: | ---: | ---: | ---: |
| random | 512 | 1 | 7.74× | 7.03× |
| random | 2,048 | 8 | 9.63× | 7.62× |
| random | 4,096 | 64 | 2.36× | 2.23× |
| random | 8,192 | 16 | 5.95× | 4.94× |
| unary | 4,096 | 8 | 14.31× | 12.12× |
| periodic | 4,096 | 4 | 5.36× | 5.31× |
| shadow | 4,096 | 8 | 4.11× | 3.63× |
| shadow | 8,192 | 8 | 4.31× | 4.43× |
| skyline | 1,632 | 8 | 4.65× | 5.08× |
| skyline | 6,336 | 8 | 5.60× | 5.90× |

Source hashes, input hashes and all alternating samples are in [historical comparison](../benchmarks/results/optimized_comparison.json) and [round baseline](../benchmarks/results/optimized_previous_comparison.json).

## Exact randomization and optional approximation

The answer-preserving randomized cache verifies complete keys. Collisions cause exact recomputation; sampling is not involved. Across all 44 cases, cached and uncached outputs, pair codes and pair costs were bytewise identical. The observed uncached/cached time ratio spans **0.93–1.27×**. Cache lookup can cost more than it saves on low-reuse inputs. Default randomized execution is an exact policy, not a promise of a win on every call.

The approximate row method retains complete deletion and the one-symbol repair control, then samples complete residual rows. The table uses row budget 32 and channel budget 8. Channel sampling is a separate alternative that samples centered numerical features; it is not stacked with row sampling. Errors are the mean across nine recorded draws of $\|\hat g_{QK}-g_{QK}\|_2/\|g_{QK}\|_2$, after logit/joint mapping, relative to this engine's complete result. Both fast and unfavorable cases are included.

| Input | T | D | Credit | Complete ms | Row 32 ms | Complete / row | Row Q/K error | Channel 8 ms | Channel Q/K error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 2,048 | 1 | second | 8.541 | 10.299 | 0.83× | 3.24 | 8.403 | 0 |
| random | 2,048 | 8 | second | 4.754 | 4.990 | 0.95× | 0.391 | 5.926 | 1.43e-16 |
| random | 8,192 | 8 | second | 25.512 | 21.681 | 1.18× | 2.39 | 36.269 | 5.05e-16 |
| random | 8,192 | 16 | second | 13.617 | 13.760 | 0.99× | 0 | 13.750 | 0 |
| random | 8,192 | 64 | second | 19.855 | 20.135 | 0.99× | 0 | 20.393 | 0 |
| periodic | 4,096 | 4 | second | 24.660 | 10.974 | 2.25× | 0.0744 | 24.768 | 0 |
| shadow | 8,192 | 8 | first | 54.431 | 16.542 | 3.29× | 6.61 | 61.654 | 0 |
| shadow | 8,192 | 8 | second | 56.422 | 18.939 | 2.98× | 0.147 | 57.591 | 0 |
| skyline | 6,336 | 8 | first | 97.403 | 45.248 | 2.15× | 5.19 | 101.025 | 0 |
| skyline | 6,336 | 8 | second | 101.853 | 55.376 | 1.84× | 0.16 | 105.759 | 0 |

A zero-residual certificate removes only rows proved to contribute zero. It explains some near-machine-precision errors, but supplies no uniform error guarantee for the remaining estimator. With D≤8, channel budget 8 may retain every feature; such a timing does not demonstrate sampling acceleration. Some approximations are slower than complete evaluation, and first-order long-boundary cases can have large relative error. Budgets specify work, not an error tolerance.

V and embedding gradients stay complete. Singleton corners needed for a selected joint pair are restored by internal route evaluations. The caller still performs one ordinary forward and one backward; no extra model/loss call is required. Use complete mode unless application-level measurements justify approximation: compare Q and K separately, downstream parameter/input gradients, and held-out loss with fixed seeds and a shared optimizer schedule.

## Ordinary training: 48 small runs, 1,440 updates

Two tasks, three seeds (1101–1103), two credit orders, four methods and 30 Adam updates per run. Each uses two training examples, eight held-out examples and the same learning rate 0.005, fixed from an earlier exact-only pilot. Every update checks one model forward and one component backward. The table gives final held-out loss, mean ± population standard deviation across three seeds.

| Task | T | Credit | Complete / verified cache | Row 8 | Channel 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| periodic_copy | 64 | first | 3.536 ± 0.092 | 3.546 ± 0.091 | 3.226 ± 0.245 |
| periodic_copy | 64 | second | 3.253 ± 0.162 | 3.268 ± 0.100 | 3.235 ± 0.057 |
| prefix_copy | 128 | first | 2.812 ± 0.895 | 2.141 ± 0.127 | 2.205 ± 0.310 |
| prefix_copy | 128 | second | 3.339 ± 0.187 | 3.133 ± 0.186 | 4.038 ± 0.216 |

Complete and verified-cache parameter hashes match **at every update**. Initial mean held-out losses were 3.654 for periodic_copy and 3.361 for prefix_copy. Second credit improves the periodic task in these seeds; first credit finishes lower on the prefix task. Channel sampling worsens second-credit prefix loss from 3.339 to 4.038. The smaller row-sampled prefix loss does not establish a general advantage of noisy credit.

These T=64/128 tests verify integration and give a local training-quality check. Three seeds and 30 steps cannot establish large-model quality or long-sequence throughput. Full train/test losses, step timings, framework-call counts and per-update parameter hashes are in [optimized_toy.json](../benchmarks/results/optimized_toy.json).

## Gates passed, and the numerical boundary

The final validation ran **68 tests: 67 passed and one CUDA-only test was skipped**. Production source hashes stayed unchanged. Coverage includes literal all-bit rerouting, supplied diagonal costs, bit 63, empty inputs, ordinary autograd, checkpointing, CPU autocast, saved configuration and framework RNG parity. The five separately built ASan/UBSan oracle executables all passed; leak detection was disabled, so this is not a leak-check claim.

Independent focused checks include 5,641 Q-frontier fixtures; 6,932 short/near-K cases; 5,896 indexed-row cases with 1,995,800 literal routes; and 5,682 fingerprint fixtures with forced collisions. The integration oracle covers 635 C ABI calls, 2,166,640 gradient values and 2,016 joint costs; its maximum absolute discrepancy is 7.11e−14. Budget rejection is checked for unchanged output/state, and fingerprint collisions are verified against literal routes.

The correlation oracle checks 217,267 interval queries. Its maximum error divided by the sum of absolute products is 1.66e−16. A deliberately ill-conditioned cancellation case has **0.505 absolute error**: expansion summation cannot recover information already rounded away in FFT coefficients. Complete means algebraically complete and route-exact, not arbitrary-precision arithmetic. Measure cancellation-sensitive applications against their own numerical tolerance.

## Reproducible receipts

- [Final four-mode timings and per-draw errors](../benchmarks/results/optimized_engine.json)
- [Interleaved historical-engine comparison](../benchmarks/results/optimized_comparison.json)
- [Interleaved round-baseline comparison](../benchmarks/results/optimized_previous_comparison.json)
- [Final correctness gate](../benchmarks/results/optimized_validation.json)
- [Final ASan/UBSan gate](../benchmarks/results/optimized_sanitizers.json)
- [Final training trajectories and parameter hashes](../benchmarks/results/optimized_toy.json)

Earlier receipts remain available for provenance: [engine](../benchmarks/results/subquadratic_engine.json), [historical comparison](../benchmarks/results/subquadratic_comparison.json), [validation](../benchmarks/results/subquadratic_validation.json), [sanitizers](../benchmarks/results/subquadratic_sanitizers.json), [toy training](../benchmarks/results/subquadratic_toy.json), and [earlier toy comparison](../benchmarks/results/subquadratic_toy_comparison.json). They describe earlier source snapshots, not the timings above.
