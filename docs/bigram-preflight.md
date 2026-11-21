# Certifying two-symbol work before reduction

PR #10 could build features, scan causal pairs and accumulate partial gradients before its 64T visit allowance expired. The new path chooses whether to run before allocating features or gradient accumulators. After admission its numerical loop can finish all reports within 128T visits. It retains the existing equality, tie and credit calculations. Other paths in the #10 engine retain their own policies.

## What the preflight counts

Let $P^Q_t$ and $P^K_r$ pack two adjacent query and key symbols. Let $B$ be the actual directory bucket function and $C_t(j)$ count key pairs with $1\le r<t$ in bucket $j$. Including the unedited word, there are $2b+1$ masks in $M=\{0,2^0,\ldots,2^{2b-1}\}$. The number of list entries read by the numerical sweep is exactly

$$
W=\sum_{t=2}^{T-1}\sum_{m\in M} C_t\big(B(P^Q_t\mathbin{\oplus}m)\big).
$$

This includes collisions and multiple probes to the same physical bucket. A cheap sufficient bound uses the largest bucket frequency $F$ over all usable key pairs. If $(T-2)(2b+1)F\le128T$, the entire stream fits. Otherwise the code counts $W$ backwards, removing the key ending at $t-1$ after checking each query. It may stop as soon as the allowance is exceeded. For an admitted stream the reverse pass leaves the original head array ready for the forward numerical sweep. The cap and exact-boundary comparisons avoid integer-product overflow.

Preflight requires $O(Tb)$ work and reuses the directory storage. The numerical pass still has a linear visit allowance, with each exact arm bounded by the existing eligibility condition. Both the independent-uniform expected bound and the deterministic general bound from #10 continue to hold. The wider allowance is a performance choice, independent of correctness.

## The observed restart

For independent 8-bit logits with bit probability 0.8 at $T=8192$, seeds 160309 and 771317 produce 587,987 and 563,169 causal pair visits. Both counts exceed the old limit of 524,288 and fit the new limit of 1,048,576. In a separate instrumented first-order run on #10, both seeds went through three rejected numerical paths (two-symbol, packed suffix, general trie) before the residual computation. At $T=4096$ the same seed labels yielded 143,009 and 166,026 visits and the two-symbol path completed immediately. [Reference diagnostic trace](../benchmarks/results/bigram_preflight_reference_trace.txt)

## Paired results

The reference is PR #10 at `c3a9368`. The candidate is `34c57da`. Both run the same public CPU exact backward on the same inputs and proposal seeds. Native rows alternate seven timed calls per checkout after warm-up on one pinned Xeon Platinum 8573C thread. The tables use geometric means of the per-input median ratios (reference time / candidate time). Absolute times below are medians of the per-seed medians. The PyTorch step also includes projections, forward, loss and an SGD update. Each length generates independent inputs from its seed and shape. A length sweep can change the selected structural path as well as the amount of data.

| Group | Faster / measured | Geometric mean | Slowest | Fastest | Samples |
| --- | ---: | ---: | ---: | ---: | --- |
| Independent Gaussian, 4 widths, 3 lengths, 2 seeds | 27 / 48 | 1.03× | 0.88× | 1.30× | [JSON](../benchmarks/results/bigram_preflight_grid.json) |
| Biased 0.8, same grid | 29 / 48 | 1.25× | 0.84× | 6.61× | [JSON](../benchmarks/results/bigram_preflight_grid.json) |
| Biased 8 bits, 4 lengths, 3 fresh seeds | 21 / 24 | 2.63× | 0.98× | 6.74× | [JSON](../benchmarks/results/bigram_preflight_holdout.json) |
| Full PyTorch mixer steps, 4 shapes, 3 seeds | 12 / 24 | 1.00× | 0.84× | 1.13× | [JSON](../benchmarks/results/bigram_preflight_layer.json) |

### Fresh skewed inputs

| Length | First-order ms (#10 → preflight) | Second-order ms (#10 → preflight) | First / second speedup |
| ---: | ---: | ---: | ---: |
| 6144 | 16.22 → 16.21 | 20.00 → 18.62 | 0.99× / 1.06× |
| 8192 | 160.77 → 25.19 | 165.22 → 30.31 | 6.37× / 5.56× |
| 10240 | 215.15 → 34.51 | 215.74 → 40.69 | 6.17× / 5.46× |
| 16384 | 116.43 → 84.66 | 127.97 → 96.46 | 1.36× / 1.33× |

### Separating a larger allowance from admission

The cap-only checkout is #10 with the literal `uint64_t(n)*64` in `bigram_credit.h` replaced by `uint64_t(n)*128`. It still reduces until the limit is exhausted. Each column below comes from a direct interleaved comparison, on the same two seeds and inputs. Each cell is first / second order.

| Length | Cap-only vs #10 | Preflight vs cap-only | Preflight vs #10 |
| ---: | ---: | ---: | ---: |
| 4096 | 1.11× / 1.12× | 0.93× / 1.02× | 0.94× / 0.98× |
| 8192 | 6.55× / 5.69× | 0.95× / 0.99× | 6.55× / 5.79× |
| 16384 | 0.85× / 0.88× | 1.59× / 1.54× | 1.28× / 1.29× |

At length 8192 the wider allowance lets the first reduction finish. Much of that speedup is already present in the cap-only run. At length 16384 the larger allowance still gets exhausted; counting before reducing then avoids paying for discarded gradients. On already admissible inputs the count can be overhead. The change does not promise a speedup on independent Gaussian inputs or full mixer steps. [Cap-only vs #10](../benchmarks/results/bigram_preflight_cap_only.json) · [Preflight vs cap-only](../benchmarks/results/bigram_preflight_ablation.json)

### Numerical coverage

All native rows preserved forward and V/embedding bytes, joint proposal codes and numerical joint costs within tolerance. The largest combined Q/K relative L2 discrepancy was 1.228e-16. Across the 24 PyTorch steps the largest absolute differences were 0 for outputs, 0 for gradients and 0 for updated parameters.

The exhaustive suite passed 72 tests with one CUDA transfer test skipped on this CPU-only host. Native ASan/UBSan ran the two-symbol and training ABI oracles with leak detection disabled. New tests verify admission at the actual direct and projected list sizes, rejection one visit below the boundary, and that a rejected plan never calls its numerical reducer. [Validation](../benchmarks/results/bigram_preflight_validation.json) · [Sanitizers](../benchmarks/results/bigram_preflight_sanitizers.json)

## Reproduction

Run from the PR checkout with an available CPU index. The second reference worktree reproduces the cap-only ablation.

```bash
git worktree add --detach ../rosa-reference c3a9368202912dd8b6903801e8b52e49e897fd8d
git worktree add --detach ../rosa-cap-only c3a9368202912dd8b6903801e8b52e49e897fd8d
sed -i 's/remaining=uint64_t(n)\*64;/remaining=uint64_t(n)*128;/' ../rosa-cap-only/conv_rosa_transformer/csrc/subquadratic/bigram_credit.h
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
opts=(--reference-checkout ../rosa-reference --cpu 7 --repeats 7)
python benchmarks/random_qkv.py "${opts[@]}" --families gaussian biased --lengths 4096 8192 16384 --widths 5 8 12 16 --seeds 160309 771317 --output build/bigram_preflight_grid.json
python benchmarks/random_qkv.py "${opts[@]}" --families biased --lengths 6144 8192 10240 16384 --widths 8 --seeds 333067 589651 938537 --output build/bigram_preflight_holdout.json
python benchmarks/compare_layer_step.py "${opts[@]}" --seed-offsets 0 7919 104729 --output build/bigram_preflight_layer.json
python benchmarks/random_qkv.py --reference-checkout ../rosa-cap-only --cpu 7 --repeats 7 --families biased --widths 8 --lengths 4096 8192 16384 --seeds 160309 771317 --output build/bigram_preflight_ablation.json
(cd ../rosa-cap-only && python benchmarks/random_qkv.py --reference-checkout ../rosa-reference --cpu 7 --repeats 7 --families biased --widths 8 --lengths 4096 8192 16384 --seeds 160309 771317 --output build/cap_only.json)
python tests/run.py --exhaustive --output build/validation.json
python tests/sanitize.py --no-leaks --tests subquadratic_bigram subquadratic_training --output build/sanitizers.json
```
