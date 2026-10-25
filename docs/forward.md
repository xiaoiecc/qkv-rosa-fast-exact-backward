# A lean, exact causal forward

[中文](forward.zh-CN.md) · [Implementation](../conv_rosa_transformer/csrc/forward.h)

## What the matcher returns

`rosa_core.matching_stats(q, k)` returns `(lengths, routes)` for two sequences of packed
`uint64` words. At row t, consider each endpoint `0 <= r < t` and its longest equal suffix:

```math
L_t(r)=\max\{\ell\in\mathbb Z_{\ge0}:\ell\le\min(t,r)+1,\;
Q_{t-\ell+1:t+1}=K_{r-\ell+1:r+1}\},
```

where slices are half-open. Choose the greatest length, breaking ties by the latest r.
Every empty match is `(0,-1)`; empty input returns two empty lists. For
`Q=K=[0,1,0,1,0]`, row 4 returns `(3,2)`: endpoint 4 is not yet causal.
The layer subsequently reads `V[r+1]` or the empty embedding, outside this routing API.
The full-K index serves batch/prefill matching; it has no incremental decoding interface.

## Why one scan needs no binary lifting

A suffix automaton (SAM) groups substrings with the same occurrence-endpoint set,
`endpos`. A transition is usable at row t exactly when its target's earliest endpoint
satisfies `first < t`; otherwise follow a suffix link and try a shorter suffix.
The stored matched length stays within the current state's length interval.

The moving causal boundary preserves the amortized argument. Remove the last symbol
from any current match: its predecessor ends at `r-1 < t-1`, so it was already legal
on the previous row. The longest length therefore grows by at most one. Every fallback
decreases it by at least one, and it never becomes negative. Across T rows there are
at most T fallbacks and 2T transition attempts. This trajectory needs neither a Q sort
nor the backward's lifting and occurrence-list indexes.

## Three proofs that can settle the latest endpoint

After a successful transition, let the state's endpoint set be E, with
`first=min(E)`, `last=max(E)`, and `count=|E|`. We already know `first < t`.

1. **The last occurrence is visible.** If `last < t`, return `last`.
2. **There are no holes.** If `count == last-first+1`, E contains every integer
   between its extremes. Its causal maximum is therefore `min(last,t-1)`.
3. **Direct extension reaches the upper bound.** With no suffix-link fallback,
   let p be the previous nonempty match's latest endpoint. Every new legal
   endpoint r has a predecessor in that previous endpoint set, so `r <= p+1`.
   If `p+1 < t` and `K[p+1] == Q[t]`, this bound is attained. After a fallback,
   the matched suffix has changed, so this certificate is disabled.

These tests never change the matched state or bypass `first < t`. They are exact
conditions available on every input, with one general query when none applies.

## Build the endpoint tree only when a query needs it

In the suffix-link tree, a state's prefix-terminal descendants are exactly its
occurrence endpoints. A DFS places them in one interval. Store only these T terminals.

On the first unresolved query at row t, place endpoints `0,...,t-1` in their terminal
slots and build a range-maximum tree bottom-up in O(T). Later unresolved queries
activate newly visible endpoints and query the state's interval. Each endpoint is
activated at most once. If certificates answer every query, the tree is never allocated.

## Randomness changes the work, never the answer

SAM transitions use a flat hash table with a per-process random seed. Lookups compare
the complete state number and all 64 symbol bits. Collisions cause additional probes,
never false matches. Capacity is at least twice the maximum edge count. Each consumed
symbol's hash is reused across fallbacks, and state salts are cached.

No universal-hashing theorem is established here; bad collisions can degrade runtime.
Assuming constant average probe cost, SAM construction and matching take O(T), while
general endpoint maintenance takes O(T log T). The complete bound is **O(T log T) time
and O(T) auxiliary space**, with that hashing assumption. Certificates remove endpoint
work when their proofs apply; the complete forward is not unconditionally linear.

## Measured forward latency

The [saved report](../benchmarks/results/forward.json) contains 21 cases, raw samples,
input hashes and binary hashes. On 2026-10-25, logical CPU 8 of an Intel Xeon Platinum
8573C ran seven samples per case, alternating old/new order. Native time includes SAM
construction, auxiliary indexes and every route. Public time also includes validation
of contiguous NumPy `uint64` inputs and conversion of both outputs to Python lists.
Both exclude payload reads/aggregation, projections, compilation and first-call warmup.

| Input | T | Native matcher (ms) | Public `matching_stats` (ms) |
|---|---:|---:|---:|
| All same | 8,192 | 0.367 | 0.634 |
| Random binary | 8,192 | 2.297 | 2.377 |
| Random 16-symbol alphabet | 8,192 | 1.893 | 2.162 |
| Random 8-bit symbols | 8,192 | 1.583 | 1.534 |
| Random 64-bit symbols | 8,192 | 0.475 | 0.788 |
| Periodic | 8,192 | 0.596 | 0.833 |
| Shadow | 8,192 | 0.750 | 1.151 |
| Random binary | 65,536 | 36.823 | 40.422 |
| Random 64-bit symbols | 65,536 | 6.220 | 8.069 |

The 64-bit case forces one quarter of Q to match the preceding K word, so
it includes real historical matches. Periodic uses period four with one K
mutation; shadow is `Q=0^T, K=10^(T-1)`. The two timing scopes were measured
separately; occasional median inversions reflect variation, not negative wrapper cost.

### Small comparison with the previous forward

The external baseline is the preceding delivered implementation's
`SamQRepairs(...,d=0)` scan, not the original upstream backward implementation.

| Input | T | Previous native (ms) | Current native (ms) | Speedup |
|---|---:|---:|---:|---:|
| Random binary | 65,536 | 59.915 | 36.823 | 1.63× |
| Random 8-bit symbols | 8,192 | 7.731 | 1.583 | 4.88× |
| Random 64-bit symbols | 8,192 | 5.374 | 0.475 | 11.31× |

Across all cases, native speedups were 1.63–11.31× and public speedups
1.48–9.26×. They are measurements of these inputs, not uniform guarantees.
Reproduce with [forward.py](../benchmarks/forward.py), supplying
`--baseline-dir <previous-checkout> --cpu <available-core> --repeat 7`.

## Checks and measurement provenance

[Independent checks](../tests/test_forward_lean.py) passed all 21,845 binary
Q/K pairs through length seven, 68 additional input/seed combinations, and
checks of the fallback bound and lazy allocation. All 21 timed cases produced
identical old/new routes. Vector-capacity counters in the report are not RSS.

The current `forward.h` matches the measured SHA256
`76a0d37bfb258de0fb9fcaa2a6ad63a91a997dc5bb98b13aaa0ea606df043c26`.
Later credit-only source changes are outside the measured whole-library
snapshot. The saved run belongs to the recorded binary and source hashes;
it must not be presented as a measurement of a later complete source tree.
