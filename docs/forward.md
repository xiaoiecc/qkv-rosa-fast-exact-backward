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
at most T fallbacks and 2T transition attempts. This single trajectory does not need the suffix and numerical indexes used
to share work across all backward edits.

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
the complete state number and all 64 symbol bits. Each lookup inspects at most **16
table slots**. If those slots are occupied by other keys, lookup and insertion use
an overflow `std::map` keyed by the full `(state, symbol)` pair. Collisions can change
the work but cannot change a match or cause an unbounded probe chain. Capacity is at
least twice the maximum edge count. Each consumed symbol's hash is reused across
fallbacks, and state salts are cached.

There are O(T) transition-dictionary operations in SAM construction and in the single
query trajectory. Each operation costs O(log T) in the worst case because of the
ordered overflow map. General endpoint maintenance also costs O(T log T), so the
complete bound is **O(T log T) time and O(T) auxiliary space for every seed**, without
an average-hashing assumption. If the transition operations are settled by the flat
table and endpoint certificates settle all queries, construction and scanning take O(T); the complete
forward is not claimed unconditionally linear. Randomness changes the frequency of
the exact overflow lookup, not the worst-case guarantee.

## Validation and historical timing evidence

[Independent checks](../tests/test_forward_lean.py) compare causal routes with literal matching and test the fallback bound and lazy endpoint allocation. The route API remains `rosa_core.matching_stats(q, k)`; [forward.h](../conv_rosa_transformer/csrc/forward.h) implements the matcher used by the native forward.

The [forward.json receipt](../benchmarks/results/forward.json) records a 2026-10-25 forward-only experiment, including 21 input cases, timing samples, input identities, and source/binary hashes. Its results belong to that recorded build and scope: they are historical measurements, not a newly measured complete training step. The comparison implementation is an external historical source, not a shipped runtime backend.

The retained [forward benchmark](../benchmarks/forward.py) can compare the current forward with an explicitly supplied external checkout through `--baseline-dir`. Its public scope includes input validation and conversion of route arrays to Python lists; native scope includes construction and matching. Neither includes payload aggregation, projections, loss, or optimization. Follow [the current benchmark guide](../benchmarks/README.md) when reporting a fresh run.

Current backward and training measurements are in [subquadratic-results.md](subquadratic-results.md). The [main algorithm guide](../ALGORITHM.md) explains how the backward shares work across all bit edits; it uses the same longest/latest causal routing definition.
