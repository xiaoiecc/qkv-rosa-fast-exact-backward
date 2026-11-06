# Exact randomization: cache work, preserve the answer

[中文](exact-randomized.zh-CN.md) · [Algorithm](../ALGORITHM.md) · [Approximate backward](randomized-backward.md) · [Measurements](subquadratic-results.md)

`randomized=True` enables a bounded cache inside the same suffix-index algorithm. It does not sample bit edits, candidate matches, rows, or channels. Complete keys are verified before a cached result is used, and a miss executes the ordinary exact query. With fixed inputs, credit rule, and other seeds, switching this option on and off leaves the complete backward outputs bitwise identical.

```python
from conv_rosa_transformer.rosa_layer import NBitQKVRosa

cached = NBitQKVRosa(128, 8, randomized=True)   # default
uncached = NBitQKVRosa(128, 8, randomized=False)
```

Copy the same parameter state and use the same forward/proposal seeds when comparing the two. The boolean changes query acceleration, not the training rule or the independent joint-bit proposer.

## 1. Which question is cached?

For row $t$, let $F_t(r)$ be its exact suffix-match length at K endpoint $r$. The repeated indexed question is

```math
R(t,h,c)=\max\{r<c:F_t(r)\ge h\},
```

with an empty answer of $-1$. The causal cutoff is clipped to $c\le t$. A suffix-rank interval identifies matches of the length-$h$ Q suffix; a wavelet-matrix predecessor query selects the latest legal K occurrence. Deletion, interval projection, and repair trimming can ask the same question more than once.

Length-one, impossible-length and latest-endpoint certificates run before memo lookup. The cache is allocated only when an actual suffix-interval query remains unresolved. Streams settled by direct certificates pay no cache allocation or hash lookup.

The cache entry stores all of `(t, h, c, result)`. A seeded integer mixer maps the three query integers to a slot. The array has power-of-two capacity between $2n$ and $4n$, so both memory and replacement work are bounded.

## 2. Why collisions cannot change a route

A lookup has three possible outcomes:

| Outcome | Action |
|---|---|
| Slot contains the complete same query key | Return its exact stored endpoint |
| Slot is empty | Evaluate the suffix-index query and store the result |
| Slot contains a different key | Evaluate the suffix-index query and replace the slot |

The hash determines placement only. It is never accepted as proof that two queries or symbols are equal. Even if every query were placed in one slot, the program would remain correct; reuse would merely be poor. Packed symbols themselves are compared through full-value compression and exact suffix/LCP structures.

The full-word dictionaries used by the one-symbol control and adjacent-pair certificate also have bounded probe chains. After at most 16 probes, they use an ordered full-key map. Suffix-symbol lookup uses bounded probing followed by sorted search. Adversarial collisions therefore preserve logarithmic worst-case lookup, not merely answer correctness. [Collision tests](../tests/cpp/bounded_hash.cpp) deliberately force these overflow cases.

Caching also leaves the numerical schedule alone. It substitutes an integer endpoint for a recomputation of the same integer endpoint; the caller then performs the same floating-point operations in the same order. This is why bitwise on/off equality is a meaningful contract here.

## 3. Avoid paying for low reuse indefinitely

A cache can slow down work that rarely repeats. The implementation measures a window of 512 probes. If fewer than 32 hit, it skips the next 2,048 probes and computes those queries directly, then observes another window. Later phases can have different reuse, so bypass is temporary.

These are implementation constants, not accuracy controls. A bypass never drops a contribution. The policy limits unproductive probing, but is not a promise of universal acceleration: hashing, slot checks, and extra memory can still cost more than the hits save. Report end-to-end timings as well as hits and misses.

The implementation is in [index.h](../conv_rosa_transformer/csrc/subquadratic/index.h). An index belongs to one stream; its mutable cache is not shared by simultaneous calls. Independent batch/group streams have independent indices.

## 4. Seeds and reproducibility

The native options separate `proposal_seed`, used by the bounded joint-bit proposal, from `repair_seed`, used for indexed query placement and as a sampling salt when approximation is explicitly requested. The adapter freezes the options with the forward state. Switching `randomized` does not change proposal seeds or the framework RNG consumption for that forward.

For first-order complete mode with no joint proposals, this cache needs no additional Torch RNG draw. Second-order proposal randomness is still present when the cache is disabled; disabling a lookup accelerator does not disable the chosen training rule. Approximate mode has its own explicit sampling behavior, documented in [randomized-backward.md](randomized-backward.md).

Fixed seeds make runs reproducible within the same implementation and numerical environment. The cache's equality guarantee is the on/off comparison with all other inputs and random choices held fixed, not a promise that different sampling seeds, compilers, or floating-point environments produce the same training trajectory.

## 5. Exact search and floating-point completeness are different contracts

The current complete engine batches interval sums and uses FFT where appropriate. Its routes are exact and no mathematical credit terms are omitted, but binary64 grouping and FFT can round differently from a historical direct reduction. That fact applies with this cache both enabled and disabled.

`backward_mode="approximate"` is the separate choice that permits a statistical estimate. It samples complete residual rows, or explicitly selected numerical channels, after the relevant route maxima are resolved. `randomized=True` alone enables neither estimator.

[The deletion oracle](../tests/cpp/subquadratic_delete.cpp) compares cache-on and cache-off outputs bytewise while checking both against literal routing. [The native integration tests](../tests/test_subquadratic_engine.py) exercise the complete training entry. Current speed and numerical error belong in [the measured results](subquadratic-results.md), not in a claim that randomization always wins.
