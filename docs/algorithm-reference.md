# Algorithm reference and implementation map

[中文](algorithm-reference.zh-CN.md) · [Derivation](../ALGORITHM.md) · [Full proof](subquadratic-proof.md) · [Deletion proof](subquadratic-deletion-proof.md)

The repository's one backward engine is the subquadratic construction taught in [ALGORITHM.md](../ALGORITHM.md). This page records contracts and points to the corresponding source and independent tests. It does not describe a second implementation.

## Mathematical contract

At row $t$, legal K endpoints satisfy $r<t$. Compare routes lexicographically by `(match_length, endpoint)` and normalize every zero-length route to `(0,-1)`. A nonempty route returns $P_{r+1}=(2V_{r+1}-1)\odot e_1$; an empty route returns $e_0$.

For a singleton Q/K/V bit edit $i$, the first-order target is $C_i=\sum_t g_t^\top(Y_t^{\operatorname{flip}(i)}-Y_t)$. Q/K second-order mode substitutes the fixed local score $g_t^\top\delta+\tfrac12\sum_c h_{t,c}\delta_c^2$. The native training convention keeps V linear. Ordinary embedding gradients follow the baseline route. Selected joint-bit corrections are a bounded additional stage, described in [native-training.md](native-training.md).

A forbidden owner does not shift positions. Every flipped match either avoids it or repairs its unique one-bit mismatch, giving `Flip = max(Delete, Repair)`. Deletion-minus-baseline is shared across all bits of the owner; repair-minus-deletion is added only where repair wins.

## Source and proof map

| Part | Invariant | Implementation / evidence |
|---|---|---|
| Forward and joint replay | Causal SAM trajectories; at most 16 flat transition probes, then ordered full-key lookup | [forward.h](../conv_rosa_transformer/csrc/forward.h), [forward proof](forward.md) |
| Lazy suffix index | Reuse saved baseline routes; full packed-symbol equality, bounded run certificates and exact LCP/occurrence completion | [index.h](../conv_rosa_transformer/csrc/subquadratic/index.h) |
| Prefix winner | Eligible lexicographic neighbors maximize reversed LCP; occurrence predecessor resolves latest endpoint | [index.h](../conv_rosa_transformer/csrc/subquadratic/index.h) |
| One-symbol control and complete residual | `D + R₁ + Σ(R−R₁)`; exact support certificates and a complete report budget precede evaluation | [approx_onegram.h](../conv_rosa_transformer/csrc/approx_onegram.h), [approx_row_certificate.h](../conv_rosa_transformer/csrc/approx_row_certificate.h), [approx_indexed_row.h](../conv_rosa_transformer/csrc/approx_indexed_row.h), full proof Section 2 |
| Shared deletion | Short-window occurrences form APs; full LCS profile is capped-affine except one boundary point | [delete.h](../conv_rosa_transformer/csrc/subquadratic/delete.h), [deletion proof](subquadratic-deletion-proof.md) |
| Q deletion and negative requests | One plateau stream feeds the full deletion difference array and a Fenwick time sweep; no repeated whole-row projection | [delete.h](../conv_rosa_transformer/csrc/subquadratic/delete.h), full proof Section 5 |
| K deletion contraction | All residues of a period share one block sweep; integer owner buckets retain one pending event per row | [delete.h](../conv_rosa_transformer/csrc/subquadratic/delete.h) |
| Bounded full-Q frontier | Edited cursors cannot revive; live deletion barriers number exactly `L[t]`; live-state and total-probe caps bound accepted and rejected attempts | [q_frontier.h](../conv_rosa_transformer/csrc/subquadratic/q_frontier.h), full proof Section 5 |
| Compressed Q repair | Static source sets are disjoint per row; a branch becomes light at most once; small-class envelopes merge two ordered lifetime lists | [repair.h](../conv_rosa_transformer/csrc/subquadratic/repair.h), full proof Section 5 |
| Complete near-winning K reports | Two exact halves cover long challengers; the damaged-baseline window covers shorter ones; owner buckets reuse the reports for deletion | [near_k.h](../conv_rosa_transformer/csrc/subquadratic/near_k.h), full proof Section 6 |
| Verified full-window K reports | A small global baseline bounds every causal LCS; fingerprint hits are fully verified, including occurrence predecessors; all reports share one work budget | [fingerprint_k.h](../conv_rosa_transformer/csrc/subquadratic/fingerprint_k.h), full proof Section 6 |
| Short K repair | Virtual words avoid copied substrings; compact trie pieces preserve longest/latest order | [short_k.h](../conv_rosa_transformer/csrc/subquadratic/short_k.h), full proof Section 6 |
| Long K repair | Every long one-mismatch match contains an exact grid anchor; its winning lifetime is a suffix | [repair.h](../conv_rosa_transformer/csrc/subquadratic/repair.h), full proof Sections 7–9 |
| Numerical intervals | Affine endpoint runs are shifted interval correlations; complete blocks use streamed snapshots | [correlation.h](../conv_rosa_transformer/csrc/subquadratic/correlation.h) |
| Quadratic score | Signed nonempty payloads have fixed coordinatewise squares; add one empty indicator | [engine.h](../conv_rosa_transformer/csrc/subquadratic/engine.h) |
| Native training | Frozen local diagonal and bounded joint corners; original linear V scatter | [training_impl.h](../conv_rosa_transformer/csrc/training_impl.h) |

All numerical arrays are row-major. Owner/time intervals are inclusive; suffix-rank intervals are half-open. Restricted-row deletion receives original time indices, including empty, noncontiguous, or unsorted distinct subsets. It does not renumber causality. Full owner/time range projections include unaffected baseline rows; they are not deletion-minus-baseline deltas.

## Parameters and costs

Let $b$ be packed bit width and $d$ the numerical dimension. The complete algorithm uses terminal length $A$, Q source threshold $\kappa$, long K anchor width $B$ with short cutoff $L_{\mathrm{cut}}=4B$, correlation width $W$, and deletion time-block width $H$. The cursor cap is $C=2\max(32,2b+2\lceil\log_2(n+1)\rceil)$, and the endpoint-report budget is based on $W_n=n\max(1,\lfloor n^{3/4}\rfloor)$.

| Phase | Time up to logarithms |
|---|---:|
| Baseline and lazy exact indexes | $n+nd$ |
| One-symbol control | $nb(d+1)$ |
| Accepted complete endpoint reports | $(d+1)M+nb$, with $M=O(W_n)$ |
| Verified full-window K attempt, $C_K=O(b+\log n)$ | $nC_K\log n+n(b+d)+W_n(d+C_K+\log n)$ |
| Shared deletion, $A\asymp\sqrt n$ | $(d+1)n^{3/2}$ |
| Bounded full-Q cursor attempt | $nC(d+\log n)+nb$ |
| Compressed Q heavy/light and small classes, $\kappa\asymp(n/b)^{1/3}$ | $(d+1)n^{5/3}b^{1/3}+bn^{3/2}$ |
| Shared Q deletion with $M_Q\le nb$ negative requests | $(d+1)n^{3/2}+M_Q$; direct short requests add $M_0d$ |
| Short K | $nb(d+1)B$ |
| Long K positive sums | $d n^2/B^{1/2}$ |
| Long K deletion projections | $(d+1)n^2/B^{1/3}$ |
| Fixed-budget joint replay | $Jn(d+1)$ |

These rows include alternative evaluations of the same terms. Complete residual reports require $M\le W_n$; two-half K reports require $M\le\min(4W_n,8nbL_{\mathrm{cut}})$. Both count their complete sets before materializing reports. The full-Q frontier publishes only on success; it then replaces both Q deletion and repair. Its internal `include_deletion=false` option returns just repair-minus-deletion. If it rejects, the shared plateau sweep and compressed repair complete the calculation. Shared Q deletion and its negative requests use one sweep, not two. Requests of at most four rows may be evaluated directly; $M_0$ counts those requests and keeps their dimension cost explicit.

With fixed $b,d,J$ and $B\asymp n^{3/4}$, time is $\widetilde O(n^{7/4})$. Retained records use $\widetilde O(n+nd+nb+nb\kappa+n^2/B)$ space, hence $\widetilde O(n^{4/3})$ for fixed dimensions. The fixed 32-row residual history, owner buckets and lazy bounded caches add only $O(n)$ words at fixed widths. These are complete-algorithm bounds, not measured speedups. The tilde hides logarithms, not $b,d,J$ or scalar precision. In particular, the short kernel has a $D^2$ coefficient when $b=d=D$.

Native workspace counters report a **buffer capacity estimate** of tracked storage, not process RSS or a guaranteed allocator/runtime peak. This measurement convention is separate from the asymptotic space proof.

## Numerical and randomized contracts

Route choices are exact. Complete mode computes every term of the finite-credit formula, but grouped binary64 reductions and FFT do not preserve a historical scalar reduction order bit for bit. Strong cancellation can amplify relative error. Exact randomized caching compares complete query keys and leaves this engine's arithmetic schedule unchanged; enabling it is bytewise answer-equivalent to disabling it for the same inputs and other seeds. [Exact randomization](exact-randomized.md) explains the cache and adaptive bypass.

Explicit approximate mode samples complete residual rows or numerical channels. Conditional expectation is an algebraic statement in real arithmetic, not a training guarantee or universal relative-error bound. It does not sample candidates before route maxima. [Randomized backward](randomized-backward.md) gives the estimators and their controls.

## Independent checks

| Test | What its oracle checks |
|---|---|
| [subquadratic_delete.cpp](../tests/cpp/subquadratic_delete.cpp) | Literal suffix scans, AP coverage/profile, full Q/K deletion, arbitrary time ranges, selected-row boundaries, cached equality |
| [subquadratic_repair.cpp](../tests/cpp/subquadratic_repair.cpp) | True edited strings for Q repair and long K coverage |
| [subquadratic_q_frontier.cpp](../tests/cpp/subquadratic_q_frontier.cpp) | Literal full-Q and repair-only credits, permanent death, live/probe-cap rejection, and no partial result on failure |
| [subquadratic_short.cpp](../tests/cpp/subquadratic_short.cpp) | Virtual LCP, trie/cutoff dictionaries, weighted dominance, literal short repair, complete K witnesses and reused deletion |
| [subquadratic_indexed_row.cpp](../tests/cpp/subquadratic_indexed_row.cpp) | Literal full and one-symbol edited/deleted routes, exact residual cancellations, first/diagonal scores, weighted rows and pre-existing outputs |
| [subquadratic_fingerprint_k.cpp](../tests/cpp/subquadratic_fingerprint_k.cpp) | Literal complete K credit, forced fingerprint collisions, exact predecessors, placement seeds and transactional rejection |
| [subquadratic_correlation.cpp](../tests/cpp/subquadratic_correlation.cpp) | Direct arbitrary shifted interval sums |
| [test_subquadratic_engine.py](../tests/test_subquadratic_engine.py) | Native full credits, first/second order, randomization, approximation and training integration |

Packed bit 63, `UINT64_MAX`, empty routes, independent numerical dimensions, equal-length ties, and final partial blocks are part of the boundary coverage. Small oracle programs may enumerate quadratic tables; those tables are not used by the production engine. See the [test guide](../tests/README.md) and [current measurements](subquadratic-results.md).
