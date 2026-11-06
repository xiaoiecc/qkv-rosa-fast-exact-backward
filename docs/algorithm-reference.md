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
| Suffix index | Full packed-symbol equality; exact LCP and occurrence intervals | [index.h](../conv_rosa_transformer/csrc/subquadratic/index.h) |
| Prefix winner | Eligible lexicographic neighbors maximize reversed LCP; occurrence predecessor resolves latest endpoint | [index.h](../conv_rosa_transformer/csrc/subquadratic/index.h) |
| Shared deletion | Short-window occurrences form APs; full LCS profile is capped-affine except one boundary point | [delete.h](../conv_rosa_transformer/csrc/subquadratic/delete.h), [deletion proof](subquadratic-deletion-proof.md) |
| Deletion contraction | All residues of a period share one block sweep; row generators retain constant state | [delete.h](../conv_rosa_transformer/csrc/subquadratic/delete.h) |
| Q repair | Static source sets are disjoint per row; a branch becomes light at most once | [repair.h](../conv_rosa_transformer/csrc/subquadratic/repair.h), full proof Sections 5–6 |
| Short K repair | Virtual words avoid copied substrings; compact trie pieces preserve longest/latest order | [short_k.h](../conv_rosa_transformer/csrc/subquadratic/short_k.h), full proof Section 6 |
| Long K repair | Every long one-mismatch match contains an exact grid anchor; its winning lifetime is a suffix | [repair.h](../conv_rosa_transformer/csrc/subquadratic/repair.h), full proof Sections 7–9 |
| Numerical intervals | Affine endpoint runs are shifted interval correlations; complete blocks use streamed snapshots | [correlation.h](../conv_rosa_transformer/csrc/subquadratic/correlation.h) |
| Quadratic score | Signed nonempty payloads have fixed coordinatewise squares; add one empty indicator | [engine.h](../conv_rosa_transformer/csrc/subquadratic/engine.h) |
| Native training | Frozen local diagonal and bounded joint corners; original linear V scatter | [training_impl.h](../conv_rosa_transformer/csrc/training_impl.h) |

All numerical arrays are row-major. Owner/time intervals are inclusive; suffix-rank intervals are half-open. Restricted-row deletion receives original time indices, including empty, noncontiguous, or unsorted distinct subsets. It does not renumber causality. Full owner/time range projections include unaffected baseline rows; they are not deletion-minus-baseline deltas.

## Parameters and costs

Let $b$ be packed bit width and $d$ the numerical dimension. The complete algorithm uses terminal length $A$, Q source threshold $\kappa$, long K anchor width $B$ with short cutoff $4B$, correlation width $W$, and deletion time-block width $H$.

| Phase | Time up to logarithms |
|---|---:|
| Shared deletion, $A\asymp\sqrt n$ | $(d+1)n^{3/2}$ |
| Q heavy/light, $\kappa\asymp(n/b)^{1/3}$ | $(d+1)n^{5/3}b^{1/3}$ |
| Short K | $nb(d+1)B$ |
| Long K positive sums | $d n^2/B^{1/2}$ |
| Long K deletion projections | $(d+1)n^2/B^{1/3}$ |
| Fixed-budget joint replay | $Jn(d+1)$ |

With fixed $b,d,J$ and $B\asymp n^{3/4}$, time is $\widetilde O(n^{7/4})$. Retained records use $\widetilde O(n+nd+nb+nb\kappa+n^2/B)$ space, hence $\widetilde O(n^{4/3})$ for fixed dimensions. These are complete-algorithm bounds, not measured speedups. The tilde hides logarithms, not $b,d,J$ or scalar precision. In particular, the short kernel has a $D^2$ coefficient when $b=d=D$.

## Numerical and randomized contracts

Route choices are exact. Complete mode computes every term of the finite-credit formula, but grouped binary64 reductions and FFT do not preserve a historical scalar reduction order bit for bit. Strong cancellation can amplify relative error. Exact randomized caching compares complete query keys and leaves this engine's arithmetic schedule unchanged; enabling it is bytewise answer-equivalent to disabling it for the same inputs and other seeds. [Exact randomization](exact-randomized.md) explains the cache and adaptive bypass.

Explicit approximate mode samples complete residual rows or numerical channels. Conditional expectation is an algebraic statement in real arithmetic, not a training guarantee or universal relative-error bound. It does not sample candidates before route maxima. [Randomized backward](randomized-backward.md) gives the estimators and their controls.

## Independent checks

| Test | What its oracle checks |
|---|---|
| [subquadratic_delete.cpp](../tests/cpp/subquadratic_delete.cpp) | Literal suffix scans, AP coverage/profile, full Q/K deletion, arbitrary time ranges, selected-row boundaries, cached equality |
| [subquadratic_repair.cpp](../tests/cpp/subquadratic_repair.cpp) | True edited strings for Q repair and long K coverage |
| [subquadratic_short.cpp](../tests/cpp/subquadratic_short.cpp) | Virtual LCP, trie/cutoff dictionaries, weighted dominance, literal short repair |
| [subquadratic_correlation.cpp](../tests/cpp/subquadratic_correlation.cpp) | Direct arbitrary shifted interval sums |
| [test_subquadratic_engine.py](../tests/test_subquadratic_engine.py) | Native full credits, first/second order, randomization, approximation and training integration |

Packed bit 63, `UINT64_MAX`, empty routes, independent numerical dimensions, equal-length ties, and final partial blocks are part of the boundary coverage. Small oracle programs may enumerate quadratic tables; those tables are not used by the production engine. See the [test guide](../tests/README.md) and [current measurements](subquadratic-results.md).
