# Computing every bit credit without visiting every edit–row pair

[中文](ALGORITHM.zh-CN.md) · [Full proof](docs/subquadratic-proof.md) · [Deletion proof](docs/subquadratic-deletion-proof.md) · [Training interface](docs/native-training.md)

The new C++ engine changes the unit of work. It represents many counterfactual routes by intervals, arithmetic progressions, and a small set of winning matches, then sums their numerical contributions in batches. For fixed bit width and numerical dimension, its worst-case bound is $\widetilde O(n^{7/4})$ time and $\widetilde O(n^{4/3})$ space. The tilde hides logarithmic factors.

Those factors and the constants matter. A lower exponent does not promise a faster training step at a particular sequence length. This is the repository's single backward engine. Historical implementations appear only as measured baselines in the results, not as selectable backends.

## 1. The quantity being computed

Let $Q$ and $K$ contain $n$ packed symbols, each with $b$ bits. At row $t$, only K endpoints $r<t$ are legal. Define

```math
F_t(r)=\operatorname{LCSuffix}(Q[0..t],K[0..r]),
\qquad (L_t,r_t)=\max_{r<t}(F_t(r),r).
```

The maximum compares lengths first and endpoints second. Every zero-length result is normalized to $(0,-1)$. A nonempty route returns the payload **after** its endpoint: $Y(r)=P_{r+1}$. The empty route returns $e_0$.

For one bit edit $i$, first-order credit is $C_i=\sum_t g_t^\top(Y_t^{\operatorname{flip}(i)}-Y_t)$.

For QKV-ROSA, $P_s=(2V_s-1)\odot e_1$. V never changes routes: one scatter gives $C^V_{s,j}=-2(2V_{s,j}-1)e_{1,j}\sum_{t:r_t+1=s}g_{t,j}$ in $O(nd)$ time. The embedding gradients use the same baseline routes.

The numerical vectors have dimension $d$. Keeping $b$ and $d$ separate helps expose costs: finding one-bit neighbors depends on $b$; taking a dot product depends on $d$. The usual layer uses the same width for both, but the internal kernels and their tests do not assume that equality.

This is a finite change in the hard component output, projected on its upstream gradient. It is not the derivative of a threshold and does not rerun an arbitrary downstream loss. Route selection always uses longest-then-latest order, never the numerical score $g_t^\top Y$.

## 2. Where the fourth, third, and second powers come from

A literal implementation tries $O(nb)$ edits. For each edit it visits $n$ output rows, tries $n$ endpoints, and compares up to $n$ preceding symbols. Its matching work is $O(bn^4)$.

Cache suffix-match lengths in a dynamic program, and each edited forward costs $O(n^2)$, giving $O(bn^3)$. Sharing unchanged comparisons and representing an edit by the mismatch it repairs removes another factor. But a program that still visits every owner and output row has committed to a quadratic set of questions. Making each answer cheap cannot remove that set.

We need a representation in which one operation answers many of those questions. There are two sources of structure: matching occurrences repeat periodically, and the same winning alignment often survives over a time interval. Neither property is a promise that the input belongs to a particular text family. They follow from exact matches wherever those matches occur.

## 3. Separate what an edit destroys from what it creates

Temporarily forbid a match from crossing the edited symbol. Keep every position where it was; this operation does not shift either sequence. Call the best surviving route **deletion**.

A match after the bit flip either avoids the owner, so deletion already contains it, or crosses the owner. In the latter case, its original alignment has exactly one mismatch, and that mismatch is the bit being flipped. Call the best such route **repair**. Therefore

```math
\operatorname{Flip}=\max(\operatorname{Delete},\operatorname{Repair}).
```

All bits of an owner share the same deletion term. Only their repairs differ. Numerically, add deletion-minus-baseline once per owner, then add repair-minus-deletion wherever repair wins.

![Subquadratic credit engine: shared deletion, Q repair, short and long K repair, and streamed numerical contraction](docs/images/subquadratic-overview.svg)

For a one-mismatch center $(u,s)$ with $s<u$ and $Q_u\oplus K_s=2^j$, let $\ell$ and $a$ be its exact left and right extensions. At time $t=u+h$, for $0\le h\le a$, its route is $(h+1+\ell,s+h)=(t+\alpha,t+\beta)$, where $\alpha=\ell+1-u$ and $\beta=s-u$.

Its lifetime is an interval. More usefully, its priority against another live repair is the fixed pair $(\alpha,\beta)$. We can sweep endpoints and maintain that priority; there is no need to recompute it at every row.

## 4. Deletion: compress the occurrences before summing them

### A staircase for Q deletion

Write $R_t[h]$ for the latest causal K endpoint whose match length is at least $h$. As $h$ increases, $R_t[h]$ can only move left.

Forbidding Q position $u\le t$ caps every suffix match at $t-u$. If the cap is zero the route is empty. Otherwise its endpoint is $R_t[\min(L_t,t-u)]$. Thus constant parts of the $R$ staircase become intervals of owners with the same payload.

For example, take `Q = xxxxababa`, `K = ababa#bay`, and row $t=8$. The baseline route is $(5,4)$ and returns $P_5$:

| Required suffix length | Latest endpoint | Payload |
|---|---:|---|
| 1 or 2 | 7 | $P_8$ |
| 3, 4, or 5 | 4 | $P_5$ |

Deleting Q owner 6 or 7 changes the payload to $P_8$; deleting owner 8 makes it empty. The other owners leave the payload unchanged, even though some shorten the selected match. One interval update handles both owners 6 and 7.

Constructing the staircase by trying every length would still be expensive. Choose a terminal length $A$, and find occurrences of the length-$A$ suffix in K start-position blocks of width $A$. Each block occupies a text fragment of length at most $2A-1$. Occurrences of one pattern in such a fragment form an arithmetic progression. First, second, and last occurrence queries recover it without listing its members. This is the occurrence fact used in [Kociumaka et al., *Internal Pattern Matching Queries in a Text and Applications*](https://arxiv.org/abs/1311.6235).

For progression endpoints $r_0,r_0+p,\ldots,r_1$, let $h_Q$ be the maximal $p$-periodic suffix of Q at $t$, and let $c$ be the common left boundary of the corresponding periodic K suffixes. Their full match lengths are $F_t(r)=\min(h_Q,r-c+1)$, except possibly at $r=c+h_Q-1$. While both strings stay periodic, their equal last period propagates left. If one periodic extent ends first, its preceding character breaks that equality. Only when both extents end together can their nonperiodic predecessors continue matching. The code measures that one endpoint directly.

The last ordinary endpoint dominates all earlier ordinary endpoints of its progression for the $R$ staircase. Keep it and the exception. Across a row, this leaves $O(n/A)$ long candidates, plus fewer than $A$ short lengths.

### K deletion has two competing sides

For K owner $s$, a surviving match either ends before $s$, or ends after $s$ and stops before crossing it:

```math
A_t(s)=\max_{r<s}(F_t(r),r),
\qquad
H_t(s)=\max_{s\le r<t}(\min(F_t(r),r-s),r).
```

As $s$ moves right, $A_t(s)$ improves and $H_t(s)$ worsens in route order. They cross once. The suffix index finds that crossing, including the endpoint tie. The same occurrence progressions then describe the two regions: fixed endpoints on the $H$ side, and fixed or periodic endpoint runs on the $A$ side. Only the final $A$ owners of the baseline match can require individual indexed queries; once the $A$ side wins with sufficient length, its stream covers that remaining fringe as well.

The next trap would be to expand a periodic run during numerical evaluation. Instead, group **all residues of one period** together. For period 3, the payloads for its three residue coefficients change like this inside one owner block:

| Owner | Residue 0 | Residue 1 | Residue 2 |
|---:|---|---|---|
| 3 | $P_3$ | $P_1$ | $P_2$ |
| 4 | $P_3$ | $P_4$ | $P_2$ |
| 5 | $P_3$ | $P_4$ | $P_5$ |

Each residue switches once. Compute its old and new dot products, then prefix-sum their differences. A block of width $p$ costs $O(pd)$ and there are $O(n/p)$ blocks: $O(nd)$ per period. Each interval start or end changes one coefficient vector in $O(d)$ work.

All deletion work is $\widetilde O((d+1)(nA+n^2/A))$; take $A\asymp\sqrt n$. Each row lazily emits its next event, and a heap merges these streams. The implementation never stores the implicit owner–row matrix or every expanded progression member. See [delete.h](conv_rosa_transformer/csrc/subquadratic/delete.h) and the [deletion proof](docs/subquadratic-deletion-proof.md).

## 5. Q repairs: use a counting argument to decide what to expand

Fix Q owner $u$ and bit $j$. Its possible K sources already satisfy $s<u$. As the right context grows, sources only disappear; new causal sources cannot arrive later for this owner.

At a fixed output row, the static occurrence sets of different edited Q suffixes are disjoint. Two different owners would require the same aligned symbol to be both original and flipped. Different bits at the same owner also disagree. Their total occurrence count is therefore at most $n$.

Choose a threshold $\kappa$. There are at most $n/(\kappa+1)$ branches with more than $\kappa$ static sources at a row. Evaluate those heavy branches directly. When a branch first has at most $\kappa$ sources, enumerate them once and sweep their expiration times. This produces at most $\kappa$ affine winner intervals for its remaining light tail.

The distinction between static and causal counts matters: the disjointness proof charges the static occurrence sets, while the actual winner must obey $s<u$. The C++ index keeps those queries separate.

The total heavy work is $\widetilde O(n^2/\kappa)$; the light list has at most $nb\kappa$ records. Every live Q repair is longer than Q deletion, so deletion subtraction needs just one existence interval per owner and bit. The implementation is in [repair.h](conv_rosa_transformer/csrc/subquadratic/repair.h).

## 6. K repairs: make short matches cheap and long matches discoverable

### Short matches become a dictionary problem

Choose a short-length cutoff $L$. For one K owner and bit, only $L+1$ nearby endpoints can produce a winning short repair that crosses the owner. Represent their edited reversed prefixes as **virtual words**: an original substring, one edit position, one bit mask, and a length. Comparing two words jumps between their at most two edits using exact LCP queries. It does not copy or scan $L$ symbols.

Sorting these words and building a compact trie partitions the Q suffix-rank line into $O(L)$ pieces. On each piece, one endpoint represents the best matching dictionary word. A query can diverge inside a compressed trie edge: all words below that edge then have the same match length, so their largest endpoint supplies the correct tie.

The engine removes every piece portion whose match length exceeds $L$. Weighted offline dominance queries sum the remaining row scores. They compare match length with baseline length, and endpoint with baseline endpoint when lengths tie. Explicit corrections cover early causal rows and owners near a baseline endpoint; an incremental endpoint-cutoff dictionary answers those corrections in polylogarithmic time. There is no extra scan through all $L$ ages per exceptional row.

This gives $\widetilde O(nb(d+1)L)$ time with bounded batches of virtual words and numeric jobs. See [short_k.h](conv_rosa_transformer/csrc/subquadratic/short_k.h). When $b=d=D$, its coefficient contains $D^2$; the exponent in $n$ does not make that factor disappear.

### Long matches contain an exact anchor

Set $L=4B$. A one-mismatch match longer than $4B$ has an exact side of length at least $2B$. That side contains a full grid-aligned Q block of length $B$.

For each Q grid block, query its suffix-array interval and enumerate only exact occurrences with causal K starts. There are at most $O(n^2/B)$ such anchors; failed block starts need no individual probe. Extend each anchor to the ends of its exact run, then inspect its two neighboring mismatch centers. Keep only one-bit XORs and deduplicate centers. This covers every long repair; `k_anchor_probes` counts the actual causal anchors processed.

A fixed corridor can beat deletion only on a suffix of its lifetime. The proof tracks the exact occurrences that would block it: extending the pattern can remove those blockers, but cannot bring them back. One binary search trims the corridor. The surviving corridors use the fixed $(\alpha,\beta)$ priority from Section 3, so an ordinary event sweep produces $O(n^2/B)$ winner segments.

There are also useful exact certificates. If there is no causal one-bit mismatch center, all repair terms are zero. If the largest baseline exact match has length $E$, any one-mismatch match has length at most $2E+1$. These are consequences of the same matching rule, used inside the engine rather than separate input-family kernels.

## 7. An interval description must lead to a fast numerical sum

An affine winner with endpoint $t+\beta$ contributes $\sum_{t=l}^r g_t^\top P_{t+\beta+1}$: an arbitrary shifted interval correlation. Expanding it row by row would throw away the structural saving.

Split time into blocks of width $W$. A query has two short fringes and a middle of complete blocks. Fringes are direct dot products. For complete blocks, cache payload FFTs, transform each gradient block once, sum channel products in frequency space, and invert one spectrum per useful block pair. Process blocks in order while retaining one running sum per shift. Two signed snapshots retrieve each query's middle. No block-by-shift answer table is stored.

For $M$ requests, the balanced cost is $\widetilde O(dn\sqrt M+n+M)$ with $O(nd+M)$ storage. The C++ planner compares block sizes using the request lengths; it can leave short requests entirely direct. See [correlation.h](conv_rosa_transformer/csrc/subquadratic/correlation.h).

Repair-minus-deletion also needs the negative term on an owner/time interval. The deletion kernels accept selected original row indices, so a similar complete-block/fringe decomposition works. Q uses $A\asymp\sqrt n$ internally. K uses $A\asymp\sqrt H$ for $H$ selected rows, giving $\widetilde O((d+1)(n^2/\sqrt H+MH))$ for $M$ requests. The engine chooses $H\asymp(n^2/M)^{2/3}$, clipped to the input length, using the **actual** request count. At worst-case $M=O(n^2/B)$ this becomes $H\asymp B^{2/3}$; sparse envelopes permit larger blocks. A single owner-prefix vector and two snapshots per request replace a quadratic table.

## 8. Put the costs together

The following table suppresses logarithms, but keeps bit width and numerical dimension visible.

| Part | Time |
|---|---:|
| Shared Q/K deletion | $(d+1)n^{3/2}$ |
| Q heavy rows | $(d+1)n^2/\kappa$ |
| Q light records and positive sums | $nb\kappa+d n\sqrt{nb\kappa}$ |
| Q deletion subtraction | $(d+1)n^{3/2}\sqrt b$ |
| Short K repair | $nb(d+1)B$ |
| Long K anchors and envelopes | $n^2/B$ |
| Long K positive sums | $d n^2/\sqrt B$ |
| Long K deletion subtraction | $(d+1)n^2/B^{1/3}$ |

For fixed $b,d$, take $\kappa\asymp n^{1/3}$ and $B\asymp n^{3/4}$, with the appropriate clipping for short sequences. Q costs $\widetilde O(n^{5/3})$ and K determines the full $\widetilde O(n^{7/4})$ bound. Dimension-aware choices use $\kappa\asymp(n/b)^{1/3}$ and $B\asymp(n/b)^{3/4}$ when $b\le n$.

Retained records occupy $\widetilde O(n+nd+nb+nb\kappa+n^2/B)$, or $\widetilde O(n^{4/3})$ for fixed dimensions. The complete engine is not linear-space. Its deletion and correlation workspaces are near-linear apart from their requested outputs. The [full proof](docs/subquadratic-proof.md) gives the parameter accounting and coverage arguments.

## 9. C++ choices that keep the bound usable

[index.h](conv_rosa_transformer/csrc/subquadratic/index.h) coordinate-compresses full `uint64_t` symbols before introducing sentinels. Bit 63 and `UINT64_MAX` remain ordinary symbols. Forward and reverse suffix arrays are shared by deletion and both repair kernels.

The suffix array uses radix doubling; LCP range minima use contiguous sparse-table levels. Wavelet matrices answer occurrence predecessor queries and causal prefix winner queries. For the latter, the lexicographic predecessor and successor among eligible reversed K suffixes attain the maximum LCP. This removes a binary search over possible match lengths from each prefix query. A final occurrence query resolves the latest-endpoint tie.

Deletion row generators retain constant state, with one pending heap event per live stream. Period buffers are allocated only when needed; equal endpoints certify constant $R$ plateaus. Two consecutive occurrences also certify a periodic run: one exact LCE extends it across discovery blocks, and no new occurrence phase can appear inside it. Adjacent pieces with the same periodic endpoint formula are coalesced, including compatible fixed tails. Discovery boundaries therefore do not become unnecessary numerical events. These choices preserve the general algorithm; none requires a catalogue of special input kernels.

The numerical lift centers on the first reachable payload `P[1]`. `P[0]` is never returned and is excluded from channel-span estimates. This prevents channel sampling from inventing noise when all reachable outputs are identical.

## 10. First order, diagonal trust, joint bits, and randomness

The same structural engine accepts a fixed diagonal quadratic row score $\phi_t(y)=g_t^\top(y-Y_t)+\frac12\sum_c h_{t,c}(y_c-Y_{t,c})^2$.

Every nonempty payload is a signed $e_1$ vertex, so its componentwise square is fixed. The quadratic term can therefore be represented by adjusted linear coefficients plus one empty-payload indicator channel. The implementation centers the features before forming them to reduce cancellation. The [training code](conv_rosa_transformer/csrc/training_impl.h) keeps V's original linear credit, supplies the Q/K trust diagonal, and appends a bounded set of joint-bit corrections. With $J$ selected pairs, indexed joint replays add $\widetilde O(Jn(d+1))$ work. An increasing $J$ must remain in the complexity statement.

The outer model still performs one forward, one loss evaluation, and one backward. Joint corner evaluations are internal component work. The diagonal is an explicit local training model, not a recovered downstream Hessian; the [training guide](docs/native-training.md) derives its interpretation.

There are two randomness contracts:

| Mode | What randomness changes | What remains complete |
|---|---|---|
| Exact acceleration | Placement in a bounded full-query-key cache | All structural queries and numerical contributions |
| Approximate backward | Selected complete residual rows, or explicitly selected numerical channels | Hard forward and route decisions of every evaluated contribution |

The exact cache checks the entire `(row, length, cutoff)` key. Collisions cause replacement or exact recomputation. A short observation window disables most probes when reuse is low and periodically tries again. No fingerprint equality decides a route, and the cache does not reorder arithmetic. Its seed can change runtime, but need not improve every workload.

With row sampling, the engine keeps full deletion and a one-symbol repair control. It samples complete remaining row corrections and applies inverse-probability weights **after** their route maxima have been resolved. Sampling candidate matches before taking their maximum would target a different quantity. Exact certificates remove known-zero residuals first. With an explicit positive channel budget, the engine instead runs the complete structural calculation on a fixed-size weighted channel sample; constant and empty terms are handled separately. In the current implementation these are two alternatives within approximate mode, not two automatically compounded sampling stages.

These estimators target the chosen credit conditionally in real arithmetic. They do not provide a universal relative-error or training-loss guarantee. Selected joint corners are restored by complete internal evaluations when needed. API choices and budgets are documented in [native-training.md](docs/native-training.md).

## 11. Three meanings of “exact”

The distinction matters when comparing this engine with the earlier one:

1. **Route exactness.** Every string comparison, causal cutoff, and longest/latest tie is exact. The randomized cache preserves this contract.
2. **Algebraic completeness.** Complete mode includes every term of the finite-credit formula. Interval grouping and convolution are identities over exact arithmetic.
3. **Floating-point evaluation.** C++ uses binary64 sums and FFTs. Grouping changes addition order, and FFTs add roundoff. Complete mode is not a promise of bitwise identity with the previous engine's reduction order. An exact sum of already rounded FFT coefficients cannot undo their earlier rounding.

Switching the exact cache on and off within this engine leaves its arithmetic schedule unchanged; that is a different, stronger bitwise comparison. Approximate row or channel sampling has an additional statistical error and must be assessed separately.

The local tests use literal edited strings and endpoint scans as independent oracles:

| Test | Main checks |
|---|---|
| [subquadratic_delete.cpp](tests/cpp/subquadratic_delete.cpp) | Exhaustive binary strings, AP coverage and full-length profiles, owner/time ranges, empty and unsorted row subsets, bit 63, independent numeric dimensions, cache bitwise equality |
| [subquadratic_repair.cpp](tests/cpp/subquadratic_repair.cpp) | Q repair envelopes and long K coverage versus literal edits |
| [subquadratic_short.cpp](tests/cpp/subquadratic_short.cpp) | Virtual comparisons, compact tries, causal dictionaries, weighted dominance, short-credit oracle |
| [subquadratic_correlation.cpp](tests/cpp/subquadratic_correlation.cpp) | Arbitrary shifted intervals and numerical contraction |
| [test_subquadratic_engine.py](tests/test_subquadratic_engine.py) | Native integration and training options |

For example, the deletion test can be run without Python or PyTorch:

```bash
c++ -O2 -std=c++17 tests/cpp/subquadratic_delete.cpp -o /tmp/rosa-delete-test
/tmp/rosa-delete-test
```

This is a CPU C++17 implementation. Index construction, dominance queries, and numerical batch setup have substantial finite-size costs. Compare complete backward and training-step timings, memory, and the chosen approximation error on the intended workload when choosing the credit order and approximation budget for a model.
