# One suffix frontier for exact Q/K backward

This research branch replaces the exact Q/K counterfactual dispatch in PR #10 with one algorithm. The complete public CPU training entry uses it at all supported widths. The statistical mode, model forward and selected joint-corner replays retain their existing semantics. [Implementation](../conv_rosa_transformer/csrc/subquadratic/suffix_frontier.h) · [Paired results against #10](single-frontier-results.md)

## What can a single flip change?

Let $(L,r_*)$ be the original longest/latest route at row $t$. For one endpoint $r<t$, let $a$ be the exact common suffix length of $Q[0:t+1]$ and $K[0:r+1]$. A flip on either side can create a longer match at this endpoint only if the two symbols just before the common suffix differ in exactly one editable bit. The bit and the owner are then determined. The new length equals $a+1$ plus the exact common suffix length ending just before that mismatch. A single candidate thus serves both Q and K.

If an edit destroys exact matches, the surviving suffixes still use their original symbols. For Q, a flip at age $u$ clips every original match to at most $u$. For K, a flip at position $s$ clips an original matching window ending at $r$ to $r-s$ when that window covers $s$. The row reducer takes the best surviving (length, endpoint) and compares it with the best repair of that coordinate. Equal lengths always select the latest endpoint. Winner changes are projected using the repository's original numerical credit rule.

## Short witnesses and a single cutoff

Let $A=2^b$, where $b$ is the packed symbol width, $D=\max(1,\lceil\log_A T\rceil)$, and choose $p=\min(t,L+1,D)$ once for each row. At every length $1\le h\le p$ the index stores exact classes of causal key suffixes and their occurrence lists. The same loop queries the exact class and its Hamming-distance-one neighbors.

- **Exact survivors.** At depth $h<p$, keep the latest $h+1$ endpoints, or all of them when there are fewer. A key owner can belong to at most $h$ different length-$h$ windows. An omitted endpoint can therefore never beat the latest surviving retained endpoint at this height. One latest endpoint already suffices for a clipped query match. At depth $p$ expand all exact endpoints and compute their full common suffix lengths. These records suffice for every deletion length, including lengths above $p$.
- **Repairs of length at least $p$.** The final $p$ symbols are exact or are Hamming neighbors of the original query suffix. Expand those two sets and verify the remaining arms. A longer match whose mismatch is older than $p$ appears in the exact set.
- **Short Q repairs.** For each edited offset and bit, retain the latest endpoint at the deepest shorter neighbor class hit by the edited suffix. Older offsets do not affect a shorter match.
- **Short K repairs.** Outside the original winning occurrence a short repair can win only by tying at $h=L<p$ at a later endpoint. Traverse that portion of the matching neighbor class. Inside the original winning occurrence a repair shorter than $p$ must end between its owner and at most $p-2$ positions to the right. One contiguous window covers all such ends. Compare each repair against the clipped winner, including its endpoint tie.

Each row updates the same Q/K sums. Exact records feed a common staircase of latest endpoints by match length. The staircase represents whole ranges of owners sharing an answer. A distinct-cell repair is accumulated once after taking the maximum for that edit.

## Representation and work bounds

At admissible native lengths each depth-$D$ key fits in 64 bits. If $b\ge \log_2 T$, $D=1$. Otherwise $Db<\log_2 T+b\le2\log_2 T$. For the standalone interface, immutable tag bits above the editable width are included in symbol equality and packing. Every occurrence test compares all its packed bits. A sparse level stores full keys and can enumerate neighbors from two disjoint bit bands. A one-bit difference preserves one band exactly. Within each band the code can use its known distinct class count to bound lookup work. Dense and sparse levels implement the same query. A long common arm can use an indexed longest-common-extension query after the first machine word matches. [Packed levels](../conv_rosa_transformer/csrc/subquadratic/packed_suffix_level.h)

For independent uniform keys and a query stream independent of those keys, the exact Q/K core takes expected $\widetilde O(T)$ time at fixed dimensions. For $p=D$, even the unconditional radius-one list has expected size at most $t(1+bD)/A^D\le1+bD$. When $p=L+1<D$, the strict report requires absence of any exact length-$p$ match. A later tie at length $L$ requires absence of a later exact occurrence. The fixed-depth word-avoidance bounds for these joint events already account for overlapping key windows. They give $O(1+b(1+\log_A(t+1)))$ expected endpoints for the two reports combined. At the boundary $p=t$ at most one length-$t$ key window exists. The short exact lists add at most $O(D^2)$ records per row, and $\Pr(L\ge h)\le t/A^h$ bounds the expected local window length by $O(1+\log_A(t+1))$. Lookup, verifying arms, sorting row records and applying the credits multiply these terms by at most polylogarithmic factors at fixed dimensions. The bound uses no work budget, failed-attempt charge or switch to an earlier backward engine. [Word-avoidance derivation](random-qkv.md#expected-work-of-the-strict-and-latest-endpoint-tests)

For arbitrary inputs the bound is $\widetilde O(T^2)$ time and $\widetilde O(T)$ workspace at fixed dimensions. At a single depth a key endpoint belongs to one exact class. The full-depth classes and the latest-tie classes therefore each report at most $t$ endpoints per row. The local window also costs $O(t)$. All row records are discarded after aggregation, and the index uses at most $O(TD)$ occurrence slots plus a linear-logarithmic suffix index.

## Why the worst case matters

With $Q=K$ constant, the full-depth exact bucket alone contains $t-p+1$ endpoints once $t\ge p$. The explicit report loop therefore takes $\Omega(T^2)$ time on this input. There is much less numerical information to write. For $s\ge1$, flipping $K_s$ makes the best intact left route end at $s-1$ (length $s$). The intact right route ends at $t-1$ (length $t-1-s$). The left route wins exactly for $s<t\le\min(2s,T-1)$. With $V_j$ the payload at position $j$, its entire first-order contribution is

$$
\sum_{t=s+1}^{\min(2s,T-1)}\langle g_t,V_s-V_t\rangle.
$$

Two prefix sums, of $g_t$ and $\langle g_t,V_t\rangle$, evaluate this interval per owner. This example separates a quadratic number of changed routes from the much smaller work needed for their sum. A general improvement would need to discover and aggregate such intervals or progressions inside the shared suffix representation.
