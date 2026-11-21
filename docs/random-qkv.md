# Exact backward on random packed symbols

For a stream of length $n$ and fixed bit width $b$, numerical dimension and joint proposal budget, the complete CPU backward has expected $\widetilde O(n)$ time when the keys are independent uniform $b$-bit words and the queries are independent of those keys. The query sequence can even be fixed. Values and upstream gradients may be arbitrary. The same implementation retains the deterministic $\widetilde O(\min(n+E,n^{5/3}))$ time bound, with $E$ the number of equal causal symbol pairs. Word operations and binary64 arithmetic use the repository's existing cost model. The distribution is an assumption for the expected-time theorem. [Paired native and full-layer measurements](random-qkv-results.md) cover unbiased, skewed and broad-width cases. Every accepted computation uses exact symbol equality and causal latest-end ties.

## Which endpoints could change the answer?

Write $(L,r_*)$ for the longest matching suffix at row $t$ and its latest endpoint in the key prefix $K[0:t]$. Editing a Q symbol at age greater than $L$ cannot reach it with an exact tail. For the other ages, follow the original suffix through the edit, toggle the word's bit and follow its continuation. The deepest matched prefix of this reversed word gives the winner, choosing the most recent endpoint.

For a key edit inside the original winner, first compute the best route avoiding the edited position. At each depth $h$, inspect the most recent original matching endpoint. If this covers the owner $s$, it ends before $s+h$; skip up to $h$ list entries to reach the latest endpoint strictly before $s$. Compare those surviving routes. Any repair shorter than $L$ can win only if the edit damages the original winner. If the owner is $s$ and the shortened winner ends at $r$, then $s\le r<s+L$, so a local window of at most $2L$ possible endpoints covers these cases. For a global candidate compute its original tail length, then check the first mismatch and the exact arm to its left. This determines the only possible edited key coordinate. Several candidates for the same coordinate share a longest/latest comparison before the row's numerical contribution is added.

## Expected work of the strict and latest endpoint tests

Let $A=2^b$. Fix the query stream and let the keys be independent uniform symbols.
Conditioning on Q gives this setting whenever the full query stream is independent of K. There is no distributional assumption on payloads or row
weights. For row $t$, let $(L,r_*)$ be its exact longest/latest baseline.
A K edit outside this occurrence can change the endpoint only if a repaired
candidate has length at least $L+1$, or length $L\ge1$ and endpoint
$r>r_*$. A length-$h$ suffix from either test differs from the original query
in at most one of its $bh$ bits. Thus two range reports suffice: a Hamming
radius-one report at $h=L+1$ over all causal ends and one at $h=L$ over
$r_*<r<t$. Only the former is needed when $L=0$. Owners in the original
occurrence number $L$ and can be handled locally for shorter repairs.
All these implications are deterministic.

**Avoiding a word.** For any fixed $h$-symbol word $w$, let $u=A^h$.
In a block of $u+h-1$ independent key symbols the number $X$ of occurrences
has expectation one. Two overlapping starts at distance $j<h$ have joint
probability at most $A^{-h-j}$; the probability is zero if the overlap is
inconsistent. Disjoint starts are independent. Consequently
$\mathbb E X^2\le2+2\sum_{j\ge1}A^{-j}\le4$ and
$\Pr(X>0)\ge(\mathbb E X)^2/\mathbb E X^2\ge1/4$. Since $h\le u$,
independent whole blocks in any length-$m$ segment give
$\Pr(w\text{ absent})\le C\exp(-cm/u)$ for universal positive constants.
Splitting an interval at a single local window only changes the constant.

**Strict-length report.** Fix the query and a depth $h$. A candidate window
at endpoint $r$ lies in the Hamming ball with probability
$p_h=(1+bh)/A^h$. When $h=L+1$, no exact $h$-occurrence exists anywhere.
Test for absence using independent blocks disjoint from the candidate
window. Their total length is at least $t-h$, so the joint probability
is at most $C p_h \exp(-c t/A^h)$. Summing over endpoints gives
$C(1+bh)\lambda_h \exp(-c\lambda_h)$ at this depth,
where $\lambda_h=t/A^h$.

**Latest-endpoint report.** For $h=L\ge1$, no exact $(h+1)$-occurrence
exists in the first half of the key prefix. No exact $h$-occurrence can end
strictly after $r$ when $r>r_*$. Test the latter in the portion after
$\max(r,\lfloor t/2\rfloor)$, whose length is at least
$(t-r-2)/2$. These regions are disjoint. Remove the candidate window where
necessary, losing at most $h$ symbols and a constant factor. The tests and
the local hit then involve independent key symbols conditional on Q.
Summing over $r$ gives the upper bound

$$
C(1+bh)\min(\lambda_h,1)\exp(-c\lambda_{h+1}).
$$

Here $\sum_r\exp(-c(t-r)/A^h)\le C\min(t,A^h)$.
Across depths the sequence $\lambda_h$ decreases by a factor $A\ge2$.
At the crossing of 1 there are at most two boundary depths, earlier depths
have exponentially suppressed terms, and the remaining tail is geometric.
It follows that the expected combined number of global reported endpoints
at row $t$ is $O(1+b(1+\log_A(t+1)))$, with an absolute constant. In
particular there is no multiplicative alphabet-size factor. The strict-only
tie distinction matters: reporting depth $L$ indiscriminately over all past
endpoints can return many candidates that tie in length but lose on time.
Repeated and overlapping key substrings were allowed throughout.


## The finite directory, budgets and fallback

The general implementation builds the trie of reversed key suffixes to depth $D=\max(1,\lceil\log_A n\rceil)$. Each node records its latest endpoint and an occurrence chain; following the chain stops at the requested causal time. Edges are exact pairs of node and full-word symbol class IDs. A bounded-probe map with ordered overflow gives deterministic logarithmic worst-case lookup time. Reaching depth $D$ starts verification against the original strings.

For $h>D$, the strict and latest requests first report a radius-one $D$-suffix. Even after dropping the condition on $L$, this costs at most $t(1+bD)/A^D\le1+bD$ in expectation per request. For a Q edit of any given age and bit, its truncated $D$-suffix has at most $t/A^D\le1$ expected exact endings; this also holds when the edit is beyond $D$. Sum over all ages up to the deterministic guard $H=\max(1,\lceil5\log_A(n+1)\rceil)$. All longer-than-directory original matches needed for deletion come from one such exact list. On admitted streams, $M=\max_t L_t\le H$ bounds each original arm in a verification, and $2H+1$ bounds a singly edited match. Expanding those arms and testing the at most $H$ damaged owners incurs only polynomial factors in $H$, hence logarithmic factors in $n$ at fixed width. The remaining work consists of $O(nDb(H+1))$ or fewer trie operations up to logarithmic factors, the numerical projections, and writing the outputs.

Let $G=n\lfloor n^{2/3}\rfloor$ up to the minimum size of one per row. The census counts equal pairs until it exceeds $G$. When $A\ge n$ a one-level directory is unnecessary: conditioning on Q gives $\mathbb E E=n(n-1)/(2A)\le n/2$, so the existing equality-sensitive fallback already has near-linear expected time. On streams sent to a deeper general trie, the occurrence budget is $8n+\min(E,G)/4$ up to integer rounding. On exhaustion, the deterministic fallback costs $\widetilde O(n+\min(E,G))$, chargeable to the consumed records on that same input. Thus no independence between exhaustion and the cost of fallback is assumed. By union bound, $\Pr(M>H)\le n^2 A^{-H}$; even the full worst-case work on that event has negligible expectation. The fixed-width native interface uses integer positions and a size-checked node ID. With packed $b$-bit data the number of trie nodes is at most $1+n+\sum_{i<D} A^i\le 1+3n$ within the existing native length limit.

Earlier paths either complete within near-linear work or have fixed/near-linear preflight caps. Small equality joins that are accepted before the trie contain at most $4n$ pairs. The general subquadratic fallback and its space bound remain applicable. The expected bound includes building the index, candidate verification, and failed attempts, not just counting matching words. [Implementation](../conv_rosa_transformer/csrc/subquadratic/random_context.h)

## Practical reductions across bit widths

- In the two-symbol sweep, every match of length at least two after one edit has a suffix in the Hamming-radius-one neighborhood of the query's last two words. Count those exact packed pairs with a causal dictionary. For fair independent bits, the expected genuine report count is $\binom{n-1}{2}(1+2b)/2^{2b}$, without any independence assumption between overlapping windows. If wide-word one-symbol reports exceed the work limit, the same two-symbol sweep supports up to 16 bits. Numerical accumulator rows are allocated only for observed query symbols when the vocabulary exceeds the sequence length. During a reverse scan, accumulate length-one row weights by symbol until an original matching key resets that symbol's sum. This batches the many length-one key repairs. A limit on raw reports covers hash collisions and crowded buckets. [Two-symbol implementation](../conv_rosa_transformer/csrc/subquadratic/bigram_credit.h)
- For narrow words, two disjoint occurrences of the same suffix certify that one survives any single K flip. A shifted XOR localizes the edited bit. Word-sized suffix directories for widths 1 through 8, masking unused high bits on non-divisor widths, a second packed word for long repaired matches, and the strict/latest endpoint rule avoid separate per-edit dictionary searches. [Packed implementation](../conv_rosa_transformer/csrc/subquadratic/packed_context.h)
- If all baseline routes are empty, the keys have no causal exact symbol pair with the queries. Under one edit, every possible match then has length at most one. Verify the one-bit neighbors, sum their credits, and write the dense signed outputs required by the API. The short-match sweep similarly writes deletion credit directly to affected owners, avoiding spurious floating-point residuals at unrelated positions. [Native training path](../conv_rosa_transformer/csrc/training_impl.h)
- In forward and when replaying joint edits, dropping the final symbol from a matching suffix proves $L_t\le L_{t-1}+1$. A bounded exact suffix directory uses this upper limit and falls back to the original suffix automaton. PyTorch's CPU adapter copies directly into the required contiguous layout. [Forward implementation and guarantees](forward.md)
