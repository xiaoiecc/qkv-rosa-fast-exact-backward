# Exact bit credit on retained causal coordinates

The complete CPU backward runs one Q/K reducer for both credit orders. Its inputs are the packed streams, the ordinary forward routes, and the numerical projection for each row. Bit flips keep their physical labels throughout the computation. The public layer also computes V and embedding gradients and the selected joint corners.

## Which time positions can disappear?

Write an edge from query position $u$ to key position $s$ if $s<u$ and their complete words are equal or differ by exactly one editable bit. Every symbol pair in every exact or singly repaired causal suffix must be an edge. A position with neither an incoming nor an outgoing edge can never occur on either side of such a suffix. Contract each consecutive gap of these positions to one separator, keeping Q and K in the same increasing order.

A valid match lies inside two contiguous retained blocks. Conversely, a match in contracted time cannot cross a separator. Its symbol pairs are still causal in the original order, so a separator cannot match any one of them even after the edit. Lengths, edit owners and the order of tied endpoints are therefore preserved. A removed output row has zero Q/K credit. At retained endpoint $r'$ the implementation stores the payload that originally followed $r$, including when that payload belongs to a removed time. Joint flips are evaluated on the original streams because two edits can cross a separator that one edit cannot.

The support pass uses two complementary projections of the editable bits. Every edge agrees on at least one projection. A light bucket visits at most eight candidates and checks their complete words. A heavy bucket marks all earlier endpoints through its most recent requesting time. This can retain extra coordinates, and it caps discovery work at $O(T)$ for every input. A directly addressed small alphabet can use the identity coordinate map. The numerical reducer always uses the same longest/latest rule.

For uniform independent $b$-bit words, let $E$ count the causal edges. By linearity of expectation,

$$
\mathbb E[E]=\binom{T}{2}\frac{b+1}{2^b}.
$$

If the light postings resolve all buckets, the compact map needs at most $4E+1$ coordinates. Each edge marks at most two positions and gaps add at most one more position per marked position plus one. For example, $T=8192,b=32$ gives about 0.258 causal edges on average. The returned dense gradients still cost $\Omega(Tb)$ to write. The [support experiment](../benchmarks/interval_support.cpp) records the actual retained sizes, including conservative bucket marks and the identity-map threshold.

## Which route wins after one flip?

For a row with original route $B=(L,e)$, fix the stream, owner and bit. Let $S$ be the best original suffix avoiding the owner. Let $U$ be the better of $S$ and all one-symbol repairs, and $W$ the better of $U$ and all longer repairs. Each maximum compares length, then endpoint. If $f_t$ is the first-order or diagonal score at row $t$, the contribution is

$$
[f_t(S)-f_t(B)]+[f_t(U)-f_t(S)]+[f_t(W)-f_t(U)].
$$

The first term clips an exact occurrence before or after the owner. For a Q owner of age $a<L$, the surviving route has length $a$ and the latest endpoint matching that suffix. For K only the winning interval $[e-L+1,e]$ can suffer damage. Short rows query causal postings. Long rows use the repository's deletion primitive. Both produce the first term alone.

When the winning suffix consists of one repeated word and every causal occurrence of that word is in one run $[A,e]$, deleting $s$ inside the winner leaves two pieces. The right piece keeps endpoint $e$. The left piece wins precisely when $s-A>e-s$, and its endpoint is $s-1$. Capping the left length at $L$ preserves this comparison inside the winning interval. In the left-winning interval the returned payload is $V_s$. Difference arrays accumulate the row weights over this interval and evaluate one dot product per owner. With identical constant Q and K the singleton reducer takes $O(Tb)$ at fixed numerical dimension proportional to $b$.

For the second term, an edit at the last Q word can use the latest matching K word. A K singleton repair can beat the survivor only when $L\le1$ and the edit is later than the existing endpoint. If $L\ge2$, damaging an interior word leaves a nonempty suffix ending at $e$ that wins a length-one tie. Flipping the terminal word itself cannot make that same terminal match. For a given edited K word, the one-symbol contribution therefore sums row weights by query class until the next exact occurrence resets the class.

For a longer repaired alignment ending at $r$, let $a$ be the exact common suffix length ending at $t,r$. The only possible changed words are $Q_{t-a}$ and $K_{r-a}$. When they differ by one editable bit, their exact repaired length is $a+1+\operatorname{LCSuffix}(t-a-1,r-a-1)$. The same candidate serves a Q edit and a K edit. Only the best route per physical edit survives the row. Floating reductions can reorder additions.

## How many candidates are inspected?

K word classes get compact ranks, with a separate digit for Q words absent from K. Only real edges of the original full words generate edited digits. A rolling packed word indexes up to $D$ suffix digits. At a row, the lookup depth is $p=\min(t,\max(2,\min(D,L+1)))$. Exact and one-edited pivot postings cover all repaired alignments of length at least $p$. For Q, the existence of a shorter edited suffix is monotone in its length, so one binary search finds its longest/latest occurrence. For K there are only two ways a shorter repair can win. It can tie at length $L$ at a later endpoint or touch the original winning interval. In the latter case the endpoint is within $p-2$ positions of that interval's right edge or inside it. Two disjoint exact pivot witnesses certify that every K edit retains length at least $p$, eliminating the shorter search for the row.

Let $M$ be the retained length, $J$ the total postings visited by the pivot, late ties and the bounded short-deletion queries, and $H$ the sum of lengths of local damaged windows inspected. At fixed word width the reporting work is output-sensitive in $M+J+H$, up to logarithmic factors. The long-row deletion primitive also has its own cost. Packed arm comparisons that extend beyond a machine word use exact indexed suffix queries. The packed-suffix dictionary verifies full keys, and predecessor queries use exact suffix ranks. The engine never discards a partial gradient to run another exact backward.

On unrestricted inputs $J$ and $H$ can reach quadratic order. For fixed bit width, numerical dimension and joint budget the conservative time upper bound for this implementation is $\widetilde O(T^2)$. The $\widetilde O(T^{5/3})$ unrestricted time guarantee belongs to the earlier structural construction.

## Expected work for independent uniform keys

At fixed bit width, score dimension and joint budget, the expected time of the complete algorithm is $\widetilde O(T)$ for independent uniform $b$-bit K words and any $b$-bit Q stream independent of K. V and the row weights may be arbitrary. The argument is for the supported length range with enough bits in a machine word to pack a pivot. It does not extend the independence assumption to Q and K produced by shared learned projections.

1. Condition on the complete query. Let $\sigma=2^b$ and $D_0=\lceil\log_\sigma T\rceil$. Even after contraction the pivot depth uses the original horizon and is at least $D_0$. Ranks take at most $b$ bits for these inputs and at most 28 bits in the supported length range. If $p\ge D_0$, every reported pivot has at most one edited physical bit in its last $D_0$ words. For each original row the expected number of eligible original endpoints is at most $T(1+bD_0)\sigma^{-D_0}$. Contraction preserves all such matches injectively, so the same upper bound applies to retained rows.
2. If $p<D_0$, either the pivot covers the entire retained causal prefix (at most one endpoint), or the original row has no exact suffix of length $p$. Fix a length $h$ and endpoint $r<t$. Its radius-one probability is at most $(1+bh)\sigma^{-h}$. From a disjoint partition of the first $t$ keys into length-$h$ windows, discard at most two that intersect the candidate. The others are independent of the candidate. The probability of missing the exact query word in all of them is at most $\exp[-(\lfloor t/h\rfloor-2)_+\sigma^{-h}]$. Multiplying and summing over $r$ gives $O(h(1+bh))$ in expectation per length. This bound also covers the forced length-two pivot on a row whose baseline is empty.
3. For late ties of length $h=L$, no exact length-$h$ window ends between the candidate and the row. Applying the same disjoint-window bound to the keys strictly after $r$ and summing the resulting geometric tail gives $O(h(1+bh))$ expected visits for that length. Shorter Q repairs need only a logarithmic number of dictionary lookups per edited pivot digit.
4. By a union bound over fewer than $T^2$ causal alignments, the probability of any baseline exceeding $\lceil3\log_2 T\rceil$ is at most $1/T$. On its complement every local damage interval has logarithmic length. Short-deletion posting queries skip at most $h$ endpoints at depth $h$. Any longer exact posting they visit shares the same $D_0$-word prefix from step 1. Each remaining owner query has polylogarithmic cost. A repair-free short row queries its barriers directly. The long-row deletion routine therefore runs only on the exceptional event. Even charging the unrestricted soft-quadratic bound there contributes only $\widetilde O(T)$ in expectation.
5. The support scan costs $O(T)$ and the projection and dense output passes cost $O(Tb)$ at fixed numerical width. At fixed $b$ there are at most $2^b$ full physical word classes, independent of $T$. The full-key suffix dictionaries use bounded-probe maps with ordered overflow. The causal forward refines one row at a time by a bounded number of postings or exact indexed predecessor queries. Summing over rows, lengths and a fixed number of joint corners gives the stated bound.

## Implementation and verification

- [Support discovery and edit graph](../conv_rosa_transformer/csrc/subquadratic/edit_alphabet.h)
- [Exact interval reducer](../conv_rosa_transformer/csrc/subquadratic/interval_credit.h)
- [Coordinate and payload mapping](../conv_rosa_transformer/csrc/subquadratic/engine.h)
- [Causal forward and joint replay](../conv_rosa_transformer/csrc/causal_windows.h)
- [Independent physical-flip oracle](../tests/cpp/interval_credit.cpp)
- [Paired native backward and full PyTorch timings](causal-interval-results.md)
