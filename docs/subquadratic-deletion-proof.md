# Exact-combinatorial subquadratic Q/K deletion

Deletion proof, 2026-11-06. This document concerns the **shared deletion contribution**. The complete engine is derived in [the full proof](subquadratic-proof.md). Its native implementation is [delete.h](../conv_rosa_transformer/csrc/subquadratic/delete.h).

## Result and numerical contract

For strings Q,K of length n, payload vectors P and row gradients g of dimension D, the deletion contributions of **both Q and K**, with the original longest-match/latest-endpoint order, can be computed in

$$
 O\!\left((nB+n^2/B)(D+\log^3 n)+n\log^4 n\right)
$$

time for any integer 1 <= B <= n. Taking B = ceil(sqrt(n)) gives deterministic

$$
 O(Dn^{3/2}\operatorname{polylog}n).
$$

Materializing all events would use O(n log n+nD+n²/B) machine words. The native streaming schedule uses **O(n log n+nD+B²D+n/B)** machine words; at B=ceil(sqrt n), this is **O(n log n+nD)**. Section10a gives the owner-bucket schedule and its proof. No global event list is retained. Section5a additionally answers arbitrary Q owner/time requests from the same plateau stream, without repeating its generation.

All string equality tests, endpoints, lengths, ties, occurrence queries, and combinatorial decisions are exact. The numerical reduction is an identity over the reals, using additions, subtractions and scalar products only. An implementation in binary64 changes the order of summation and therefore does **not** promise bitwise equality with a row-by-row oracle or with production's old event order. No FFT, truncation, sampling or approximate string matching is used.

The native implementation uses the streaming schedule. The [independent C++ oracle](../tests/cpp/subquadratic_delete.cpp) constructs literal deleted routes and does not reuse the AP representation or A/H crossing formulas.

## 1. Definitions and exact index primitives

All indices are zero based. At row t, valid K endpoints satisfy 0 <= r < t. Set

$$
 F_t(r)=\operatorname{LCSuffix}(Q[0..t],K[0..r]).
$$

The baseline is the lexicographically greatest pair (L,r0) = max_r(F_t(r),r), except that length zero is always normalized to endpoint -1. A positive route ending at r reads payload P[r+1]; an empty route reads e0. The baseline row score is beta_t = g[t] · P[r0+1], or g[t] · e0 for an empty baseline. The desired deletion credit of an owner is the sum over rows of deleted score minus baseline score.

The standard suffix-array index on Q, separator, K, separator supports:

* LCE and LCS in O(1), using the forward and reversed concatenations, LCP arrays and RMQ.
* The suffix-array interval I(t,h) of the length-h Q suffix ending at t in O(log n), by binary search with LCP queries.
* The first or last K start position in a numerical interval [a,b] whose suffix-array rank belongs to I(t,h), in O(log n), using a wavelet matrix over suffix-array order containing K start positions.

The native radix-doubling suffix array takes O(n log n) construction time; LCP sparse-table and wavelet storage take O(n log n) words. Faster suffix-array/IPM machinery is unnecessary for the bound. The displayed overall theorem conservatively allows extra logarithmic factors.

Write

$$
 R_t[h;c]=\max\{r<c:F_t(r)\ge h\},\qquad R_t[h]=R_t[h;t].
$$

A predecessor query on K starts with inclusive upper bound c-h gives this value, then adds h-1. This detail prevents an endpoint equal to the causal cutoff from leaking in.

For `BEST(t,c)`, the reversed Q suffix has a maximum LCP with one of its two lexicographic neighbors among reversed K suffixes with endpoints below c. A second wavelet matrix finds those eligible predecessor/successor ranks; the longer LCP gives the optimal length, and one R query gives the latest endpoint at that length. This takes O(log n). `BEST(t,t)` is the saved baseline; `BEST(t,s)` is A(s) below.

All indices are lazy. A full-word match at a requested position boundary already certifies the first/last occurrence. Precomputed same-symbol run lengths can decide an LCS/LCE by inspecting at most two run boundaries; unresolved cases use RMQ. These checks have a fixed operation bound on every input. The optional exact query cache is allocated only after such certificates fail, compares its complete key, and cannot change a route on a collision. Full-word dictionaries use bounded probes plus ordered lookup, keeping their worst-case bounds independent of hash luck.

### Baseline substring reduction

For a nonempty baseline, the following identity is exact:

$$
 F_t(r)=\min\{L,\operatorname{LCSuffix}(K[0..r0],K[0..r])\}
 \quad (r<t).
$$

Indeed, the last L symbols of Q[0..t] equal those of K[0..r0], so matches shorter than L agree in the two comparisons. No causal endpoint can have F_t(r)>L by the baseline definition. Thus K deletion depends on the selected K substring, cap L, and causal time. The algorithm below retains Q in the index because its periodic suffix extent has an especially direct description; this does not alter the reduction.

## 2. Local occurrence progressions

Fix a row with L >= B and let W=Q[t-B+1..t], of length B. Every endpoint with F_t(r)>=B corresponds to an occurrence start x=r-B+1 of W in K. Causality is x<=t-B.

Partition those possible starts into blocks

$$
 [jB,\min(jB+B-1,t-B)].
$$

There are floor(t/B) such blocks. In one block, every potential occurrence is contained in a K fragment of length at most 2B-1. A standard internal-pattern-matching fact says that occurrences of a pattern x in a fragment y with |y|<2|x| form one arithmetic progression. We use exactly this combinatorial fact, not an optimal IPM oracle.

**Primary source:** Kociumaka, Radoszewski, Rytter and Waleń, *Internal Pattern Matching Queries in a Text and Applications*, Fact 1.1, [arXiv:1311.6235](https://arxiv.org/abs/1311.6235), [full text](https://arxiv.org/pdf/1311.6235). The full-text Fact 1.1 states the one-progression property under the strict factor-two length condition.

Three index queries recover each block's progression: first start x, second start y>x, and last start z. If x is absent, skip the block; if y is absent, it is a singleton. Otherwise p=y-x, and all starts are x,x+p,...,z. Necessarily 1<=p<B. The corresponding endpoints form a,a+p,...,b, where a=x+B-1 and b=z+B-1.

No occurrence is enumerated to recover this progression.

The native index can extend the progression beyond its discovery window. Let
$x<y$ be its first two starts and $p=y-x$. One same-string LCE certifies that
$K[x..y+\operatorname{LCE}(K[x..],K[y..]))$ is $p$-periodic. Every full
pattern occurrence inside this run has phase $x\bmod p$: an occurrence in a
different phase could be shifted left by multiples of $p$ to a start strictly
between $x$ and $y$, contradicting the definition of the second occurrence.
The AP therefore extends through the last causal full-pattern start in the
verified run. Search resumes past the larger of the original window and
that run's last permissible start; the original-window AP fact covers the
case where the run stops early. This only skips verified occurrences, so the
$O(n/B)$ discovery bound and the length profile below remain valid.


## 3. One exact exception to a whole periodic profile

For a string X and endpoint j, define E_p(X,j) as the maximum length of a p-periodic suffix ending at j, capped by the beginning of X. For j>=p,

$$
 E_p(X,j)=p+\operatorname{LCSuffix}(X[0..j],X[0..j-p]).
$$

The formula follows immediately from the equalities X[i]=X[i-p] required inside a p-periodic substring.

For the progression a,a+p,...,b above, set

$$
 h_Q=E_p(Q,t),\quad h_K=E_p(K,b),\quad \ell=b-h_K+1.
$$

### Lemma 1: all progression endpoints have the same periodic left boundary

For every r in the progression,

$$
 E_p(K,r)=r-\ell+1\ge B.
$$

**Proof.** The first two occurrences overlap with shift p, so W is p-periodic. All consecutive occurrences overlap with the same shift p, hence their union from a-B+1 through b is p-periodic. Thus the maximal periodic suffix at b starts at ell<=a-B+1. Restricting that suffix to an earlier endpoint r shows E_p(K,r)>=r-ell+1.

If ell=0, it cannot extend farther. Otherwise maximality at b means K[ell-1]!=K[ell+p-1]. These two positions still lie on or before every progression endpoint because r-ell+1>=B>p. Therefore the periodic suffix ending at any r cannot extend to ell-1 either. This proves equality. ∎

### Lemma 2: the LCS profile is capped affine except at one endpoint

For every progression endpoint r other than possibly

$$
 r_* = \ell+h_Q-1,
$$

we have

$$
 \boxed{F_t(r)=\min(h_Q,r-\ell+1).}
$$

If r* belongs to the progression, obtain its exact F_t(r*) with one ordinary LCS query. There is at most one exception per block.

**Proof.** The last B symbols match, hence the last p symbols match, and both periodic suffix extents are at least B. Periodicity propagates this equality leftwards until the shorter periodic suffix ends. Thus F is at least the minimum of the two extents.

Suppose h_Q<r-ell+1. If the Q suffix reaches the beginning of Q, F cannot exceed h_Q. Otherwise Q's symbol immediately before that suffix differs from the symbol p positions later, by maximality of its periodic extent. K is still inside its periodic suffix at the corresponding earlier position. The two symbols p positions later already match, so the earlier symbols must differ. Hence F=h_Q. The symmetric argument applies when h_Q>r-ell+1.

Only equal extents escape this mismatch argument. Equality is r=ell+h_Q-1, a unique numerical endpoint. A genuine LCS query handles any further matching context there. ∎

A block therefore becomes at most three ordered pieces: an ordinary progression before r*, a singleton r*, and an ordinary progression after r*. On each ordinary piece, F(r)=min(h_Q,r-ell+1) is nondecreasing.

### Concrete exception which invalidates a tempting shortcut

Take

```
Q = [4,4,4,4,4,2,0,1,0,1]
K = [3,2,0,1,0,1,0,1,3,3]
t = 9, B = 3.
```

W=[1,0,1] occurs at starts 3 and 5 in start block [3,5], giving endpoints 5 and 7 and period p=2. Here h_Q=4, h_K=6 and ell=2. The exceptional endpoint is r*=5: F_t(5)=5, whereas the ordinary formula gives 4. At endpoint 7, F_t(7)=4. Keeping only the latest endpoint would lose the length-5 route. Keeping the latest endpoint **and the exact exception** is sufficient.

No assumption that p is a primitive period is needed in either lemma.

## 4. A subquadratic bound for the entire R staircase

For h>=B, ordinary progression endpoints other than the last one can be discarded from the computation of R_t[h]. Their F is no greater than the last ordinary value, and their endpoint is earlier. The actual last endpoint b has F at least its ordinary formula, even if it is itself exceptional. Therefore the only necessary candidates from a block are b and its exceptional endpoint r*, if present.

Visit these O(t/B) pairs (F,r) by decreasing r. Starting at high-length threshold B-1, append a step [previous_F+1,F] whenever F increases. This is exactly the latest-endpoint envelope for all h>=B. The native AP generator already emits increasing endpoints, so reversing its candidate list supplies this order without another sort.

Consequently the number of distinct R steps, including h<B, is at most

$$
 (B-1)+2\lfloor t/B\rfloor.
$$

Choosing B≈sqrt(t) proves O(sqrt(t)) steps per row and O(n^{3/2}) steps overall. This is an upper bound on the **original uncompressed R staircase itself**. Its previously stated quadratic worst-case ceiling is loose.

The bound can be tight in order. Let K begin with zero runs of lengths m,m-1,...,1, separated by ones, then add a suffix of ones of comparable total length. Let Q consist entirely of zeros. For Theta(m²) rows after all these runs are causal, R has m different steps, so the total is Theta(n^{3/2}) for n=Theta(m²).

## 5. Q deletion

Deleting Q owner u at row t caps surviving match length at h=t-u. For 0<h<=L, the winner is (h,R_t[h]); h=0 is empty; h>L leaves the baseline unchanged.

For any h with r=R_t[h], the exact plateau extends through F_t(r). Indeed, a later endpoint matching any length in [h,F_t(r)] would already match h and contradict the definition of r. This gives the following bounded generator.

1. Add the empty-route difference to owner t.
2. Starting at h=1, query r=R_t[h], set z=min(L,F_t(r)), emit the owner interval [t-z,t-h], and continue at h=z+1. Stop immediately if r is the baseline endpoint, since all remaining endpoint deltas are zero.
3. Perform at most B such jumps. If any lengths remain, their starting threshold h is greater than B. Use the progression construction from Section4 at this actual threshold; its cost is O(n/h), hence O(n/B). Each step [a,b] with endpoint r emits the constant scalar g[t]·P[r+1]-beta_t on [t-b,t-a].

A scalar difference array contracts all fixed intervals in O(1) updates per interval, followed by one owner prefix scan. Including h=L is harmless: its endpoint is the baseline and its contribution is zero. One can omit it in an implementation.

This still yields O(B+n/B) emitted intervals per row, with polylogarithmic index work and O(D) numeric work per interval. It can jump over a long plateau in one step without assuming that the whole input has any particular form.

## 5a. All Q time ranges from the same plateau stream

Let M requests ask for the deleted score at owner u over an arbitrary time interval [l,r]. A plateau generator emits delta_t on an owner interval [a,b], where the delta is relative to the original row score beta_t. A range-add/point-query Fenwick tree maintains

$$
V_T(u)=\sum_{t<T}\bigl(\operatorname{score}(\operatorname{DelQ}(u,t))-\beta_t\bigr).
$$

Also maintain the scalar baseline prefix B_T=sum_(t<T) beta_t. The requested value is

$$
\boxed{V_{r+1}(u)+B_{r+1}-V_l(u)-B_l.}
$$

Sort two signed snapshots per request by time. Before consuming row t, answer every snapshot at time t; then feed that row's plateau intervals to the Fenwick tree. An interval addition is two point updates in the tree's difference representation. A snapshot is one prefix query at its owner. These operations are O(log n); sorting the M events contributes O(M log M).

The same emitted interval also updates an ordinary scalar difference array. Its final owner prefix gives the complete Q deletion credit. Thus total Q deletion and every requested negative repair projection share one generation pass, with no retained row-by-owner profiles and no repeated time-block scans. With B≈sqrt(n), the sweep costs

$$
\widetilde O((D+1)n^{3/2}+M)
$$

time and O(n+M) extra scalar storage, beyond the shared exact index and inputs. Unrequested gaps may be skipped when only range projections are requested: they cancel out of every future snapshot difference. When the full deletion credit is also requested, all rows still feed the ordinary difference array.

The native base case directly evaluates at most four rows for each short request. Charge O(M_0 D) for M_0 such requests if D and M are arbitrary. For the complete engine, M≤nb and fixed b,D, this is absorbed by the displayed asymptotic bound. Expansion accumulators protect the sweep's snapshot differences from avoidable cancellation; their scalar representation follows the fixed-format contract in the full proof. They do not claim equality with every row-by-row binary64 evaluation order.

## 6. K deletion and the short-owner split

Only owners

$$
 s\in[first,r0],\qquad first=r0-L+1,
$$

can affect a nonempty baseline. At such an owner,

$$
 A(s)=\max_{r<s}(F_t(r),r),
$$

$$
 H(s)=\max_{s\le r<t}(\min(F_t(r),r-s),r),
$$

and the deleted route is max(A,H), using the original lexicographic order.

For the last B owners,

$$
 s\in[\max(first,r0-B+1),r0],
$$

compute the route directly by indexed queries. A(s)=BEST(t,s). The greatest H length is the largest h for which R_t[h]>=s+h, obtained by binary search. Its endpoint is R_t[h]. This costs O(log³ n) per short owner, for at most nB owners overall.

The native short fringe groups certified constant endpoints before these point queries. On either the A-winning or H-winning region, the selected endpoint is nondecreasing in owner order. Equal endpoints at both ends of an interval certify a constant payload if that endpoint lies outside the interval: it lies before every owner, forcing the A region, or after every owner, forcing the H region. Otherwise split and query recursively. This is confined to at most B fringe owners, so failure of every certificate still incurs only O(B) indexed queries per row.

The remaining **long-owner interval** is [first,end], where end=r0-B. It may be empty. For every s in that interval, the baseline endpoint by itself retains H length r0-s>=B. Therefore no endpoint with original F_t(r)<B can win on this interval. All candidates needed for the remaining work are represented by the progression pieces of Sections 2–3.

## 7. Compressed prefix A envelope

Process the ordinary AP pieces and exceptional singletons in increasing endpoint order. Initialize the current prefix winner with BEST(t,first), which includes all endpoints r<first.

For an ordinary piece with endpoints a,a+p,...,b and F(r)=min(h_Q,r-ell+1), first clip to endpoints r in [first,end-1]. Let f0 be the current record length.

* If h_Q<f0, this whole piece cannot set a record.
* Otherwise the first new-record endpoint is the first progression point at or beyond max(a,ell+f0-1).
* Every subsequent endpoint in this piece sets a record because F is nondecreasing, and equality still improves the latest endpoint.

Thus the entire surviving run of records is represented by one tuple (a',b,p,ell,h_Q). Its first owner is a'+1. The prefix record endpoint at owner s is

$$
 r_A(s)=\min\left(b,\ a'+p\left\lfloor\frac{s-1-a'}p\right\rfloor\right).
$$

For an exceptional singleton r with exact length f, emit a record exactly when f>=f0, then update f0. An exception whose length exceeds h_Q automatically prevents subsequent ordinary endpoints of that piece from being emitted until some later piece catches up.

Each record lasts until one owner before the next record begins. Within that owner range, split at b+1: up to b+1 it has the periodic endpoint formula; from b+2 onward it is the fixed endpoint b. There are O(t/B) records and splits per row.

**Short-length caveat.** If the initialization has length below B, omitted low-F endpoints may change A before the first represented high-F endpoint. We do not claim that the initial stale record gives those low-length A values. They cannot win against H>=B on the long-owner interval. Once A has length at least B, its envelope is represented exactly. The crossing search below uses the exact index, so it does not rely on that stale initial low-length record.

Pseudocode:

```
current = BEST(t, first)
records = [fixed initial record starting at first]
for piece in increasing endpoint order:
    clip piece endpoints to [first, end-1]
    if singleton (r,f):
        if f >= current.length:
            append fixed record starting at r+1
            current = (f,r)
    else ordinary (a,b,p,ell,hQ):
        if hQ < current.length: continue
        a = first AP point >= max(a, ell+current.length-1)
        if a > b: continue
        append periodic record starting at a+1
        current = (min(hQ,b-ell+1),b)
close each record immediately before the next record's owner start
```

No implicit A endpoint is expanded.

### Why explicit A records can be quadratic

For Q=K=010101..., at row t the matching endpoints have the same parity as t and satisfy F_t(r)=r+1. Every such endpoint is a new prefix record, so each row has Theta(t) explicit A records. Moreover A wins on a suffix of the affected owner interval of Theta(t) size, and those winning records remain Theta(t) in number. Summed over rows, explicit A enumeration is Theta(n²), even when R has very few steps. Constant-symbol-run compression does not remove this period-two family. The progression representation does.

## 8. A/H crossing and exact H intervals

At any positive A length f, every H endpoint is at least s, whereas every A endpoint is below s. Thus H wins all equal-length ties. A wins precisely when H cannot support length f, equivalently

$$
 R_t[f]<s+f\quad\Longleftrightarrow\quad s+f\ge R_t[f]+1.
$$

As s increases, A's length is nondecreasing, R_t[f] is nonincreasing in f, and s+f increases. Therefore the predicate is monotone and has one crossing. On the long-owner interval use

```
A_wins(s):
    (f,r) = BEST(t,s)
    return f >= B and R(t,f) < s+f
```

Binary-search the first true owner, or the sentinel end+1 if none. Its cost is O(log⁴ n) per row. The condition f>=B explicitly excludes the irrelevant initial low-length A part.

The native implementation reuses the AP scan instead of repeatedly invoking BEST inside that binary search. On one ordinary piece, its capped-affine record formula gives A(s); the already built R staircase answers R_t[f]. Test the piece's end first, skip a losing piece, and binary-search only the first piece that can win. This last search uses integer formulas and staircase lookup, with no new suffix queries. When R has a single plateau, the floor-affine crossing inequality can instead be solved directly for its first feasible progression block and owner. The partially consumed AP scanner is then passed to the winning A stream. The monotonicity argument is exactly the one above.

For a high R step [a,b] with endpoint r, H selects endpoint r exactly on the owners

$$
 [R_t[b+1]-b,\ r-a].
$$

When b=L, there is no higher length; replace the lower bound by first before clipping. Proof: H reaches some length at least a on this step iff r-s>=a. It reaches no length at least b+1 iff R_t[b+1]<s+b+1, which for integer positions is s>=R_t[b+1]-b. These are exactly the two bounds. The selected H length inside the interval is min(b,r-s), and the latest endpoint remains r.

Clip these constant-endpoint H intervals to [first,cross-1]. Clip the compressed A records to [cross,end]. Together with the short owners, they cover every affected owner exactly once, with the original tie rule.

## 9. Numerical contraction of all periodic endpoints

The only nonconstant route pieces left by K deletion are owner intervals [lo,hi] with one row t, a period p<B and a residue c. Their payload index is

$$
 k(s)=s-((s-c)\bmod p),\qquad c=(a'+1)\bmod p.
$$

They contribute g[t]·P[k(s)] at each owner. The baseline subtraction can be accumulated separately as one fixed range per row. Fixed alternative routes use ordinary scalar difference updates.

It is essential to process **all residues of a period together**. Processing each residue by an n-position scan would introduce an unnecessary nB² term.

For a fixed p, let G_c(s) be the vector sum of g[t] over active intervals with residue c. Each interval produces two coefficient events: +g[t] at lo and -g[t] at hi+1 (omit an end event at n). We need

$$
 Z(s)=\sum_{c=0}^{p-1} G_c(s)\cdot P[k_c(s)].
$$

Sort all events for this p by owner. Partition owners into blocks starting at a=jp, with offsets r=0,...,p-1 (the final block may be shorter).

### State present before the block

Let A_c=G_c(a-1), before events at owner a. On the block,

$$
 k_c(a+r)=
 \begin{cases}
 a+c,&c\le r,\\
 a-p+c,&c>r.
 \end{cases}
$$

Define U_c=P[a-p+c] and V_c=P[a+c]. For the first block extend negative-index payloads by zero; actual routing intervals never need an invalid negative index. The contribution of A throughout the block is

$$
 base=\sum_c A_c\cdot U_c,
$$

$$
 Z_0(a+r)=base+\sum_{c\le r}A_c\cdot(V_c-U_c).
$$

Compute all p outputs with one prefix scan, at O(pD) cost.

### Events inside the block

An event of vector v at owner a+e affects only one residue c, from offset e onward. Its selected payload changes at most once inside the block:

* If e<c, add v·U_c on offsets [e,c-1], and v·V_c on offsets [c,p-1].
* If e>=c, add v·V_c on offsets [e,p-1].

Clip to the final block's actual length. These are at most two scalar range updates to a block-local difference array, costing O(D) per event. Also update A_c += v so that the state is correct for the next block. A final block-local prefix scan adds all event corrections to Z0.

This is an exact decomposition of every interval's contribution; the event order at equal owners is immaterial over the reals. No convolution is involved.

For one period, block setup costs O(nD), and all events cost O(M_pD) after sorting. Across at most B-1 periods, this is

$$
 O(nBD+MD+M\log M),\qquad M=O(n^2/B).
$$

A coefficient array of size pD is reused from block to block. The major extra memory is the stored event list, not the block scratch.

### Compact contraction pseudocode

```
for p in occurring periods:
    events = sorted(events_for_all_residues_of_p)
    A[0:p,0:D] = 0
    for block a=0,p,2p,...:
        m = min(p,n-a)
        U[c] = payload[a-p+c], or zero if the index is negative
        V[c] = payload[a+c] for c<m
        base = sum_c dot(A[c],U[c])
        Z[0:m] = base + prefix_sum_c(dot(A[c],V[c]-U[c]))
        correction_diff[0:m+1] = 0
        for event (a+e,c,v) in this block:
            if e<c:
                range_add(correction_diff,e,min(c-1,m-1),dot(v,U[c]))
                if c<m: range_add(correction_diff,c,m-1,dot(v,V[c]))
            else:
                range_add(correction_diff,e,m-1,dot(v,V[c]))
            A[c] += v
        output[a:a+m] += Z + prefix_sum(correction_diff)[0:m]
```

For a last partial block with c>=m, no current-block payload V_c is accessed; an event for that residue uses only U_c. The C++ oracle covers this partial-block boundary.

## 10. Complete time and space accounting

| Work | Count | Cost per item / total |
|---|---:|---:|
| Forward/reverse suffix indices | 2, built on demand | O(n log n) build, O(n log n) words |
| Baseline and A initialization | O(n) | Saved forward routes / O(log n) indexed queries |
| Bounded Q plateau jumps | O(nB) | O(log n + D) |
| Exact short K owners | O(nB) | O(log³ n + D) |
| Local occurrence start blocks | O(n²/B) | Three O(log² n) occurrence queries, O(1) LCS calls |
| AP pieces, high R/A records, fixed intervals | O(n²/B) | O(log n + D) with sorting included |
| A/H binary searches | O(n log n) queries | O(n log⁴ n) total |
| Periodic intervals and events | O(n²/B) | O(D) contraction and O(1) owner-bucket queue work |
| All-residue block scans | At most B-1 periods | O(nBD) |
| Final owner scan | n | O(n) |

Taking B=ceil(sqrt(n)) proves the stated worst-case subquadratic deletion bound, for every input string, including nonconstant periodic inputs. Randomness is unnecessary.

Q and K share baseline routes and suffix indices; their accumulation buffers are separate. Q's full deletion and its time-range requests additionally share the same plateau emission, as proved in Section5a.

The bulk-event variant uses O(n log n+nD+n²/B) space, plus O(n/B) row-local structural records. The streaming variant below uses O(n log n+nD+B²D+n/B) space. Storing the full affected owner-row route matrix is diagnostic and is not part of either claimed algorithm.

## 10a. Streaming the events in near-linear working memory

The event list is unnecessary. Two scheduling changes preserve all formulas above.

### Simultaneous owner sweep for all periods

For every p<B retain its p-by-D coefficient array A, current block's p scalar base outputs Z, p+1 scalar correction differences, one running correction scalar, and current block start. Total coefficient storage is

$$
 D\sum_{p=1}^{B-1}p=O(B^2D).
$$

At each owner s, first initialize the current block for every p dividing s, using coefficients from preceding owners. Then consume all events whose owner is s. Such an event updates A immediately and adds its one-switch correction to the current p-block's scalar difference array. Finally consume each period's base Z plus the current correction prefix at this owner. Events at the block's first owner must be processed **after** that block is initialized; otherwise their contribution would be counted twice.

The block identity from Section9 remains unchanged: A at block entry supplies the contribution of already active intervals, and within-block event corrections supply precisely the difference after each start or end. When the next block starts, A already contains those events, so the old correction array is discarded. A final partial block uses the same zero/partial boundary rules. Block initialization across all p costs O(nBD), and one per-period scalar consumption per owner costs O(nB). Event processing costs O(D) each.

### One lazy A iterator and one reusable event slot per row

Compute a row's high R records, H intervals and exact A/H crossing as before. Apply H's fixed-payload intervals to the scalar owner difference array, then discard this row's R/AP scratch. If an A-winning suffix exists, initialize its A route exactly by BEST(t,cross). This length is at least B. Re-query the occurrence start blocks forward, beginning with the block containing endpoint cross; at most one AP block's constant number of pieces is held at a time.

Within an ordinary AP piece, the same capped-affine profile and record rule produce the current record or skip it. A record's final owner bound is determined by the next record, so the iterator holds one current record and one look-ahead record. It emits disjoint owner intervals in increasing order. A fixed tail updates the scalar owner difference array immediately. A periodic interval yields its start and end to the event scheduler.

Every event has an integer owner in [0,n), and a row's next event never precedes the current owner. Store n bucket heads and one reusable event slot plus a next pointer per row. Popping a start frees that row's slot for the corresponding end; popping an end frees it for the next interval's start. Each insertion and removal takes O(1). A following start may be at the same owner as the previous end; exhaust the current bucket, including newly inserted events, before consuming that owner's numeric result. Fixed contributions may be discovered after their owners have already been visited in the periodic sweep, because their independent scalar difference array is prefix-summed only at the very end.

The total number of occurrence blocks inspected is O(n²/B); passing the scanner from the crossing search also avoids repeating its consumed prefix. The generators hold O(1) state each, the buckets hold at most O(n) pending events, and only one row needs O(n/B) AP/R scratch at a time. Event scheduling takes O(n+n²/B). No expanded endpoint array or global event array is retained.

Combining the exact suffix index, input/output arrays, current period blocks, row-local scratch and owner buckets gives O(n log n+nD+B²D+n/B) words, with the same overall time bound. Q deletion applies scalar interval updates row by row; its optional time-range Fenwick tree is a separate O(n+M) workspace.

### Validation of the streaming schedule

The native oracle checks full Q/K deletion credits, every short threshold on exhaustive binary strings, arbitrary owner/time projections, noncontiguous selected rows, high-bit symbols, and numerical dimensions independent of symbol width. It also checks the exact randomized cache against uncached answers.

The restricted-row API accepts original time indices and a reusable global suffix index. It builds no stream for rows outside that set. This yields the row-block scheduling used for arbitrary time-range projections in the [complete proof](subquadratic-proof.md).

## Engineering and numerical validation

The same native code is used by the NumPy and ordinary autograd interfaces. [The engine guide](subquadratic-engine.md) explains the bounded verified cache, monotone staircase certificates and constant-factor choices. These reduce work without changing the deletion proof.

Native workspace counters are a **buffer capacity estimate** from tracked reserved arrays. The asymptotic word-storage bounds above are mathematical statements; the counters are not process RSS measurements or guaranteed overall allocator/runtime peaks.

Binary64 aggregation changes addition order. The tests compare mathematical route semantics and use scale-aware numerical tolerances; they do not establish a universal relative-error bound for heavily cancelled inputs. Standalone address/undefined-behavior sanitizers and complete backward benchmarks are separate engineering gates.
