# A complete subquadratic algorithm for the mathematical QKV-ROSA counterfactual VJP

Structural proof, 2026-11-06. The bound implemented by the native subquadratic engine is

$$
\boxed{\widetilde O(n^{7/4})\text{ time and }\widetilde O(n^{4/3})\text{ space}}
$$

for fixed packed-symbol bit width and fixed numerical dimension. Here $\widetilde O$ hides powers of $\log(2+n)$, **not** bit width, numerical dimension, or arbitrary-precision scalar costs. The theorem computes all single-bit credits without enumerating all owner-by-row routes. It is an exact combinatorial construction and an exact arithmetic reduction. The C++ implementation follows this reduction. Its floating arithmetic does not reproduce the legacy binary64 event order bit for bit; a complexity bound also does not predict which engine is faster at a finite input size.

The compact short-repair trie and a heavy/light numerical split improve the Q-only phase to $\widetilde O(n^{5/3})$; K still determines the full $7/4$ exponent.

## 1. Problem and computational contract

Let $Q,K$ be strings of length $n$, each symbol a packed $b$-bit word. Let $g_t,P_s,e_0$ be numerical vectors of dimension $d$. The implementation motivating this work has $b=d=D\le64$, but these are separate parameters in the analysis.

At row $t$, legal K endpoints are $0\le r<t$. Define

$$
F_t(r)=\operatorname{LCSuffix}(Q[0:t+1],K[0:r+1]),
\qquad (L_t,r_t)=\max_{r<t}(F_t(r),r).
$$

The order is longest length, then latest endpoint. Every zero-length route is normalized to $(0,-1)$. Its payload is

$$
Y(r)=\begin{cases}e_0&r=-1,\\P_{r+1}&r\ge0.\end{cases}
$$

Thus the selected payload follows the K endpoint. Let $Y_t=Y(r_t)$. For every Q or K position $s$ and bit $j$, compute

$$
C^X_{s,j}=\sum_t g_t^\top\bigl(Y_t^{X(s,j)\text{ flipped}}-Y_t\bigr),
\qquad X\in\{Q,K\}.
$$

In QKV-ROSA, $P_s=(2V_s-1)\odot e_1$, where V has $d$ bits per position. Its V credits are included as well. These are counterfactual hard-output differences projected on a supplied upstream vector, not derivatives of threshold quantization and not arbitrary nonlinear downstream loss reruns. This matches `ALGORITHM.md`, Sections1–3.

All string decisions are deterministic and exact. Numerical operations are interpreted in an exact arithmetic domain with quasi-linear convolution, or in the usual exact real-arithmetic model. Integer and finite-dyadic realizations must charge their scalar bit widths; Section12 states the distinction from native floating-point execution.

Empty strings and bounded small lengths are handled directly. In the asymptotic parameter choices below, take $n\ge4$ and $1\le b\le n$; the motivating fixed $b\le64$ satisfies the latter for all sufficiently long sequences.

## 2. Shared primitives and the delete/repair identity

Build forward and reversed generalized suffix arrays for Q and K, with distinct sentinels, LCP arrays and RMQ. A straightforward deterministic construction takes $n\operatorname{polylog}n$ time and space. It provides exact LCE/LCS queries, suffix-rank prefix intervals, and numerical predecessor/successor queries on K occurrence positions. A range tree or merge-sort tree suffices; no optimal suffix-index result is needed. All auxiliary grouping and deduplication can use comparison sorting or balanced ordered maps; active corridor membership can use an ID-indexed Boolean array. Thus the stated worst-case bounds do not depend on expected hash-table performance. The C++ implementation uses sorted groups, flat trees and bounded exact query caches; cache collisions only cause recomputation.

Write

$$
R_t[h;c]=\max\{r<c:F_t(r)\ge h\},\qquad R_t[h]=R_t[h;t].
$$

The length-h Q suffix defines a suffix-rank interval; a K start-position predecessor query with the correct causal cutoff gives R. Binary search on length gives the best exact route below any endpoint cutoff. It also gives the K deletion route described below in polylogarithmic time. The independently checked [native index](../conv_rosa_transformer/csrc/subquadratic/index.h) realizes these operations without a quadratic equality table; its best-route query uses the two eligible suffix-rank neighbours to avoid binary search on length.

Deleting an owner means forbidding a match from crossing it; symbols are not shifted. An output match after one bit flip either avoids the owner, hence survives deletion, or crosses the owner, hence originally had exactly one mismatch at that position. Therefore

$$
\operatorname{Flip}=max(\operatorname{Delete},\operatorname{Repair}).
$$

The numerical credit decomposes into one deletion-minus-baseline contribution shared among all bits at that owner, plus repair-minus-deletion where repair wins. This identity is exact including empty routes and latest-endpoint ties.

A causal one-mismatch center $(u,s,j)$ satisfies

$$
s<u,\qquad Q_u\oplus K_s=2^j.
$$

Let

$$
\ell=\operatorname{LCSuffix}(Q[0:u],K[0:s]),\qquad
 a=\operatorname{LCE}(Q[u+1:n],K[s+1:n]).
$$

For $t=u+h$, $0\le h\le a$, its maximal repaired route is

$$
(f(t),r(t))=(h+1+\ell,s+h)=(t+\alpha,t+\beta),
\quad \alpha=\ell+1-u,\quad\beta=s-u.
$$

Call this its corridor. Its capacity is $a+\ell+1$, and its priority among other corridors alive at the same time is the fixed key $(\alpha,\beta)$.

## 3. Universal deletion: $\widetilde O((d+1)n^{3/2})$

The essential string fact is that occurrences of a pattern W in a text fragment of length less than $2|W|$ form one arithmetic progression. This is the occurrence observation in the introduction of Kociumaka, Radoszewski, Rytter and Waleń, *Internal Pattern Matching Queries in a Text and Applications*, [arXiv:1311.6235](https://arxiv.org/abs/1311.6235), subsequently published in SIAM Journal on Computing, [DOI10.1137/23M1567618](https://doi.org/10.1137/23M1567618); the journal's Remark1.1 describes recovery from first/second/last occurrences. Only the combinatorial fact is required; ordinary indexed first/second/last-occurrence queries recover each progression.

Fix terminal length A. For a row t, consider W=Q[t-A+1:t+1]. Divide possible K start positions into blocks of width A. Every block's occurrences fit in a fragment of length at most $2A-1$, so O(n/A) progression descriptors cover every endpoint with $F_t(r)\ge A$.

Singletons are retained as fixed endpoints with one exact LCS query. For a progression with at least two endpoints $r_0,r_0+p,\ldots,r_1$, let $h_Q$ be the maximal p-periodic suffix length of Q at t. The corresponding K periodic suffixes all have one common left boundary c. Except at the one possible endpoint $r_*=c+h_Q-1$,

$$
F_t(r)=\min(h_Q,r-c+1).
$$

The exception is evaluated by one exact LCS query. To prove the formula, propagate the matching final p symbols left while both strings remain p-periodic. If their periodic extents differ, the string whose extent ends first either reaches its beginning or has a period-breaking preceding character, while the other string is still periodic; the preceding characters then cannot match. Equal extents are the sole exception. This argument does not require the primitive period.

Two consequences are sufficient.

**R staircase.** The latest endpoint of an ordinary progression dominates all its earlier ordinary endpoints for every threshold h. Retaining that endpoint and the exact exception leaves O(n/A) candidates. Sorting by decreasing endpoint and retaining increasing F gives every R step for h≥A. There are at most A−1 short steps. Thus there are O(A+n/A) steps per row.

**K prefix envelope.** For endpoints below the deleted owner s,

$$
A_t(s)=\max_{r<s}(F_t(r),r).
$$

In one ordinary progression, the first endpoint setting a new record is found by a ceiling operation on its capped-affine profile; all subsequent qualifying progression endpoints set records. The entire run remains a single descriptor. Its latest endpoint before s is a floor-affine expression of s until the progression ends, then a fixed endpoint.

The other side of K deletion is

$$
H_t(s)=\max_{s\le r<t}(\min(F_t(r),r-s),r).
$$

The greatest feasible H length h satisfies $R_t[h]\ge s+h$. Only owners in the baseline match window can change its answer. Handle its last A owners individually with the exact index. For the remaining owners, H has length at least A, so the progression representation contains all potentially winning endpoints. As s increases, A is nondecreasing while H is nonincreasing in route order. Hence one indexed crossing search separates their winning regions. The R steps give O(n/A) fixed-payload H intervals; the prefix envelope gives O(n/A) periodic or fixed-payload A intervals. Details, including inequalities at latest-endpoint ties, are proved in [the deletion proof](subquadratic-deletion-proof.md).

Q deletion is simpler. For u≤t, cap the maximum length at h=t−u. If h=0 the route is empty; otherwise the route is

$$
(\min(L_t,h),R_t[\min(L_t,h)]),
$$

with the baseline unchanged when h≥L_t. Every R step therefore gives a fixed-payload interval of Q owners.

Fixed-payload intervals are scalar difference-array updates. A periodic K-owner interval has selected payload index

$$
s-((s-c)\bmod p),
$$

for one residue c and period p<A. Group all residues of one p together. At an owner block of width p, keep one d-vector coefficient per residue. Each payload switches from its previous block to its current block once, at that residue. A prefix sum of these p dot-product differences evaluates the whole block. Each interval start/end changes one residue coefficient and requires only O(d) scalar/vector work. Thus all intervals of one period cost O(nd+Ed), where E is their event count; summing p<A costs O(nAd+Ed), not the number of implicit endpoints.

The total deletion time is

$$
\widetilde O\bigl((d+1)(nA+n^2/A)\bigr).
$$

A streaming merge of the row-local sorted event generators avoids storing every periodic event. Its workspace is

$$
O(n\log n+nd+A^2d+n/A).
$$

Taking A≈√n yields $\widetilde O((d+1)n^{3/2})$ time and $\widetilde O(nd+n)$ workspace. Both Q and K deletion are shared among their b bit channels.

## 4. Arbitrary shifted interval correlation

For M requests, compute

$$
\operatorname{Corr}(l,r,s)=\sum_{t=l}^r g_t^\top P_{t+s}.
$$

There is no common-endpoint assumption. Partition both arrays into width-W blocks. Query boundary fragments use O(Wd) direct work. Every complete g block contributes to every shift through convolutions with the P blocks. Cache the P-block transforms, transform each g block once, multiply and sum frequency products across d channels, and perform one inverse transform per block pair.

Maintain one running prefix per shift as g blocks are processed. Two ordered snapshot events per request subtract and add the needed complete-block contribution. The full table of block-prefix-by-shift answers is never stored. Time and workspace are

$$
O\left(dn\log W+\frac{(d+\log W)n^2}{W}+dMW+n+M\right),
$$

$$
O(nd+M).
$$

With W chosen to balance the middle terms, this is $\widetilde O(dn\sqrt M+n+M)$. Requests with no complete block are entirely direct. The native implementation is [correlation.h](../conv_rosa_transformer/csrc/subquadratic/correlation.h); its independent direct-sum oracle is [subquadratic_correlation.cpp](../tests/cpp/subquadratic_correlation.cpp). An affine route interval supplies exactly one such request, with shift β+1.

## 5. Q repairs: heavy rows directly, light tails by convolution

Fix a Q owner u and bit j. Its possible sources s are causal positions with $K_s=Q_u\oplus2^j$. They all exist at age zero and only expire as h increases; their static priorities are $(\ell(u,s),s)$.

At row t, the complete edited suffix is

$$
(Q_u\oplus2^j)\,Q[u+1:t+1].
$$

For different $(u,j)$ ending at that row, their full-static K occurrence endpoint sets are disjoint. If u₁<u₂ and both suffixes ended at one K endpoint, the character aligned with u₂ would be both original Q[u₂] and flipped Q[u₂]. Equal owners with different bits also disagree. Thus the total of their occurrence counts is at most n.

Choose k≥1. A branch is heavy while its full-static occurrence count exceeds k. There are at most n/(k+1) heavy branches per row, hence O(n²/k) heavy branch-row visits in total. Their credit is evaluated directly, one row at a time.

At the first row where a branch has at most k static sources, enumerate those sources, discard s≥u, and compute their left LCS and future right LCE. The source set only shrinks, so this is the branch's only light transition. A fixed-priority heap of at most k expiring sources gives at most k affine winner intervals for its remaining tail. The total number of stored light intervals is

$$
M_Q^{\rm light}\le nbk.
$$

This is different from first enumerating the full Q envelope. The hybrid algorithm does not materialize the full heavy envelope: its bound charges heavy row visits and light intervals separately.

### Exact winner and count queries

Process Q owners in increasing order and activate K sources s=u−1. For each K symbol, maintain points

$$
(x_s,y_s)=(\operatorname{rank}(K[s+1:n]),
\operatorname{rank}(\operatorname{reverse}(K[0:s]))),
$$

with value s in a dynamic two-dimensional range tree. The current right context gives an x interval. Among active points in that interval, reverse-rank predecessor and successor of reversed Q[0:u] attain the greatest left LCP; a rectangle maximum over the corresponding left-prefix interval chooses the latest s. All operations are polylogarithmic. A separate sorted list of all x coordinates for each symbol gives the **full-static** count and, on transition, at most k source reports. No new source can become causal later for the same owner: s<u is time-independent.

This realizes the claimed O(nb+n²/k+nbk) structural work up to logarithms. Heavy counts must use the static set for the disjointness charge; the winner query must use the activated causal set. If the causal set is empty, it remains empty and the branch stops.

### Q deletion subtraction needs only one interval per branch

At t=u+h, Q deletion has length at most h. Every live repaired route has length h+1+ℓ≥h+1, so it always beats Q deletion. A branch's repair-exists times are a single interval $[u,u+\max_s a(u,s)]$. Its light part is again one interval starting at its light transition. Therefore the negative term needs at most nb Q-deletion time-interval requests, even though the positive term has nbk route intervals.

For a time block containing H rows, run only those rows of the Q-deletion fixed-interval kernel with internal A≈√n. Its cost is $\widetilde O((d+1)(H\sqrt n+n))$: each row creates O(√n) fixed updates and the n-owner output is scanned once. Stream full time blocks, keeping one prefix total per owner and two snapshot events per request. Evaluate each request's at most 2H fringe rows by indexed deletion. For N requests this costs

$$
\widetilde O((d+1)(n^{3/2}+n^2/H+NH)).
$$

Taking H≈n/√N gives $\widetilde O((d+1)(n^{3/2}+n\sqrt N))$. At N≤nb this is $\widetilde O((d+1)n^{3/2}\sqrt b)$.

Combining heavy direct credit, light positive correlation, light structural work and negative subtraction, Q repair costs

$$
\widetilde O\left((d+1)\left[
\frac{n^2}{k}+nbk+n\sqrt{nbk}+n^{3/2}\sqrt b+nb
\right]\right).
$$

For $k\asymp(n/b)^{1/3}$ and b≤n, this is

$$
\boxed{\widetilde O((d+1)n^{5/3}b^{1/3})}.
$$

It retains O(nbk)=O(n^{4/3}b^{2/3}) light route records, plus near-linear index/input/output space. The Q-negative oracle independently verifies arbitrary ranges and the automatic-win/single-existence-interval property.

## 6. Compact short K repair in $\widetilde O(nb d L)$

Choose an actual short-route cutoff L. For each K owner s and bit j, create at most L+1 **virtual** words, indexed by endpoints

$$
r=s,\ldots,\min(s+L,n-1).
$$

The word is the reverse of the edited K prefix ending at r, truncated to length min(L+1,r+1). It is stored by reference to the original K substring, one edit offset and one flipped bit; it is not copied into L characters. A comparison or LCE of two such words splits at at most two edit positions and uses O(1) ordinary LCE queries.

Sort the words and construct their compact trie from adjacent LCPs. For every edge from prefix depth p to a deeper child, paint the Q reverse-suffix-rank interval that matches p+1 characters on that edge. The edge's representative is the largest endpoint in its child subtree. The deepest applicable paint gives the best matching word, with latest endpoint among equal-length matches.

To see this even when the query diverges inside a compressed edge, all words below that edge agree with each other through the divergence; they have the same LCP with the query. Their maximum endpoint therefore implements the tie correctly. If the query proceeds to a child edge, that deeper paint takes over. Finite words and prefix words are represented by terminal trie nodes. There are O(L) paint intervals and O(L) resulting pieces on the one-dimensional suffix-rank line.

On each piece, one representative endpoint r is fixed, while its actual match length may vary with the Q row. Remove from that piece the representative's length-(L+1) prefix interval. Precisely those rows have maximum truncated match length L+1 and are marked long. Thus every surviving piece has maximum match length at most L.

### Weighted comparison to baseline

For rows with t>s+L all candidate endpoints are causal. Away from the exceptions below, the deletion route equals baseline, so the remaining task is to test the representative's LCP m against baseline $(L_t,r_t)$. For L_t>0,

$$
\mathbf1[(m,r)>(L_t,r_t)]
=\mathbf1[m\ge L_t+1]
+\mathbf1[r>r_t]\bigl(\mathbf1[m\ge L_t]-\mathbf1[m\ge L_t+1]\bigr).
$$

For L_t=0 retain only m≥1; a zero LCP is the empty route regardless of r. For L_t>L no short word can beat unchanged baseline.

Batch O(n) virtual representatives across owners and bits, in lexical order. A row's prefix condition m≥q becomes an interval [a,c) of representative ranks. Replace it with two signed weighted events (a,+w) and (c,−w), and answer prefix queries at the representative's rank. Other coordinates are its Q-rank piece, causal time cutoff, and, for tie/overlap terms, baseline endpoint threshold. The strict term uses three-dimensional dominance, and the tie/overlap terms use four-dimensional dominance. Each piece needs only a constant number of offline queries.

Weights are the vector $(g_t,g_t^\top Y_t)$. A query returns their sum; dotting the vector part with fixed P[r+1] and subtracting the scalar part gives the total repair-minus-baseline credit. Fixed-dimensional offline dominance takes polylogarithmic work per point/query per numeric channel, with batched near-linear workspace. The older five-dimensional interval-containment implementation and the newer signed-event four-dimensional implementation agree with the literal oracle; the dimension reduction improves logarithmic overhead, not the exponent.

### Exceptional cells and why the long mask is complete

Correct these O(nL) owner-row pairs, per bit:

- early causal rows s<t≤s+L;
- owners in the last L+1 positions before the baseline endpoint, s∈[r_t−L,r_t].

The first group enforces causal endpoint cutoff r<t. The second includes every owner whose deletion could differ from baseline while leaving a winning short candidate possible. If s lies farther left inside the baseline match, the unchanged baseline endpoint retains an exact tail of length r_t−s>L; deletion is already longer than a short candidate. Owners outside the baseline window do not change baseline.

At an exception, subtract any main-kernel estimate and use indexed deletion. If deletion length exceeds L, short repair cannot win. Otherwise query the causal prefix of the owner's virtual-word dictionary using persistent endpoint-cutoff predecessor/successor and prefix-maximum structures. Nearest lexical neighbors maximize LCP; a prefix interval returns the latest endpoint among that maximum. This costs polylogarithmic time, not an L-age scan.

If a possible repaired route has right age h>L, its unchanged tail after the edited owner has length h, so deletion has length >L. If h≤L and its repaired length is >L, its final L+1 symbols contain the owner and are represented by the dictionary's long mask. Thus every unobserved long possibility either prevents all short winning or is explicitly masked. A word whose LCP ends before its edit is an unchanged tail and cannot strictly beat deletion; it is harmless.

The short kernel therefore returns exactly repair-minus-deletion at rows where repair wins and its full winning length is ≤L. Its cost is

$$
\boxed{\widetilde O(nb(d+1)L)},
$$

with O(nd+n polylog n+nb) working storage when dictionaries and dominance jobs are batched. Specializing b=d=D gives an explicit D² coefficient in the nL term. No implementation claim may replace that coefficient by D.

## 7. Find every long K corridor with $O(n^2/B)$ exact anchors

Choose block size B and set **L=4B**. Enumerate every full grid-aligned Q block of length B against every causal K block start. There are O(n²/B) pairs of starts. One LCE query decides equality, so a probe does not scan B characters.

For every equal anchor, extend maximally left and right on its diagonal and inspect its at most two adjacent mismatch centers. Keep an in-bounds causal center only when its XOR has one set bit. Deduplicate centers.

If a corridor has capacity a+ℓ+1≥4B+1, at least one of its exact sides has length ≥2B. Such a side contains a full grid-aligned Q block of length B. Its matching K block is causal on the same diagonal, and extending that exact run encounters the mismatch center. Therefore every route whose repaired length exceeds L is covered. Each equal anchor creates at most two centers, so the number of retained corridors is O(n²/B), **independent of b**: a one-bit XOR labels a unique bit.

Restrict a retained corridor to actual long times

$$
t\in[\max(u,u+L-\ell),u+a].
$$

Discard it if this interval is empty. The factor four is used for coverage, not dropped during parameter balancing.

## 8. Trim a K corridor once, then take an ordinary envelope

Fix a corridor and set c=u−ℓ, so f(t)=t−c+1. Its own K deletion splits into A and H as in Section3.

Every A endpoint is <s, while the corridor's endpoint is ≥s. Equal-length A loses the latest tie. A blocks the corridor iff the exact string Q[c−1:t+1], of length f(t)+1, occurs in K ending before s. This blocker is absent when c=0. Expressed by occurrence starts, its allowed endpoint constraint only tightens as t increases, while the pattern only extends. A blocker cannot reappear after it disappears.

An H candidate of length ≥f(t) must end at z≥s+f(t), strictly later than the repaired endpoint s+t−u. It therefore blocks even at equal length. Writing x=z−f(t)+1, existence is exactly an occurrence of Q[c:t+1] at a start in the **fixed** interval

$$
s+1\le x\le c-1.
$$

The lower bound is the H cap; the upper bound is causality z<t. Appending to the pattern only removes such occurrences. This blocker also cannot reappear.

Consequently a fixed maximal one-mismatch corridor beats deletion on a suffix of its lifetime. An exact indexed deletion comparison and binary search find that suffix in polylogarithmic time. Discard a corridor that never wins. This monotonicity is local to one corridor versus deletion of its own K owner; it is not a claim about arbitrary changing route winners.

Every remaining interval has length >L and beats deletion everywhere. For each (s,j), sweep starts and one-past-ends, with a heap ordered by fixed (α,β). There are O(1) events per corridor and at most O(1) winner segments per event. Hence

$$
M_K=O(n^2/B).
$$

Each emitted winner is the full flipped route: it beats deletion, dominates other long repairs, and is longer than every short repair. It can be added to the short kernel without pairwise intersections or an additional long-over-short correction.

## 9. Contract long K positive and negative terms

A segment (s,j,[l,r],α,β) contributes

$$
\sum_{t=l}^r g_t^\top P_{t+\beta+1}
-\sum_{t=l}^r g_t^\top Y(\operatorname{DelK}(s,t)).
$$

The positive term is one arbitrary-correlation request. Section4 gives

$$
\widetilde O(dn\sqrt{M_K})
=\widetilde O(dn^2/B^{1/2}).
$$

The negative term is an arbitrary owner/time-interval deletion projection. Restrict the outer loop of the universal K-deletion kernel to a time block of H original rows, reusing its global index. With internal terminal threshold A, it creates O(HA+Hn/A) descriptors. All-period numeric contraction costs O(nAd+Ed), even though only H rows supplied its coefficients. Including the output scan gives

$$
\widetilde O((d+1)(HA+Hn/A+nA)).
$$

Taking A≈√H makes this $\widetilde O((d+1)n\sqrt H)$. Over n/H time blocks the cost is $\widetilde O((d+1)n^2/\sqrt H)$. Keep one running owner-prefix vector, not a block-by-owner table. Two events per requested interval retrieve its complete-block portion; at most 2H fringe rows are evaluated using indexed deletion. Thus M requests cost

$$
\widetilde O((d+1)(n^2/H^{1/2}+MH)).
$$

For M=O(n²/B), choose H≈B^{2/3}. Both terms become

$$
\boxed{\widetilde O((d+1)n^2/B^{1/3})}.
$$

The streaming period workspace is O(n log n+nd+Hd), plus O(M) query metadata. Since H≤n this is near-linear apart from the requests. No n²/H table is retained. [delete.h](../conv_rosa_transformer/csrc/subquadratic/delete.h) implements this scheduling; [its independent oracle](../tests/cpp/subquadratic_delete.cpp) checks arbitrary query boundaries.

## 10. Full accounting, parameter choices, and memory

The table exposes b and d. Polylogarithmic factors are suppressed only to keep the powers readable.

| Phase | Time, up to polylogarithms | Retained non-index records |
|---|---:|---:|
| Baseline, exact suffix/range indices | n + nd | n polylog n + nd |
| Shared Q and K deletion | (d+1)n^(3/2) | near-linear workspace |
| Q heavy rows | (d+1)n²/k | streamed |
| Q light construction | nbk | nbk |
| Q light positive correlation | d n sqrt(nbk) | nbk |
| Q repair deletion subtraction | (d+1)n^(3/2)sqrt(b) | nb |
| K short repair, actual cutoff4B | nb(d+1)B | O(n)-word batches |
| K exact anchors, trimming, envelope | n²/B | n²/B |
| K long positive correlation | d n²/B^(1/2) | n²/B |
| K long deletion subtraction | (d+1)n²/B^(1/3) | n²/B |
| V credits and e0/e1 gradients | nd | nd |
| Copy shared deletion to all bit outputs | nb | nb |

For fixed b,d, choose

$$
k=\max(1,\lceil(n/b)^{1/3}\rceil),
$$

$$
B=\min(\lfloor n/4\rfloor,\max(1,\lceil n^{3/4}\rceil)),
\quad L=4B,\quad H\asymp B^{2/3}.
$$

This clips L≤n, handles small constant regimes, and has B=Θ(n^(3/4)) asymptotically. Q costs $\widetilde O((d+1)n^{5/3}b^{1/3})$. K short costs $\widetilde O(nb(d+1)n^{3/4})$; K negative costs $\widetilde O((d+1)n^{7/4})$. All remaining terms have smaller n exponent. This proves the full fixed-b,d $\widetilde O(n^{7/4})$ theorem.

An optional dimension-aware balance takes B≈(n/b)^(3/4), with the same clipping. For b≤n, the displayed table then gives $\widetilde O((d+1)n^{7/4}b^{1/4})$, with Q below it by a factor at most (b/n)^(1/12). This is a mathematical parameter choice; it does not erase the nb(d+1)B work of the compact kernel or establish a good finite-size constant. The simpler fixed-B accounting explicitly retains D² when b=d=D.

Process Q and K phases sequentially. The Q light list has O(nbk)=O(n^(4/3)b^(2/3)) records. The K corridor/query list has O(n²/B) records. Correlation uses O(nd+M) words, short-dominance batches use O(nd+n polylog n), and the deletion stream uses O(nd+n polylog n). Outputs occupy O(nb+nd). Thus a valid conservative total space bound is

$$
\widetilde O(n+nd+nb+nbk+n^2/B).
$$

At the fixed-dimension parameters this is $\widetilde O(n^{4/3})$. The algorithm is **not claimed linear-space** as a whole. The older method that retains the entire Q envelope instead uses $\widetilde O(n^{3/2})$ space. No quadratic owner-by-row table, explicit approximate-occurrence table, or complete-correlation table is part of either claimed algorithm.

## 11. V, embeddings, and a fixed diagonal row cost

V cannot change routes. All its credits follow from one route scatter:

$$
C^V_{s,j}=-2(2V_{s,j}-1)e_{1,j}
\sum_{t:r_t+1=s}g_{t,j}.
$$

The ordinary gradients of e0 and e1 also use the baseline route scatter. Their O(nd) work does not change the exponent. Elementwise orientation and Bernoulli/identity logit mappings add O(nb+nd), assuming their scalar operations are charged in the usual way.

The same construction handles a supplied diagonal quadratic row cost

$$
\phi_t(y)=g_t^\top y+\tfrac12\sum_c h_{t,c}(y_c-Y_{t,c})^2.
$$

Set a_t=g_t−h_t⊙Y_t and

$$
w_{t,\mathrm{extra}}=\tfrac12\sum_c h_{t,c}(e_{0,c}^2-e_{1,c}^2).
$$

Lift every nonempty payload to X(P)=(P,0) and the empty payload to X(e0)=(e0,1). Since every nonempty signed payload has componentwise square e1²,

$$
\phi_t(y)=(a_t,w_{t,\mathrm{extra}})^\top X(y)+\text{a term depending only on }t.
$$

The row-only term cancels in every flip-minus-baseline and repair-minus-delete difference. Hence d+1 numerical channels suffice; feature construction is O(nd). A convention with the opposite penalty sign uses the corresponding signed h. This extension is conditional on the supplied fixed row coefficients and the stated signed-payload set; it does not turn a g-dependent training surrogate into a linear VJP in arbitrary g.

There are two distinct V conventions. If the same supplied quadratic cost is applied to V flips, add $2e_{1,j}^2\sum_{t:r_t+1=s}h_{t,j}$ to the displayed first-order V formula. The native training entry applies the route penalty to Q/K only and explicitly keeps V linear (`training_impl.h`: “The route penalty protects Q/K. V remains the original linear credit.”). Reproducing that native convention uses the first-order V scatter above, with no h contribution. Both choices take O(nd); neither changes the Q/K algorithm or exponent. The native tests use the linear-V convention.

### A fixed-budget joint training stage can be appended

The existing native training entry computes its trust coefficient from each row of g in O(nd), selects proposal rows, and uses fixed candidate caps8/2/8 with a retained-candidate capacity8J, where J is the requested joint-pair budget. These facts are visible in `conv_rosa_transformer/csrc/training_impl.h` and `pair_candidates.h`. Its initial sorting is O(n log n); with the stated fixed caps, the sampled candidate work is at most near-linear in the number of inspected rows, with O(log(J+1)) heap work per admitted candidate. This proposal routine remains a bounded selection policy; it does not enumerate all useful pairs.

Once J pairs have been selected, the singleton corners are already present in the Q/K output arrays. For each joint corner, construct an exact suffix/range index for the two edited strings, compute all n baseline routes by indexed longest/latest queries, and project their payload changes. This deterministic implementation costs Otilde(Jn(d+1)), without needing the original forward matcher’s average-hash-probe assumption. Pair interpolation/coupling and final elementwise logit maps add O(J+nb+nd). For fixed J, and fixed proposal caps, this is smaller than the full single-bit bound. If J grows with n, its Jn(d+1) term must remain in the accounting.

Thus the mathematical fixed-budget native training rule can be composed with this single-bit core without restoring a quadratic term. The native training entry composes the core, ordinary embedding scatters, and the existing joint proposal rule. Its complete joint corners use the same deterministic suffix index. Arbitrary joint edits and all-pairs counterfactuals are not covered by the single-center corridor theorem. Dependence of row selection and trust coefficients on g also means the resulting native update is not a fixed map additive in arbitrary incoming gradients.

## 12. What exact arithmetic means, and what it does not

The combinatorial algorithms preserve every endpoint and longest/latest tie exactly. Their numerical formulas use ring identities: distribute dot products, sum signed range weights, convolve, and add interval contributions.

For integer inputs with |g[t,c]|<2^w_g and |P[s,c]|,|e0[c]|<2^w_p, a single output credit has magnitude below

$$
2nd\,2^{w_g+w_p}.
$$

A signed exact result therefore needs O(w_g+w_p+log(nd)) bits. With the diagonal extension, apply this accounting to the actual lifted coefficients and payloads after exact feature construction; their widths also contain the supplied h widths and the products/sums used by the lift. Exact NTT/CRT must use a total modulus large enough to distinguish this range, and its modular arithmetic and number of primes must be charged. The native engine supplied here uses floating FFT, not NTT/CRT or arbitrary-precision arithmetic.

Finite dyadic inputs can be scaled to integers, but the common scaling exponent and resulting integer bit width depend on the input representation. They are not unit-cost machine words merely because the original values were floating-point. Conversely, under a fixed finite floating-point format these widths are bounded by format constants plus log(nd), so an exact multiprecision algebraic realization preserves the n exponent while potentially having large constants.

The production definition can instead require a specific sequence of rounded row dot products, rounded subtractions, and rounded accumulated events. Rounding is not a linear map. Convolution and weighted vector aggregation generally do not preserve those events bitwise, especially across numeric channels. Exact expansion of already rounded convolution coefficients does not fix that difference. This note proves exact mathematical credits and exact string choices; it does **not** prove bitwise identity with the existing floating-point runtime.

The correlation oracle tests accuracy under reassociation. This does not establish training convergence or a universal relative-error tolerance: nearly cancelled sums can lose small residuals. The implementation rejects nonfinite lifted coefficients rather than enabling fast-math assumptions. Full training integration and speed have separate tests and benchmark receipts.

## 13. Implementation and independent checks

The proof is implemented in the following files. Tests construct literal edited strings or direct numerical sums independently of the accelerated formulas.

| Component | Implementation | Independent check |
|---|---|---|
| Exact indexing, deletion, owner/time projections | [index.h](../conv_rosa_transformer/csrc/subquadratic/index.h), [delete.h](../conv_rosa_transformer/csrc/subquadratic/delete.h) | [subquadratic_delete.cpp](../tests/cpp/subquadratic_delete.cpp) |
| Q heavy/light and anchored long K repair | [repair.h](../conv_rosa_transformer/csrc/subquadratic/repair.h) | [subquadratic_repair.cpp](../tests/cpp/subquadratic_repair.cpp) |
| Compact short K repair | [short_k.h](../conv_rosa_transformer/csrc/subquadratic/short_k.h) | [subquadratic_short.cpp](../tests/cpp/subquadratic_short.cpp) |
| Numerical interval contraction | [correlation.h](../conv_rosa_transformer/csrc/subquadratic/correlation.h) | [subquadratic_correlation.cpp](../tests/cpp/subquadratic_correlation.cpp) |
| Centered diagonal lift and complete native rule | [engine.h](../conv_rosa_transformer/csrc/subquadratic/engine.h), [training_impl.h](../conv_rosa_transformer/csrc/training_impl.h) | [test_subquadratic_engine.py](../tests/test_subquadratic_engine.py), [test_native_training.py](../tests/test_native_training.py) |

The engine uses exact support certificates to omit terms that are provably zero. These certificates reduce work without changing the above worst-case bounds. In particular, a virtual K endpoint must have a causal matching right context after its edited bit, and two equal endpoint values at the limits of a monotone R staircase certify an entire plateau. No input family is recognized or assigned a separate kernel.

The theorem concerns the compressed mathematical credit, not an explicit owner-by-row output table. A request to emit every individual edited route would itself have quadratic output size. Floating-point conditioning, approximation error and finite-size performance remain separate questions; see the bilingual [engine guide](subquadratic-engine.md).
