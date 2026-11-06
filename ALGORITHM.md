# The QKV-ROSA exact backward, from first principles

[中文推导](ALGORITHM.zh-CN.md) · [Exhaustive reference](docs/algorithm-reference.md) ·
[Usage](README.md) · [Measurements](benchmarks/README.md)

This guide explains **why the algorithm works**. It is deliberately short:
one running example, three insights, and pointers. The
[exhaustive reference](docs/algorithm-reference.md) contains every invariant,
certificate, edge case and cost parameter in full derivation order — go there
when you want to verify or modify the implementation, not when you first want
to understand it.

## 1. What the forward computes

QKV-ROSA is a *hard* retrieval mixer. One stream has two sequences of packed
`D`-bit symbols `Q` and `K`, and a `T×D` matrix of payload bits `V`. At every
output row `t`, find the longest suffix of `Q[0..t]` that also occurs as a
suffix ending at some earlier K endpoint `r < t`:

```math
F_t(r)=\mathrm{LCSuffix}(Q[0..t],K[0..r]),
\qquad (L_t,r_t)=\max_{r\lt t}(F_t(r),r).
```

Longer matches win; equally long matches are won by the later endpoint. An
empty match is normalized to `(0,-1)` and outputs a learned empty payload
`e₀`. Otherwise the output is the signed payload *one row after* the match:

```math
y_t=\begin{cases}e_0,&r_t=-1,\\ P_{r_t+1},&r_t\ge0,\end{cases}
\qquad P_s=(2V_s-1)\odot e_1 .
```

**Running example.** Take `D=1`, `Q=K=01010`. At `t=4` the candidate match
lengths at endpoints `0,1,2,3` are `1,0,3,0`. The winner is `(3,2)` — K's
suffix `010` ending at position 2 — so `y₄=P₃`.

Everything is discrete. The ordinary derivative of the hard route is zero
almost everywhere, so this layer cannot be trained by differentiating the
forward. Instead we compute a finite, counterfactual credit.

## 2. The credit definition, and why brute force is too slow

The credit of bit `(s,j)` in `X∈{Q,K,V}` is the exact output change caused by
flipping that one bit, projected onto the upstream vector `g`:

```math
\boxed{C^X_{s,j}=\sum_t g_t^\top
\left(y_t^{\mathrm{flip}(X_{s,j})}-y_t\right).}
```

This is a well-defined finite difference of the hard model — not a
straight-through approximation and not a threshold derivative. The price is
brutal: `3TD` flips, each rerunning an `O(T²)` forward, cost `O(DT⁴)` in
total.

**Example, continued.** Flip `Q₄`. The Q prefix becomes `01011`; the old
winner `(3,2)` is destroyed, and the new best route is `(1,3)`, so `y₄`
changes from `P₃` to `P₄`. With `V₃=0, V₄=1, e₁=2, g₄=3`, this single flip
contributes `3·(2−(−2))=12` to `C^Q_{4,0}`. The brute force recomputes the
entire forward — every row, every endpoint — just to learn this.

The whole algorithm answers one question: **what work is repeated between
these `3TD` reruns, and what invariants let us share it?**

| Repeated work | Invariant that removes it |
|---|---|
| Rerouting after a V flip | V never participates in matching (§3) |
| Recomputing unaffected matches for every bit | A common *deletion* route depends on the owner position, not the bit (§4) |
| Searching all new alignments per flip | Every new match *repairs* exactly one nearest mismatch (§4) |
| Emitting each deletion answer per owner | Deletion routes are monotone and compress into intervals (§5.1) |
| Searching K for edited suffixes | One shared suffix automaton, filtered causally (§5.2) |
| Advancing a unique repaired match row by row | Static uniqueness fixes its whole future lifetime (§5.3) |
| Repeating identical payload dot products | Scores depend only on row and payload class (§5.4) |

## 3. Insight 1 — V flips never reroute

V takes no part in matching, so flipping `V_{s,j}` only touches rows that had
already selected payload row `s`. All V credits fall out of one scatter along
the baseline route:

```math
C^V_{s,j}=-2(2V_{s,j}-1)\,e_{1,j}\sum_{t:\,r_t+1=s}g_{t,j},
\qquad O(TD)\text{ total.}
```

One third of the brute force is gone; no hard forward per V bit is ever run.

## 4. Insight 2 — every Q/K flip is deletion ∪ repair

Call the flipped position its *owner*. After a Q/K flip, every match in the
new routing either **avoids the owner** or **crosses it**.

- The best avoiding match is the **deletion** route. Crucially, it depends on
  the owner *position* only — all `D` bits at that owner share it.
- A crossing match must previously have had a mismatch at the owner (flipping
  an already-equal bit destroys the match). If every other symbol in that
  alignment was already equal, the flip **repairs** it. One flip repairs at
  most one nearest mismatch per alignment.

Since the two cases partition all matches, the decomposition is complete:

```math
\boxed{\mathrm{route}_{\rm flip}
=\max(\mathrm{route}_{\rm delete},\ \mathrm{route}_{\rm repair}).}
```

The credit follows the same split: add the common *(deletion − baseline)*
term once per owner, then add *(repair − deletion)* wherever a repair wins.
Nothing is recomputed per bit.

**Example, continued.** Flipping `Q₄` deletes the baseline `(3,2)`. The
alignment ending at K endpoint 3 has zero matching symbols right of its
nearest mismatch (`a=0`), zero common suffix left of the owner (`b=0`), and
its XOR is exactly the flipped bit — so it repairs to the candidate
`(a+1+b, r)=(1,3)`. That beats the deletion route, giving the new route
`(1,3)` we saw in §2. Note the same alignment also repairs the K owner at
position 3, but *that* owner's deletion keeps `(3,2)`, so its repair loses:
**sharing a candidate does not mean sharing the credit.**

## 5. Insight 3 — deletion and repair are both compressible

### 5.1 Deletion routes form monotone envelopes

Fix row `t` with baseline `(L,r₀)`, and let `R[h]` be the latest endpoint
supporting match length at least `h`.

- A Q owner `u` caps every surviving suffix at `t−u`, so its deletion route is
  `(min(L,t−u), R[·])` — queries on a fixed staircase, answerable with one
  monotone cursor for all owners in order.
- A K owner `s` splits candidates into endpoints left of `s` (unaffected, a
  nondecreasing envelope `A(s)`) and endpoints at or right of `s` (capped, a
  nonincreasing envelope `H(s)`); the deletion route is `max(A,H)`.

**Example, continued.** For `01010` at row 4 the whole K deletion envelope is
checkable by hand:

| K owner s | A(s) | H(s) | Deletion winner |
|---:|---|---|---|
| 0 | (0, −1) | (2, 2) | (2, 2) |
| 1 | (1, 0) | (1, 2) | (1, 2) |
| 2 | (1, 0) | (0, −1) | (1, 0) |
| 3 | (3, 2) | (0, −1) | (3, 2) |

`A` grows, `H` shrinks, their max has a valley — and runs of equal characters
make `A(s)` follow the closed form `min(c, s−o)` on whole intervals, so the
envelope is stored as a few compressed records instead of `T` answers.

### 5.2 One shared suffix automaton answers all repair searches

A repair candidate needs the *edited* Q suffix to occur somewhere in K — a
substring query. The backward builds one suffix automaton (SAM) of the full K
once, then answers every row's questions against it. The index knows the
future, but every answer is filtered to endpoints `r < t` by a doubly linked
list of active endpoints in terminal order (activation is `O(1)`; only causal
occurrences are ever visited). Identical Q words ask identical static
questions, so root alternatives are prepared once per Q-word *class*, sparsely,
at `O(U_Q+R)` storage instead of a dense `U_Q×D` table.

### 5.3 A uniquely-occurring repair becomes one interval event

If the shortest edited suffix has exactly **one** occurrence in the full
static K at row `t₀` (certified by `first == last`), then any future
occurrence is forced: removing its trailing new symbols must land on that same
endpoint. The whole remaining lifetime is a line,

```math
\mathrm{repair}(t)=(t+\alpha,\ t+\beta),\qquad t_0\le t\le\tau,
```

ending at the first mismatch. Instead of advancing this branch row by row,
the algorithm emits **one event** covering `[t₀,τ]`, books the correction
through a Fenwick tree of snapshots, and groups the K-side winners by their
now-fixed owner in a heap.

**Companion example.** With `Q=00000`, `K=10000`, the edited symbol `1` has
the unique endpoint 0. Flipping Q owner 1 produces routes `(1,0),(2,1),(3,2),
(4,3)` at rows 1–4 — one event, not four row visits. Flipping Q owner 2
produces `(1,0),(2,1),(3,2)` — another event with the same fixed K owner 0.

Two tempting shortcuts are invalid: uniqueness of the *longest* edited match
says nothing about shorter edited suffixes, and uniqueness among *currently
causal* occurrences says nothing about the future. The certificate must use
the short complete edited suffix against the full static K.

### 5.4 Payload scores are shared

Routes that select the same V word at row `t` need the same score `g_tᵀP`. A
frequency-admitted cache (at most `4D` payload classes, so at most `4TD`
entries, allocated on demand) removes the repeated dot products without any
approximation.

## 6. What this guide deliberately skips

The three insights above *are* the algorithm. Everything else is faithful
bookkeeping, documented in the [exhaustive reference](docs/algorithm-reference.md):

- **Floating-point exactness.** Interval bookkeeping reorders cancellations;
  the implementation carries error-free binary64 residuals (TwoSum expansions,
  birth/finish snapshots) until one final correctly-rounded merge
  ([reference §10](docs/algorithm-reference.md#10-a-real-number-identity-is-not-yet-a-safe-floating-point-program),
  [`exact_expansion.h`](conv_rosa_transformer/csrc/exact_expansion.h)).
- **The honest cost model.** The structural worst case remains quadratic up to
  logarithmic factors; the reference counts actual executed work per parameter
  instead of claiming a near-linear theorem
  ([reference §11](docs/algorithm-reference.md#11-put-the-pieces-together-and-count-the-remaining-work)).
- **Exact randomized root search**, the optional sampling estimator, and the
  second-order diagonal/joint-credit layer: see
  [exact-randomized](docs/exact-randomized.md),
  [randomized-backward](docs/randomized-backward.md), and
  [native-training](docs/native-training.md).

## 7. Read the code, then try to break it

| Mathematical role | Source |
|---|---|
| Independent hard forward and all-bit definition | [`rosa_reference.py`](conv_rosa_transformer/rosa_reference.py) |
| Public arrays, validation, native build | [`rosa_core.py`](conv_rosa_transformer/rosa_core.py) |
| Lean hard forward and lazy latest endpoints | [`forward.h`](conv_rosa_transformer/csrc/forward.h) |
| Full-key SAM dictionaries, shared Q roots, causal support, Q branches | [`sam_q.h`](conv_rosa_transformer/csrc/sam_q.h) |
| Shared suffix and endpoint queries | [`sam_routes.h`](conv_rosa_transformer/csrc/sam_routes.h) |
| R staircase, A/H, compressed owner intervals | [`delete_indexed.h`](conv_rosa_transformer/csrc/delete_indexed.h) |
| Row loop, K heaps, score sharing, contraction | [`unified_core.cpp`](conv_rosa_transformer/csrc/unified_core.cpp) |
| Residual-preserving aggregation | [`exact_expansion.h`](conv_rosa_transformer/csrc/exact_expansion.h) |
| Bounded causal-band joint proposals | [`pair_candidates.h`](conv_rosa_transformer/csrc/pair_candidates.h) |
| Batched training, metric, blocks and logit mapping | [`training_impl.h`](conv_rosa_transformer/csrc/training_impl.h) |
| Framework-independent C interface | [`training.h`](conv_rosa_transformer/csrc/training.h) |
| Thin PyTorch autograd adapter | [`rosa_layer.py`](conv_rosa_transformer/rosa_layer.py) |

Run `python tests/run.py` for quick independent checks, or
`python tests/run.py --exhaustive` for full binary enumeration and structural
oracles. Useful exercises before modifying a data structure:

1. In the running example, compute every Q deletion route at row 4. Explain
   why changing a winning match *length* need not change its payload or credit.
2. Prove the A/H crossing condition with the endpoint tie-break included.
   What goes wrong if `>=` is changed at the wrong boundary?
3. Construct a short edited pattern with two static occurrences, only one of
   which is currently causal. Explain why a temporary uniqueness test fails.
4. For the companion example of §5.3, derive the fixed K owner for every Q
   flip and count the interval-lifetime work directly. Which work is eventized
   and which is still a sum?
