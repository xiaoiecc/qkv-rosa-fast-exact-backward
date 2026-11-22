# Algorithm & Complexity Guide

[简体中文](algorithm.zh-CN.md) · [Home](../README.md) · [Training Guide](training.md) · [C/C++ API](c-api.md)

This guide explains what hard QKV-ROSA backpropagation actually computes, why the literal definition takes $O(b\,T^4)$ time, and how the engine's **single causal interval sweep** computes the exact same credits in **$\widetilde{O}(T)$ expected time** on independent uniform inputs ($\widetilde{O}(T^2)$ worst-case at fixed width and joint budget).

---

## 1. Problem Definition

Consider a stream of length $T$ with packed $b$-bit unsigned query/key symbols $Q_t, K_t \in [0, 2^b)$ ($1 \le b \le 64$) and binary value vectors $V_t \in \{0, 1\}^b$.

### 1.1 Hard causal forward

At each step $t$, we search strictly earlier key positions $r < t$ for the longest common suffix $F_t(r) = \operatorname{LCSuffix}(Q[0..t], K[0..r])$:

$$
(L_t, e_t) = \max_{0 \le r < t} (F_t(r), r)
$$

Maximization is lexicographic: **longest match first, latest endpoint $r$ on ties**. Unmatched rows take the empty route $(0, -1)$.

The output is the payload immediately **following** the winning endpoint:
- If $e_t \ge 0$: $Y_t = P_{e_t + 1} = (2 V_{e_t + 1} - 1) \odot e_1$ (note $e_t < t \implies e_t + 1 \le t$, preserving strict causality).
- If $e_t = -1$: $Y_t = e_0$ (a learned empty embedding).

The forward matcher ([`forward.h`](../qkv_rosa/csrc/forward.h), [`causal_windows.h`](../qkv_rosa/csrc/causal_windows.h)) uses radix suffix windows plus a causal suffix automaton with bounded-probe hash tables and ordered overflow, guaranteeing $O(T \log T)$ worst-case time and $O(T)$ space.

### 1.2 Exact one-bit counterfactual credit

Given an upstream gradient $g_t = \partial L / \partial Y_t$, flip a single bit $i$ in $Q$, $K$, or $V$ and re-run the exact hard forward to get $Y^{\text{flip}(i)}$. The **first-order counterfactual credit** is:

$$
C_i = \sum_{t=0}^{T-1} g_t^\top \big(Y_t^{\text{flip}(i)} - Y_t\big)
$$

In second-order trust mode, the row score adds a diagonal quadratic penalty $f_t(y) = g_t^\top(y - Y_t) + \frac{1}{2}\sum_j h_{t,j}(y_j - Y_{t,j})^2$. Because every non-empty payload has identical per-coordinate square $e_{1,j}^2$, centering payloads around the first reachable payload $P_1$ and adding one empty-route indicator channel reduces the quadratic score to the exact same linear projection interface (`Features` in [`engine.h`](../qkv_rosa/csrc/engine/engine.h)).

- **Values ($V$):** Flipping a $V$ bit never alters routing. One $O(Tb)$ scatter along the baseline routes computes all $V$ credits as well as $e_0, e_1$ gradients.
- **Queries & Keys ($Q, K$):** A single bit flip can break an existing match or repair a one-bit mismatch into a longer match. Evaluating this naively costs $O(b\,T^4)$.

---

## 2. Step 1: Contracting Inactive Time Coordinates

Draw a causal edge from query time $u$ to key time $s < u$ whenever $Q_u$ and $K_s$ are **equal or differ in exactly one editable bit**. Every symbol pair aligned in any exact match or singly repaired match must be such an edge.

Therefore, any time step with neither an incoming nor an outgoing edge can **never** participate on either side of any exact or single-edit match:

1. [`edit_alphabet.h`](../qkv_rosa/csrc/engine/edit_alphabet.h) finds live coordinates in strictly $O(T)$ time using two complementary bit-mask projections (every pair within Hamming distance 1 agrees on at least one half). Light buckets ($\le 8$ candidates) verify full 64-bit words; heavy buckets conservatively retain all earlier keys up to the requesting time.
2. Each contiguous gap of inert positions is **contracted into a single separator** while preserving chronological order. Each retained endpoint $r'$ stores the payload that originally followed $r$ (even if $r+1$ was removed).
3. Valid matches cannot cross a separator, so match lengths, edit owners, and tied-endpoint orderings are preserved identically.

For independent uniform $b$-bit words, the expected number of causal edges is:

$$
\mathbb{E}[E] = \binom{T}{2}\frac{b+1}{2^b}
$$

At $T = 8192, b = 32$, $\mathbb{E}[E] \approx 0.258$ — almost the entire stream collapses before the main reducer even runs. (Joint two-bit edits still replay on the original uncontracted stream, since two edits can bridge a separator that one edit cannot.)

---

## 3. Step 2: Three-Term Telescoping Decomposition

For a row $t$ with baseline route $B = (L, e)$ and a fixed physical bit edit in $Q$ or $K$, define three exact lexicographic route maxima:
- **$S$ (Survivor / Deletion):** Best original exact suffix that avoids crossing the edited position.
- **$U$ (One-symbol repair):** Better of $S$ and all length-1 repairs.
- **$W$ (Complete winner):** Better of $U$ and all repairs of length $\ge 2$.

The row's credit telescopes cleanly into three disjoint increments ([`interval_credit.h`](../qkv_rosa/csrc/engine/interval_credit.h)):

$$
f_t(W) - f_t(B) = \underbrace{\big[f_t(S) - f_t(B)\big]}_{\text{1. Deletion}} + \underbrace{\big[f_t(U) - f_t(S)\big]}_{\text{2. One-symbol repair}} + \underbrace{\big[f_t(W) - f_t(U)\big]}_{\text{3. Longer repair}}
$$

### 3.1 Surviving matches (Deletion)
Blocking a position affects all $b$ bits at that position identically, so deletion credit is computed once per position and broadcast across bits:
- **Single-run fast path:** If the winning suffix is one repeated symbol whose causal occurrences form a single contiguous run $[A, e]$, deleting $s$ splits it into a right piece (endpoint $e$) and a left piece (endpoint $s - 1$, returning $V_s$). Difference arrays accumulate the left-winning interval in $O(1)$ per row, giving $O(Tb)$ total time on constant streams.
- **Short rows ($L \le \text{local\_limit}$):** Resolved directly from the rolling causal suffix postings maintained during the sweep.
- **Long rows ($L > \text{local\_limit}$):** Batched and delegated to the arithmetic-progression (AP) deletion primitive in [`delete.h`](../qkv_rosa/csrc/engine/delete.h).

### 3.2 One-symbol repairs
- A flip at the current query word $Q_t$ simply checks the latest causal occurrence of the flipped word in $K$.
- A flip at $K_s$ can beat the survivor at row $t$ **if and only if** $L_t \le 1$ and $s$ is strictly later than the existing exact endpoint (when $L_t \ge 2$, damaging an interior position leaves a non-empty suffix at $e$ that wins any length-1 tie). A single reverse pass accumulates row weights by query symbol class until the next exact key occurrence resets that class.

### 3.3 Longer repairs (length $\ge 2$)
Any repaired alignment ending at $(t, r)$ with exact common suffix length $a$ has its unique mismatch at $(Q_{t-a}, K_{r-a})$ and total length $a + 1 + \operatorname{LCSuffix}(t - a - 1, r - a - 1)$. The same candidate updates both the $Q$ edit and the $K$ edit.

During the forward sweep, `Level` maintains exact causal posting lists for packed suffix words up to depth $D$:
- At row $t$, pick pivot depth $p = \min(t, \max(2, \min(D, L + 1)))$.
- Querying exact and one-edited pivot words at depth $p$ visits every repaired alignment of length $\ge p$.
- For shorter $Q$ repairs ($< p$), existence is monotone in length, so a single exponential + binary search finds the longest/latest occurrence without scanning intermediate levels.
- For shorter $K$ repairs ($< p$), a repair can win only by tying at length $L$ at a later endpoint, or by touching the original winning interval $[e - L + 1, e]$ (endpoints within $[e - L + 1, e + p - 1)$). If two disjoint exact pivot witnesses exist at depth $p$, every $K$ edit is certified to retain length $\ge p$ and the local damaged-window scan is skipped completely.

---

## 4. Complexity Summary

At fixed bit width $b$, numerical dimension $d$, and joint-pair budget $J$:

| Input regime | Time bound | Notes |
|---|---|---|
| **Independent uniform $K$, independent $Q$** | **$\widetilde{O}(T)$ expected** | Arbitrary $V$ and upstream gradients $g$ |
| **Unrestricted worst case** | **$\widetilde{O}(T^2)$** | Output-sensitive in retained length, postings, and damaged windows |
| **Identical constant streams** | **$O(Tb)$** | Single-run difference-array path |

**Why expected $\widetilde{O}(T)$ holds for uniform keys:**
Let $\sigma = 2^b$ and $D_0 = \lceil \log_\sigma T \rceil$. When $p \ge D_0$, expected radius-1 pivot hits per row are at most $T(1 + bD_0)\sigma^{-D_0} = O(b \log T)$. When $p < D_0$, the row has no exact match of length $p$; partitioning earlier keys into disjoint length-$p$ blocks yields an exponential miss factor $\exp[-(\lfloor t/p \rfloor - 2)_+\sigma^{-p}]$, bounding expected work per depth by $O(p(1 + bp))$. By a union bound, any baseline match exceeding $\lceil 3 \log_2 T \rceil$ occurs with probability at most $1/T$, so long-row deletion contributes only $\widetilde{O}(T)$ in expectation even under its worst-case bound.

---

## 5. Three Levels of Exactness

1. **Exact routing:** All symbol comparisons check full 64-bit words (including bit 63 and `UINT64_MAX`), causal cutoffs, and longest/latest ties. Verified lookup caches (`randomized=True`) compare full keys and never change routes or outputs.
2. **Algebraic completeness:** In `backward_mode="exact"`, every mathematical term of the counterfactual credit formula is included without sampling.
3. **Floating-point rounding:** Accumulations use IEEE `float64` compiled with `-ffp-contract=off`. Because interval grouping reassociates additions, outputs match naive elementwise summation up to standard binary64 rounding (observed relative L2 $< 10^{-14}$).

---

## 6. Source Map

| File | Role |
|---|---|
| [`qkv_rosa/csrc/forward.h`](../qkv_rosa/csrc/forward.h) | Exact causal suffix-automaton forward matcher |
| [`qkv_rosa/csrc/causal_windows.h`](../qkv_rosa/csrc/causal_windows.h) | Rolling-window forward routes & joint-corner replay |
| [`qkv_rosa/csrc/engine/edit_alphabet.h`](../qkv_rosa/csrc/engine/edit_alphabet.h) | Live coordinate discovery, contraction, and 1-bit edit graph |
| [`qkv_rosa/csrc/engine/interval_credit.h`](../qkv_rosa/csrc/engine/interval_credit.h) | Core single causal interval-sweep reducer |
| [`qkv_rosa/csrc/engine/index.h`](../qkv_rosa/csrc/engine/index.h) | Lazy suffix array, LCP RMQ, wavelet matrix, verified query cache |
| [`qkv_rosa/csrc/engine/delete.h`](../qkv_rosa/csrc/engine/delete.h) | Arithmetic-progression deletion primitive for long rows |
| [`qkv_rosa/csrc/engine/engine.h`](../qkv_rosa/csrc/engine/engine.h) | Coordinate contraction driver, centered features, approximate estimators |
| [`tests/cpp/interval_credit.cpp`](../tests/cpp/interval_credit.cpp) | Independent physical bit-flip C++ oracle |
