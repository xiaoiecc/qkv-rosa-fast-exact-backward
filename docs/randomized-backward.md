# Approximate backward: keep a useful exact control, estimate the remainder

[中文](randomized-backward.zh-CN.md) · [Complete algorithm](../ALGORITHM.md) · [Exact cache](exact-randomized.md) · [Training API](native-training.md) · [Current results](subquadratic-results.md)

The complete subquadratic engine has exact routes and includes every mathematical credit term. Its verified cache only saves repeated queries. `backward_mode="approximate"` makes a separate, explicit tradeoff: compute fewer numerical contributions while targeting the same chosen local credit. It does not change the hard forward or sample a few candidates before choosing a route.

## 1. Two estimators, selected by the numerical-channel budget

| Settings in approximate mode | Computation |
|---|---|
| `approximation_channels=0` | Complete shared deletion and one-symbol repair control; stratified samples of complete remaining row corrections |
| Positive `approximation_channels` | Complete structural calculation on a fixed-size weighted sample of centered numerical channels; exact empty control |

The default row budget is 32 per batch/group stream. A zero channel budget means **all** numerical channels in the row estimator, not zero computation and not complete mode. With a positive channel budget, `approximation_rows` is unused: the two estimators are alternatives, not two automatically stacked sampling stages. Both work with first- and second-order credit.

## 2. Fix the credit before choosing a sample

Let $Y_t$ be the original hard output and hold the current upstream vector and local diagonal fixed:

```math
\phi_t(y)=g_t^\top(y-Y_t)+\frac12\sum_j h_{t,j}(y_j-Y_{t,j})^2.
```

First order uses $h=0$. For Q/K bit coordinate $i$, $C_i=\sum_t\phi_t(Y_t^{\operatorname{flip}(i)})$. The diagonal is the training model described in [native-training.md](native-training.md), not an estimate of an arbitrary downstream Hessian from one gradient.

A route is selected by longest match, then latest endpoint. If two candidates have lengths 8 and 2, sampling one and taking its maximum does not preserve the maximum of the complete set; the resulting payload can be entirely different. Inverse-probability weighting cannot generally repair that nonlinear selection. We sample complete row contributions **after** the exact candidate maximum is settled, or linear numerical channels after the structural routes are settled.

## 3. Full deletion plus one-symbol repair is a useful control

Forbid matches from crossing the owner of $i$, without shifting symbols. Denote that route by deletion. Define

```math
D_{t,i}=\phi_t(Y_t^{\operatorname{delete}(i)}),\qquad
R_{t,i}=\phi_t(Y_t^{\operatorname{flip}(i)})-D_{t,i}.
```

Then $C_i=\sum_t(D_{t,i}+R_{t,i})$. All bits of an owner share deletion, so the complete AP deletion kernel is an economical part to retain.

A one-symbol router chooses the latest causal occurrence of the current Q symbol, ignoring earlier context. Let $R_{1,t}$ be its repair-minus-deletion contribution, scored around the **actual full baseline** $Y_t$. Word classes and their occurrence lists compute these one-symbol effects cheaply. Use

```math
B=D_{\mathrm{full}}+\sum_t R_{1,t},\qquad
\rho_t=R_t-R_{1,t},\qquad C=B+\sum_t\rho_t.
```

The row estimator keeps $B$ complete and samples only $\rho_t$. The center matters for second order: substituting the one-symbol router's own baseline in its quadratic score would create a different cross term and break this identity. Implementation: [approx_onegram.h](../conv_rosa_transformer/csrc/approx_onegram.h) and [engine.h](../conv_rosa_transformer/csrc/subquadratic/engine.h).

## 4. Prove some residuals zero before sampling

Two inexpensive certificates shrink the population without adding estimation error.

**No near-equal adjacent pair.** Any match of length at least two after at most one bit edit must end in a causal two-symbol pair satisfying

```math
\operatorname{popcount}(Q_{t-1}\oplus K_{r-1})+
\operatorname{popcount}(Q_t\oplus K_r)\le1.
```

If no such pair exists, every relevant match has length at most one. Full and one-symbol routing agree, so $\rho_t=0$. Finding a near pair merely leaves the row uncertified; it does not prove a nonzero residual. Certificate lookups verify complete keys.

**No causal repair birth.** Mark $b_u=1$ when some $s<u$ has $Q_u\oplus K_s$ equal to one bit. A repair contributing at row $t$ has an unchanged right tail of length $a\le L_t$, so its Q center lies in $t-L_t\le u\le t$. If this window contains no marked birth, neither full nor one-symbol repair exists and again $\rho_t=0$. The one-symbol class plan provides these marks and their prefix sums.

Rows with zero upstream support also contribute zero under this native trust rule. The remaining rows form the sampled population. Complete mode uses the same certificates: when its counted endpoint work fits the complete-engine budget, it evaluates every residual through this exact row formula. With an empty certified population, the control is already the complete answer, including when an approximation option was requested. See [approx_row_certificate.h](../conv_rosa_transformer/csrc/approx_row_certificate.h).

## 5. Stratify in time and keep the probability in the denominator

Put the remaining $N$ rows in time order and split them into $m$ contiguous strata of nearly equal size. Select one row from each stratum $S$. With the default importance setting,

```math
p_t=\frac{0.1}{|S|}+0.9\frac{\|g_t\|_2}{\sum_{u\in S}\|g_u\|_2},
\qquad
\widehat C=B+\sum_S\frac{\rho_{T_S}}{p_{T_S}}.
```

A zero norm sum uses uniform probabilities; `importance=0` uses uniform sampling throughout. The 10% floor keeps support for every residual and bounds a stratum's inverse weight by $10|S|$.

Condition on the hard inputs, embeddings, incoming gradient, local metric, certificates, and joint proposal. In real arithmetic,

```math
\mathbb E[\rho_{T_S}/p_{T_S}]=\sum_{t\in S}\rho_t,
\qquad \mathbb E[\widehat C]=C.
```

Its total squared-error expectation is

```math
\mathbb E\|\widehat C-C\|_2^2
=\sum_S\left(\sum_{t\in S}\frac{\|\rho_t\|_2^2}{p_t}
-\left\|\sum_{t\in S}\rho_t\right\|_2^2\right).
```

This shows why the control and certificates help: they reduce or remove residuals before spending a sample. Upstream norm is only an available proxy for residual size, not a claim that the two are equal. If the budget covers the remaining population, or the sampled-row budget would exceed the implementation's complete-engine work threshold, the engine evaluates the complete credit instead.

A selected row enumerates all causal endpoints in its equal-word and one-bit-neighbor classes, reuses the global exact suffix index, computes its route maxima, and applies its sampling weight only to the resulting complete correction. It does not scan or sample arbitrary losing candidates. Code: [approx_sampling.h](../conv_rosa_transformer/csrc/approx_sampling.h) and [approx_indexed_row.h](../conv_rosa_transformer/csrc/approx_indexed_row.h).

## 6. Explicit channel sampling reduces numerical dimension

A large numerical dimension can dominate interval contraction even when the route list is compact. The positive channel budget targets that cost while retaining the complete structural construction.

For a reference signed payload $P_\ast$, use centered features $X_{s,j}=P_{s,j}-P_{\ast,j}$ and row weights $a_{t,j}=g_{t,j}-h_{t,j}Y_{t,j}$. An additional empty-payload indicator represents the remaining constant difference. This is the same exact lift used by complete second-order calculation. Channels with an identically zero coefficient or reachable payload span can be removed exactly. The empty indicator stays complete.

For the remaining $N$ channels, the implementation uses a cheap magnitude proxy $w_j=(\max_t|a_{t,j}|)(\max_{s\ge1}|X_{s,j}|)$. With requested budget $m<N$, water filling chooses

```math
\pi_j=\min\left(1,\frac{0.1m}{N}+\lambda w_j\right),
\qquad \sum_j\pi_j=m.
```

Pivotal rounding selects exactly $m$ channels while preserving these marginals. A random permutation removes dependence on fixed column adjacency. A selected channel is weighted by its **original** inclusion probability, $1/\pi_j$; certainty channels have weight one. For complete per-channel credit vectors $Z_j$,

```math
\widehat Z=\sum_{j\text{ selected}}Z_j/\pi_j,
\qquad \mathbb E[\widehat Z]=\sum_jZ_j.
```

The same selected feature set is used throughout that stream's Q/K deletion and repair contractions. Routes do not depend on this selection, and no bit-coordinate output is dropped merely because its corresponding numerical channel was not sampled. If the budget covers all eligible channels, this branch is complete. Code: [Features in engine.h](../conv_rosa_transformer/csrc/subquadratic/engine.h) and [approx_channel_pivotal.h](../conv_rosa_transformer/csrc/approx_channel_pivotal.h).

An estimate may have strongly correlated errors across coordinates. Conditional unbiasedness alone is not a relative-error or convergence bound, especially when the exact credit is small because of cancellation. Avoid postselecting exact replacements based on the sampled result being small: even replacing only the zero outcomes of an unbiased estimate that takes 0 or 2 equally changes its mean from 1 to 1.5.

## 7. What stays complete, and what error remains

The hard forward remains complete. V and embedding gradients are evaluated along the original routes. Selected joint blocks use complete local corner costs; approximate singleton corners are restored by internal forward replays when needed. The outer network and loss are not replayed.

Numerical batching and FFT are already part of the complete subquadratic engine. Their rounding error is distinct from row/channel sampling error; FFT is not restricted to approximate mode. Full mathematical credit evaluation does not promise a historical binary64 reduction order, and a fixed-budget unbiased estimator does not promise an unchanged Adam update or training trajectory.

Measure Q and K errors separately. Also measure mapped logit gradients, shared-parameter/input gradients, training loss, validation loss, full step time, and memory. A large correct component can hide an inaccurate small component in a combined relative norm. Work budgets are not accuracy tolerances. Current observations, including slowdowns and difficult cancellation cases, belong in [the results](subquadratic-results.md).

## 8. Ordinary model usage

```python
from conv_rosa_transformer.rosa_layer import NBitQKVRosa

full = NBitQKVRosa(128, 8, credit_order="second")
rows = NBitQKVRosa(
    128, 8, credit_order="second", backward_mode="approximate",
    approximation_rows=32, approximation_channels=0,
)
channels = NBitQKVRosa(
    128, 64, credit_order="first", backward_mode="approximate",
    approximation_channels=8,
)
```

All three use the same normal `loss.backward()` integration and one structural engine. The adapter saves sampling seeds with the forward state, separate from the joint proposal. `randomized=False` disables exact query caching only; an explicitly selected approximate estimator still samples. The [training guide](native-training.md) and [C API](native-api.md) cover the configuration and saved-state contract.
