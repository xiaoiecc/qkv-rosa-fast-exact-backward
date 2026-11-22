# Training Guide: Using Hard QKV-ROSA as an Ordinary Layer

[简体中文](training.zh-CN.md) · [Home](../README.md) · [Algorithm](algorithm.md) · [C/C++ API](c-api.md)

`NBitQKVRosa` wraps one batched C++17 engine. Packing, exact hard routing, counterfactual credit, local trust penalties, joint two-bit replays, and logit gradient mapping all execute in a single native call per forward/backward. PyTorch handles only the dense projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`), tensor layout conversion, and standard autograd.

Your training loop needs no custom optimizer or special hooks:

```python
import torch
from qkv_rosa import NBitQKVRosa

layer = NBitQKVRosa(channels=128, n_bits=8)
optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)

optimizer.zero_grad(set_to_none=True)
loss = (layer(x) - target).square().mean()
loss.backward()
optimizer.step()
```

---

## 1. Three Independent Options

| Option | Default | Meaning |
|---|---|---|
| `credit_order` | `"second"` | `"first"` for pure linear counterfactual credit; `"second"` adds a local diagonal trust penalty on Q/K plus bounded joint two-bit block corrections. |
| `backward_mode` | `"exact"` | `"exact"` evaluates every single-bit credit term; `"approximate"` opts into stratified residual-row or numerical-channel estimation. |
| `randomized` | `True` | Enables a verified full-key query cache in the suffix index. Never changes routes, joint proposals, or numerical outputs. |

```python
# First-order exact credit
layer_1st = NBitQKVRosa(128, 8, credit_order="first")

# Approximate backward: stratified residual rows (all numeric channels kept)
layer_rows = NBitQKVRosa(
    128, 8,
    credit_order="second",
    backward_mode="approximate",
    approximation_rows=32,
    approximation_channels=0,
)

# Approximate backward: fixed-size PPS channel sampling (ignores approximation_rows)
layer_ch = NBitQKVRosa(
    128, 64,
    credit_order="second",
    backward_mode="approximate",
    approximation_channels=8,
)
```

---

## 2. First-Order Credit and Logit Mapping

When flipping bit $i$ shifts the hard output by $\delta = Y^{\text{flip}(i)} - Y$, the first-order credit is:

$$
C_i^{(1)} = \sum_t g_t^\top \delta_t
$$

Let $b = \mathbf{1}[z > 0]$ be the current hard bit. The oriented credit (bit-1 minus bit-0) is $\Delta_i = (1 - 2b)\,C_i$. Two logit mappings are supported via `credit_to_logit`:
- **`"bernoulli"` (default):** Multiplies $\Delta_i$ by the Bernoulli sigmoid derivative $\frac{p(1-p)}{\tau}$, where $p = \sigma(z/\tau)$ (default `tau=1.0`).
- **`"identity"`:** Passes $\Delta_i$ through directly.

Value ($V$) bit credits and embedding ($e_0, e_1$) gradients accumulate directly along the baseline routes in $O(TD)$ time.

---

## 3. Second-Order Mode: Diagonal Trust & Joint Bits

### 3.1 Local diagonal trust model

In standard backpropagation, the layer receives only one upstream gradient $g_t$, which cannot uniquely recover an arbitrary downstream Hessian. `"second"` therefore specifies an explicit **output-diameter diagonal trust model** for Q/K:

$$
C^{(2)}(\delta) = g^\top \delta + \frac{1}{2}\sum_{t, j} h_{t,j}\,\delta_{t,j}^2
$$

With output-set diameter $\Delta_Y = \max\big(2\|e_1\|_2,\, \||e_0| + |e_1|\|_2\big)$ and default `trust_diagonal=0.5`, each row uses:

$$
h_{t,j} = \frac{\|g_t\|_2}{\Delta_Y}
$$

Beneficial steps along $-g_t$ up to diameter $\Delta_Y$ remain clearly negative in cost, while large unproductive route jumps are damped. This penalty applies to Q/K route changes only; $V$ and embeddings stay linear.

### 3.2 Joint two-bit blocks

Sometimes flipping bit $i$ or bit $j$ alone has no effect, while flipping both together unlocks a strong match. Given local corner costs $C_i, C_j, C_{ij}$ and interaction $\kappa = C_{ij} - C_i - C_j$, with local flip probabilities $\alpha_i, \alpha_j$:

$$
F = \alpha_i C_i + \alpha_j C_j + \alpha_i \alpha_j \kappa,
\qquad
\frac{\partial F}{\partial z_i} = \alpha_i'(C_i + \alpha_j \kappa)
$$

[`pair_candidates.h`](../qkv_rosa/csrc/pair_candidates.h) inspects at most `joint_rows` active rows (default 32) and proposes up to `joint_pairs` disjoint pairs per stream (default 1). Each chosen pair is evaluated inside the component by replaying the lightweight causal forward matcher — the surrounding model and loss still run only once.

---

## 4. Explicit Approximation (`backward_mode="approximate"`)

Two mutually exclusive estimators are selected by `approximation_channels`:

1. **Stratified residual-row estimator (`approximation_channels=0`):**
   - Computes full deletion plus the one-symbol repair control ($B = D + R_1$) exactly.
   - Eliminates rows with provably zero long-match residual using exact zero certificates.
   - Stratifies remaining rows in time, samples one complete row per stratum, and applies inverse-probability weights **after** exact route maximization (never sampling candidates before taking $\max$).
2. **Centered-channel PPS estimator (`approximation_channels > 0`):**
   - Draws a fixed-size probability-proportional-to-size sample of centered numeric feature channels via pivotal rounding (keeping the empty-route indicator complete), then runs the full exact routing engine on the sampled features.

Both branches keep hard forward, $V$/embedding gradients, and selected joint corners completely evaluated. Estimators are conditionally unbiased in real arithmetic, but do not guarantee uniform relative error bounds or monotonic loss decrease.

---

## 5. Practical Notes

- **Combining multiple losses:** Because second-order trust weights and active proposal rows depend on `grad_y`, calling `.backward()` separately on two losses is not equivalent to calling `.backward()` once on their sum. Combine loss terms before backpropagating.
- **No double autograd:** Hard credit is `@once_differentiable`; `create_graph=True` is unsupported.
- **AMP / GradScaler:** Non-finite incoming gradients propagate through the adapter so standard `GradScaler` overflow detection works as expected.
- **Checkpointing:** `state_dict()` stores linear projections and `emb0`/`emb1`. Save your layer configuration (`n_bits`, `credit_order`, `tau`, etc.) alongside weights.
- **Shape contract:** Input shape is `[B, T, C]` with $C > 0$, $1 \le \text{n\_bits} \le 64$, and $C \bmod \text{n\_bits} = 0$. Every contiguous block of `n_bits` channels forms an independent stream (`groups = C // n_bits`).
