# One forward, one backward: use ROSA as an ordinary component

[中文](native-training.zh-CN.md) · [Algorithm](../ALGORITHM.md) · [C API](native-api.md) · [Current results](subquadratic-results.md)

The layer has one C++17 credit engine. C++ owns packing, hard routing, finite credits, the local trust model, selected joint corners, and logit gradients. PyTorch owns dense projections, layout/device conversion, and attachment to the usual autograd graph. A batch uses one native forward call and one native backward call.

## Three choices within the same engine

| Option | Default | Meaning |
|---|---|---|
| `credit_order` | `"second"` | `"first"`: original linear finite credit; `"second"`: diagonal trust plus selected joint-bit cooperation |
| `randomized` | `True` | Bounded full-query-key caching; answers and arithmetic order unchanged |
| `backward_mode` | `"exact"` | Complete evaluation; `"approximate"` explicitly permits residual-row or channel estimation |

```python
from conv_rosa_transformer.rosa_layer import NBitQKVRosa

first = NBitQKVRosa(128, 8, credit_order="first")
second = NBitQKVRosa(128, 8)  # diagonal trust + one joint pair by default
approximate = NBitQKVRosa(
    128, 8, credit_order="second", backward_mode="approximate",
    approximation_rows=32, approximation_channels=0,
)
channels = NBitQKVRosa(
    128, 64, credit_order="second", backward_mode="approximate",
    approximation_channels=8,
)
```

A zero channel budget selects the residual-row estimator with all numerical channels. A positive channel budget selects a weighted sample of centered numerical channels and runs the complete structural calculation on those features; `approximation_rows` is unused in that branch. Both choices keep the hard forward complete. The [approximation tutorial](randomized-backward.md) derives the targets and error contracts.

The training loop is ordinary:

```python
optimizer.zero_grad(set_to_none=True)
prediction = model(x)        # model contains ROSA
loss = criterion(prediction, target)
loss.backward()
optimizer.step()
```

Joint corners are internal component work. The surrounding model, downstream layers, and task loss run once. First order has no joint replay. With the default one selected pair, complete mode needs at most one internal edited-route replay; approximate mode can need two additional singleton replays so that the selected block uses complete local corner costs.

## First-order credit and the logit mapping

For one bit flip with full hard output change $\delta$, the target is $C^{(1)}=g^\top\delta$, where $g=\partial L/\partial Y$. This is a counterfactual finite output difference. It is not a derivative of threshold quantization or a rerun of an arbitrary nonlinear loss.

If the current hard bit is $x$, its oriented bit-1-minus-bit-0 credit is $(1-2x)C$. The default `credit_to_logit="bernoulli"` multiplies this by $p(1-p)/\tau$, with $p=\sigma(z/\tau)$. `credit_to_logit="identity"` keeps the oriented credit directly. The same mapping is applied to all Q/K/V bits.

V cannot change matching routes. Its credits and the ordinary gradients of the empty embedding and signed-payload scale are accumulated along the original routes in $O(TD)$ work. Q/K use the shared deletion-and-repair construction described in [the algorithm](../ALGORITHM.md).

## Why the second-order option needs an explicit model

One normal backward supplies $g$. The losses $L_a(Y)=g_0^\top(Y-Y_0)+\tfrac12a\|Y-Y_0\|^2$ all have the same gradient at $Y_0$, while their curvatures differ. No component can infer that arbitrary curvature from this one upstream vector alone.

The option therefore chooses a local diagonal model:

```math
C^{(2)}(\delta)=g^\top\delta+\tfrac12\sum_{t,j}h_{tj}\delta_{tj}^2.
```

Let $\Delta_Y=\max(2\|e_1\|_2,\||e_0|+|e_1|\|_2)$ be the output-set diameter. Ignoring the numerical zero-amplitude floor, the default `trust_diagonal=0.5` gives $h_{tj}=\|g_t\|_2/\Delta_Y$, shared across a row. General `trust_diagonal=\lambda` multiplies this by $2\lambda$.

For a displacement of length $a\le\Delta_Y$ aligned with the negative gradient, the default model cost is $-a\|g_t\|_2+a^2\|g_t\|_2/(2\Delta_Y)\le-a\|g_t\|_2/2<0$. This explains the scale: useful finite jumps remain useful in the chosen model while large route changes receive a penalty. It is not a guarantee about the true task loss. Q/K routing edits use this penalty; V retains its original linear credit.

The structural engine does not recompute every quadratic displacement. Every nonempty payload is a signed $e_1$ vertex, whose componentwise square is fixed. Centering the payload features and adding one empty-payload indicator turns the score into linear projections with at most one extra numerical channel. This preserves the subquadratic structure and reduces cancellation during feature construction.

## Joint bits can cross a barrier that singleton credits miss

Two singleton edits can have no useful signal while their joint edit helps. For complete local corner costs $C_i,C_j,C_{ij}$, define $\kappa=C_{ij}-C_i-C_j$. If $\alpha_i,\alpha_j$ are local flip probabilities, the two-bit objective is

```math
F=\alpha_iC_i+\alpha_jC_j+\alpha_i\alpha_j\kappa,
\qquad \partial_{z_i}F=\alpha_i'(C_i+\alpha_j\kappa).
```

The proposer examines a bounded set of rows and retains at most `joint_pairs` disjoint blocks. The default is one pair per stream and at most 32 proposal rows. The algorithm does not enumerate every bit pair or solve interactions among all chosen blocks. A fixed budget $J$ adds $\widetilde O(JT(D+1))$ edited-index replay work; an increasing budget must be included in the complexity accounting.

Singleton corners are already available in complete mode. When numerical channels or residuals were sampled, the chosen singleton corners are recomputed completely before forming the joint correction. The joint corner itself always uses a complete internal indexed route evaluation. Selected coordinate gradients are replaced before they flow into the shared projection parameters.

## Randomness, floating-point arithmetic, and loss composition

The exact cache checks every full query key; collisions trigger recomputation. For fixed inputs, local rule, and other seeds, turning it on or off preserves this engine's gradient outputs bitwise. It need not make every input faster. Proposal randomness is separate: disabling the cache does not disable the second-order proposer.

Complete mode includes all mathematical credit terms, but binary64 grouping and FFT can round differently from a historical direct scalar reduction. Approximate mode additionally has sampling error. Neither complete algebraic equality nor an unbiased conditional estimator implies identical old float32 training trajectories. See [the reference contract](algorithm-reference.md) and [current measurements](subquadratic-results.md).

The trust coefficients and active-row proposals depend on $g$. Consequently, two separate loss backwards followed by accumulation need not equal one backward on their combined loss. Combine losses first when that is the intended update. Positive rescaling of a fixed loss is supported; hard-credit double autograd is explicitly unsupported.

Forward caches freeze hard codes, logits, embeddings, options, and seeds. The exact-cache toggle preserves framework RNG consumption. Approximate sampling and joint proposals use saved local seeds; evaluation does not draw training proposal randomness. Use the framework's ordinary RNG preservation when checkpointing. The adapter handles nonfinite incoming gradients so a normal GradScaler can detect overflow and skip an update.

## Standalone and numerical integration

The same implementation is available through [training.h](../conv_rosa_transformer/csrc/training.h) and [the NumPy adapter](../conv_rosa_transformer/native_training.py). Initialize options through the defaults function. In C, selecting first order means setting all three related fields consistently:

```cpp
RosaTrainingOptions options;
rosa_training_defaults(&options);
options.credit_order = 1;
options.joint_pairs = 0;
options.trust_diagonal = 0.0;
```

Python derives those defaults from `credit_order="first"`. The C complete and approximate entries call the same engine; the separate approximation options hold work budgets, not an algorithm selector. Array layouts, pointer ownership, return codes, and diagnostic counters are documented in [native-api.md](native-api.md).

```bash
c++ -O3 -std=c++17 -ffp-contract=off examples/native_training.cpp \
    conv_rosa_transformer/csrc/unified_core.cpp -o /tmp/rosa-native
/tmp/rosa-native
/tmp/rosa-native --first --deterministic
/tmp/rosa-native --approximate
```

This is CPU C++17 code. CUDA tensors incur transfers and synchronization. The dense projection parameters and normal model state remain ordinary PyTorch parameters; save the training configuration with a checkpoint. Choose the credit rule and approximation budget by task loss per unit time, validation behavior, numerical error, and memory, not by the complexity exponent alone.

## Migrating existing callers

Remove `credit_engine` and `repair_samples` from Python constructors and NumPy calls; they are no longer accepted. Use `credit_order` for first/second credit. For sampling, set `backward_mode="approximate"` and the row or channel budget. The C options retain their struct layout, but `repair_budget` is reserved and must be zero.
