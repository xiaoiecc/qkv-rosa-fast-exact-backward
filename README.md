# Fast Exact Backpropagation for QKV-ROSA

A single-file PyTorch reference implementation of an **exact** backpropagation
algorithm for the **QKV-ROSA** discrete retrieval architecture — not an STE, not
a soft relaxation, but the exact single-bit counterfactual VJP of the hard
routing, at a computationally acceptable cost.

## Repository contents

- [`hard_qkv_rosa_explained.py`](hard_qkv_rosa_explained.py) — the single-file,
  pure-Python, fully commented reference: the architecture definition, the exact
  backward algorithm, and a built-in self-test against the brute-force
  definition. Start here if you want to *understand* the algorithm.
- [`conv_rosa_transformer/`](conv_rosa_transformer/) — the production-oriented
  package: a C++ extension (bitwise-identical to the Python reference) with a
  JIT build, a `Conv∥ROSA` parallel-mixer Transformer (`RosaTransformer`), and a
  10-digit × 10-digit multiplication training demo with bundled results. Start
  here if you want to *use* or *benchmark* it.

## The architecture is not ours — the backward pass is

The forward architecture is taken directly from the official RWKV-8 ROSA
examples by **Bo Peng (BlinkDL)**: see `samx_qkv_slow` / `ROSA_QKV_B_1bit` in
[`251024_rosaQKV_run.py`](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v8/251024_rosaQKV_run.py)
in the [RWKV-LM `RWKV-v8` directory](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8)
(ROSA = Rapid Online Suffix Automaton; see also *RWKV-8 ROSA: Beyond Attention*
on [rwkv.com](https://rwkv.com)). This repo generalizes it slightly — n-bit hard
codes instead of 1-bit, and a separate learned `emb0` payload for the no-match
case instead of collapsing it with a matched `0` — but the retrieval semantics
are unchanged:

- The Q/K/V vector at every position is threshold-quantized into an n-bit hard
  code (a symbol).
- The output at position `t` does not depend on the whole history, only on one
  **retrieval**: in the prefix `K[:t]`, find the **latest endpoint** `r` of the
  longest suffix match of `Q[:t+1]`.
- If a match exists (`ell[t] > 0`), the output payload is `y[t] = ±emb1`
  (the sign is decided by the bits of `V[r+1]`); otherwise `y[t] = emb0`.

**What is new here** is the backward pass. Every step of the forward is a
discrete decision — quantization is a step function, routing is an argmax
(longest match + endpoint tie-break) — so by the textbook view the architecture
is "non-differentiable". The official implementation indeed ships no backward at
all (`samx_qkv_1bit_layer_op` is forward-only). This repository provides one.

## The backward pass is exact

The credit of every Q/K/V bit `(u, j)` is **defined** as: flip only this one bit,
leave everything else untouched, re-run the **hard forward**, and project the
output difference onto `grad_y`:

```
credit[u, j] = Σ_t grad_y[t] · (y_flip[t] - y_base[t])
```

This is the single-bit counterfactual VJP of the hard routing: it answers exactly
"if this one bit were flipped, how much would the whole downstream hard routing
change". The definition itself is ~20 lines (Part 0: `forward_naive` /
`backward_bruteforce`), but computing it directly costs `O(D · T⁴)` — one full
forward per bit. The rest of the file computes **the same quantity** fast:

- how flipping one bit changes the longest match is decomposed into finitely many
  analytic surfaces (deletion-context A/H oracles, one-bit repair bridges,
  maximal-run square certificates);
- a difference-range structure (DRS) then contracts the credit on those surfaces
  into gradients in `O(log T)` per surface.

The fast path and the brute-force definition are validated for bitwise numerical
agreement by the self-test at the end of the file.

## How this compares to existing open-source ROSA training schemes

Every open-source approach we are aware of either does not backpropagate through
the discrete routing at all, or uses a surrogate/approximate gradient, or only
handles a truncated variant. This repository is, to our knowledge, the first
**exact, untruncated, bit-level** backpropagation for the full QKV-ROSA semantics
with an acceptable (`~O(T log²T)`) cost:

| Project | Backward approach | Exact for the hard forward? | Cost |
| --- | --- | --- | --- |
| Official RWKV-v8 `samx_qkv` | none — the autograd op is forward-only; Q/K/V receive no gradient | n/a | — |
| [wjie98/rosa_soft](https://github.com/wjie98/rosa_soft) | exact hard forward + hand-designed **dense surrogate VJP** (concave route-score transform, probability-weighted credit, backward dropout) | no — biased surrogate by design | backward `O(T²·(W·D + D_p))` per head |
| [johanwind/wind_rosa](https://github.com/johanwind/wind_rosa) | exact single-bit-flip gradients — but for **truncated** ROSA (match length capped at K); the authors note the raw gradient requires external post-processing inside an LM | yes, but only for the truncated variant | linear in T (CUDA) |
| [zyaaa-ux/ROSA-Tuning](https://github.com/zyaaa-ux/ROSA-Tuning) | token-level "Local Counterfactual Gradient" perturbation (change-one-token simulation); K-gradients restricted to run-start positions, zero elsewhere | no — token-level heuristic, not the full bit-level counterfactual | per-token re-simulation |
| [aabbdev/rosa](https://github.com/aabbdev/rosa) | straight-through estimator around soft candidate ranking | no — differentiates a softened architecture | — |
| [bcml-labs/rosa-plus](https://github.com/bcml-labs/rosa-plus), [x-0D/RASP](https://github.com/x-0D/RASP) | statistical predictors; no gradient through routing | n/a | — |
| **this repository** | **exact single-bit counterfactual VJP**, all of Q/K/V, full untruncated semantics, bitwise-validated against the brute-force definition | **yes — by construction, and verified by self-test** | backward `O(T·log²T + Λ·log T + T·D)` |

The practical consequence: gradients computed here are not a training trick —
they are the true sensitivities of the hard routing, so optimizer behavior is
predictable and directly comparable across hyperparameters, and every gradient
can be audited against the 20-line brute-force definition.

## Complexity

Single Q/K/V bit stream of length `T`, bit width `D`.
Notation: `P` = number of causal one-bit Q/K symbol pairs (≤ `D·T²/2`, ~`D·T·log σ`
for random streams); `Λ` = total number of compiled repair terms/surfaces
(typically `O(T log T)`, higher for dense/periodic streams); `σ` = alphabet size.

| Stage | Time | Space |
| --- | --- | --- |
| Forward (suffix array + certificates) | `O(T · log²T)` | `O(T · log T)` |
| Index rebuild (backward workbench) | `O(T · log²T)` | `O(T · log T)` |
| A/H deletion oracles + K-deletion surfaces | `O(Σ_t (r_t+h_t) · log T) ≈ O(T log T)` | `O(T log T)` |
| Repair compilation: `shared_sparse` | `O(P · log T)` | `O(P)` |
| Repair compilation: `shared_diagonal` | `O(T²)` | `O(T + P)` |
| Repair compilation: `surface_run_certified` | `O(T log T + R_runs·log T + hits·log T)` | `O(C·log T)` |
| Numerical contraction (DRS + sweep + pointwise) | `O(Λ · log T + T · D)` | `O(Λ + T·D)` |
| **Forward total** | `O(T · log²T + T · D)` | `O(T · log T)` |
| **Backward total** | `O(T · log²T + Λ · log T + T · D)` | `O(T · log T + Λ)` |

### Measured backward/forward time ratio

Calibrated with the C++ reference implementation (2026-08), single stream,
mean of best-of-N, with automatic backend switching (`λ = Λ/T` is the per-position
repair density; `b₁/a₁ ≈ 4~6` because the backward rebuilds 4 suffix arrays plus
the orthogonal and A/H oracles):

| T | random (σ=64) | dense (σ=2) | periodic (long matches) |
| --- | --- | --- | --- |
| 128 | 42 / 63 | 101 / 120 | 69 / 60 |
| 256 | 54 / 57 | 76 / 84 | 75 / 92 |
| 512 | 48 / 57 | 89 / 95 | 77 / 84 |
| 1024 | 38 / 52 | 92 / 105 | 101 / 114 |
| 2048 | 46 / 128* | 96 / 114 | 144 / 136 |
| 4096 | 57 / 117* | 105 / 102 | 260 / 269 |

(Each cell is `D=2 / D=8`; `*` marks where the backend switches from
`shared_sparse` to `surface_run_certified`, producing a step in the ratio.)

Empirical formulas (C++, single stream):

- sparse random streams: `R ≈ 45 + 2·log₂T` (around 50)
- dense small-alphabet streams: `R ≈ 100` (insensitive to T)
- periodic long matches: `R ≈ 30·log₂T − 150` (T ≥ 256; λ grows with T)

Note this is the ratio of the ROSA core structure itself. In a full layer, the
four dense Q/K/V/O projections (`O(B·T·C²)` GEMM, whose backward ≈ 2× forward)
dilute it — the larger `C`, the closer the whole-layer ratio gets to 2~3.

## Usage

```python
import torch
from hard_qkv_rosa_explained import NBitQKVRosa

layer = NBitQKVRosa(channels=64, n_bits=8)
x = torch.randn(2, 128, 64)          # [B, T, C]
y = layer(x)                          # forward: O(B·G·T·log²T)
y.sum().backward()                    # exact single-bit counterfactual credit,
                                      # mapped back to the logits via credit_to_logit
```

Run the built-in verification:

```bash
python hard_qkv_rosa_explained.py
# three-way agreement check: fast forward == brute force, fast backward == brute
# force (Part 0 definitions), followed by a measured backward/forward ratio demo
```

The file is self-contained: its only dependency is `torch`. The algorithm matches
the reference implementation function by function; its C++ port agrees with this
reference bitwise on 25 streams × all backends.

### Public API

- `NBitQKVRosa(channels, n_bits, init_emb1=1.0, bias=True, tau=1.0, credit_to_logit='bernoulli')`
  — drop-in `nn.Module` with Q/K/V/O projections and the exact backward.
- `exact_stream_bit_credits(q, k, v_bits, grad_y, emb0, emb1, D)` — the fast
  single-stream entry point: returns exact per-bit Q/K/V credits.
- `forward_naive` / `backward_bruteforce` — the Part 0 canonical definitions
  (read these first; ~40 lines state the full semantics).
- `self_test()` — three-way agreement check against the brute-force definitions.
- `measure_backward_forward_ratio(T, D, regime)` — timing-ratio probe
  (pure-Python constants inflate the ratio; see the C++ calibration table above).

## File map

The single file is organized into numbered parts (banner comments inside):

- **Part 0** — canonical definitions: brute-force forward + backward
- **Part 1** — core data structures: the "surfaces" credit flows through
  (affine deletion runs / threshold runs / repair tracks / IR)
- **Part 2** — fast forward: suffix array + LCE + static certificates,
  `O(T log²T)` for `ell`/`route`
- **Part 3** — range-query toolbox: existence / leftmost / rightmost of a symbol
  in an interval in `O(log n)`
- **Part 4** — bidirectional position/suffix index: the backward workbench
  (Q/K, both directions, cut primitives)
- **Part 5** — deletion-context oracles A/H: how the baseline route changes when
  one K bit is deleted
- **Part 6** — Q-side deletion surfaces
- **Part 7** — packing & standalone suffix tools (payload LCE for contraction)
- **Part 8** — numerical contraction: exact summation of surface credit
  (DRS / K-deletion sweep line / pointwise payloads)
- **Part 9** — repair-term compiler I: zero-baseline surfaces + direct Q repair
- **Part 10** — repair-term compiler II: K-side surface compilation (the core
  stage of Equality-Shadow)
- **Part 11** — credit→logits mapping (`bernoulli` / `identity`) and statistics
- **Part 12** — shared one-bit repair bridge (sparse / diagonal backends)
- **Part 13** — maximal-run square repair certificates (Equality-Shadow engine)
- **Part 14** — bridge post-processing: first-win queries and envelope segments
- **Part 15** — adaptive backend selection + IR assembly + single-stream entry
- **Part 16** — `autograd.Function` and `nn.Module`
- **Part 17** — self-test and measurement utilities

## Limitations and future work

The algorithm published here is exact and practical, but it is not the last
word, and we want to be honest about that:

- Its backward costs `O(T·log²T + Λ·log T + T·D)` per stream. Neither the
  `log²T` factor nor the `Λ`-dependent term is obviously optimal, and the
  pure-Python reference carries large constants (the C++ port in
  `conv_rosa_transformer/` exists precisely to bring them down).
- I (Xiaoiec) also have a different algorithm locally on my own machine with
  **worst-case `O(T·log³T)`** total complexity. It is deliberately *not*
  published here: its constant factors and memory requirements are so large
  that in practice it runs far slower than the algorithm in this repository
  and exhausts memory long before any asymptotic advantage could show. It is
  mentioned only as evidence that the complexity frontier for this problem has
  not settled.

If you can design a backward algorithm for this architecture that is cheaper in
compute or memory — exact, or with a principled error bound — that would be a
genuinely valuable contribution to the ROSA ecosystem. Issues and pull requests
are very welcome.

## Related work in the ROSA ecosystem

Community ROSA projects listed by the
[official RWKV-LM RWKV-v8 README](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8):

- [aabbdev/rosa](https://github.com/aabbdev/rosa) (`rosa-torch`) — differentiable
  ROSA with an exact suffix automaton, soft candidate ranking and straight-through
  gradients. Complementary to this repo: it learns *around* the hard routing,
  while this repo differentiates *through* the hard routing exactly.
- [wjie98/rosa_soft](https://github.com/wjie98/rosa_soft),
  [johanwind/wind_rosa](https://github.com/johanwind/wind_rosa),
  [zyaaa-ux/ROSA-Tuning](https://github.com/zyaaa-ux/ROSA-Tuning) — other
  training-oriented ROSA projects.
- [bcml-labs/rosa-plus](https://github.com/bcml-labs/rosa-plus) — ROSA+ with a
  fallback statistical predictor.
- [x-0D/RASP](https://github.com/x-0D/RASP).
- [KakaruHayate/RWKV8-ROSA-FPGA](https://github.com/KakaruHayate/RWKV8-ROSA-FPGA) —
  native ROSA on FPGA; [Juste-Leo2/ROSA-GPU-RWKV8](https://github.com/Juste-Leo2/ROSA-GPU-RWKV8) — GPU implementation.

## License

Copyright 2026 xiaoiecc. Licensed under the Apache License, Version 2.0 — see
[LICENSE](LICENSE) for the full text.

ROSA and the QKV-ROSA forward architecture are due to Bo Peng (BlinkDL) for
RWKV-8; this repository implements that architecture and contributes the exact
backward algorithm, and claims no authorship of ROSA itself. It is an
independent implementation, not an official RWKV distribution.
