# Exact backpropagation for the QKV-ROSA discrete retrieval architecture.
# The forward architecture follows the RWKV-8 ROSA examples by Bo Peng (BlinkDL);
# the exact backward algorithm is this repository's contribution.
# Copyright 2026 xiaoiecc
# SPDX-License-Identifier: MIT
"""Hard QKV-ROSA: mathematical definition and a guide to the shared engine.

1. The hard forward
===================
Each stream contains T unsigned D-bit Q/K symbols and T payload bit vectors.
For output t and a strictly earlier K endpoint r<t, let F_t(r) be the common
suffix length of Q[:t+1] and K[:r+1]. Choose the lexicographic maximum

    (length[t], route[t]) = max_r (F_t(r), r),

with the convention that every zero-length result is (0, -1). Thus ties select
the latest endpoint. A successful route selects the *next* payload V[r+1]:

    y[t] = (2*V[r+1]-1)*emb1,       if route[t] = r >= 0;
    y[t] = emb0,                   otherwise.

The strict r<t condition still permits selecting V[t]. Each contiguous group
of D channels in the layer is a separate stream. A word uses bit j at weight
2**j; the 64th bit is an unsigned bit, not a signed integer sentinel.

``forward_naive`` below is imported from the small independent oracle module.
Its endpoint/character enumeration takes O(T**3 + T*D) time. It is suitable for
reading the definition and auditing small examples.

2. What the backward computes
=============================
For X in {Q,K,V}, flip just bit (s,j), run the same hard forward, and define

    C_X[s,j] = sum_t grad_y[t] dot (y_flip[t] - y_base[t]).

This finite one-bit counterfactual credit is the target quantity. It is the
change in the downstream *linearization* given grad_y; it does not rerun a
nonlinear downstream loss. It is also separate from the ordinary derivative
of the threshold function. ``backward_bruteforce`` implements this definition
literally, requiring O(D*T**4 + D**2*T**2) time and O(T*D) auxiliary space.

The module exposes two explicit ways to map this credit to a logit update:

    b = 1[z>0];  delta = (1-2*b)*C_X[s,j].
    identity:    grad_z = delta.
    bernoulli:   grad_z = sigmoid(z/tau)*(1-sigmoid(z/tau))/tau * delta.

The default is bernoulli. The hard forward always thresholds at zero. These
are user-selected training rules; embedding gradients are ordinary derivatives
of the selected embedding values.

3. A single delete/repair decomposition
=======================================
A Q/K bit flip divides candidates into two sets. Candidates unaffected by the
changed symbol remain valid: their best route is the delete context. A repaired
candidate has one mismatch at the edited bit and exact matches everywhere else.
The flipped route is the lexicographic maximum of delete and repair. A V flip
changes only the value of outputs already routed to that payload.

The engine represents these choices without enumerating every edit/time pair:

* Repeated occurrences of a long context inside a short window form an
  arithmetic progression. Deletion streams use that progression and its
  capped-affine suffix lengths, with explicit boundary exceptions.
* Q repair divides suffix contexts by occurrence count. Frequent contexts
  produce shared winner intervals; rare contexts enumerate only their few
  possible sources.
* K repair divides by context length. Short contexts become virtual edited
  suffix words and offline dominance queries. Long contexts cross a block
  anchor; their exact extensions and blocker inequalities produce intervals.
* An interval route ends at t + beta. Its numeric contribution is a restricted
  shifted correlation. Shared FFT blocks and direct fringes evaluate these
  correlations without expanding all interval rows.

The thresholds balance costs inside this one algorithm. They do not select
kernels for particular input families. The two full tutorials start from the
literal O(D*T**4) counterfactual implementation and derive these representations:
ALGORITHM.md and ALGORITHM.zh-CN.md. Independent naive functions remain only
as definitions and small-input test oracles.

4. Complexity, training choices and numeric scope
=================================================
For fixed bit width, numeric dimension and selected joint-pair budget, the
proved bounds are soft-O(T**(7/4)) time and soft-O(T**(4/3)) working space.
Width-dependent terms are given in the tutorial; finite-length timings also
depend on the actual number of intervals and query fringes. A better worst-case
bound does not guarantee faster execution on every input.

The default adds a diagonal output trust penalty to Q/K route credits and a
small number of disjoint joint-bit corrections. Select credit_order="first"
for the original linear finite credit. V stays linear. Both fit the usual
model forward / loss.backward() / optimizer.step() loop.

randomized=True enables a bounded full-key verified cache: its hits cannot
change route or numerical outputs. backward_mode="approximate" explicitly
opts into a residual-row estimator, or centered-feature PPS with a positive
approximation_channels budget. These noisy choices have no universal loss
or relative-error guarantee.

Routing is exact integer logic. Numeric contractions use binary64 with
expansion accumulation of already rounded events, plus FFT where profitable.
This is not exact real arithmetic or a promise of bitwise equality with an
older reduction order. Cache on/off is bitwise equal within this engine.
Strongly cancelling values can still have substantial relative error.

The core is CPU code. The PyTorch adapter accepts its normal tensor device,
but CUDA tensors incur CPU transfers and synchronization. Component timings
do not measure a whole Transformer or a GPU kernel.

5. Running the implementation and the independent audit
=======================================================
    from hard_qkv_rosa_explained import NBitQKVRosa
    layer = NBitQKVRosa(channels=64, n_bits=8)
    y = layer(x)  # x: [B,T,64]
    y.sum().backward()

    # Single CPU stream, five outputs:
    cq, ck, cv, lengths, routes = exact_stream_bit_credits(
        q_words, k_words, v_bits, grad_y, emb0, emb1, D)

    python hard_qkv_rosa_explained.py
    python -m conv_rosa_transformer.self_test
    python -m unittest discover -s tests -v

A C++17 compiler, NumPy, and PyTorch are required for the layer and self-test.
The native build is lazy and uses the same shared core for every public alias.
"""
from __future__ import annotations

from conv_rosa_transformer.rosa_core import (
    credits_numpy, exact_stream_bit_credits, matching_stats, pack_bits,
)
from conv_rosa_transformer.rosa_layer import (
    NBitQKVRosa, NBitQKVRosaCpp, NBitQKVRosaForward, NBitQKVRosaForwardCpp,
)
from conv_rosa_transformer.rosa_reference import (
    backward_bruteforce, forward_naive, matching_stats_naive,
)


def self_test(n_cases: int = 20, seed: int = 43, verbose: bool = True):
    """Run independent small-stream definition checks against the shared core."""
    from conv_rosa_transformer.self_test import self_test as run
    return run(n_cases=n_cases, seed=seed, verbose=verbose)


__all__ = [
    "NBitQKVRosa", "NBitQKVRosaCpp", "NBitQKVRosaForward", "NBitQKVRosaForwardCpp",
    "exact_stream_bit_credits", "credits_numpy", "matching_stats", "pack_bits",
    "forward_naive", "matching_stats_naive", "backward_bruteforce", "self_test",
]


if __name__ == "__main__":
    self_test()
