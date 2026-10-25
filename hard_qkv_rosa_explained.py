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

The production implementation applies this same decomposition on every input.
Its data structures exploit exact structural certificates within the algorithm:

* A suffix automaton advances ordinary edited Q suffixes. Its terminal-order
  occurrence chain activates endpoints as they become causal. Reporting visits
  current occurrences without scanning future entries. Latest-endpoint queries
  can return directly when an exact upper-bound or contiguity certificate holds.

* The deletion index represents endpoint/length families with formula intervals.
  If Q's equal-character suffix has length h and a matching K run begins at l,
  then F_t(r)=min(h,r-l+1), except at the single simultaneous boundary r=l+h-1.
  That boundary receives a genuine suffix query, since matching may cross both
  runs. Intersections with the other deletion envelope are solved as linear
  inequalities; compressed families need not be expanded into all owners.

* An edited Q suffix may leave ordinary SAM processing only if its *short full
  edited suffix* occurs once in the full K stream and has a matching next step.
  Its remaining route then has the form (t+alpha, t+beta), up to a certified
  extension endpoint. A mismatch permanently ends this family. Q credits use
  interval snapshots; K candidates share heaps keyed by their fixed owner/bit.
  Ordinary states and these certified families contribute to the same maximum.

* Numeric contraction follows these representations. Moving payload endpoints
  use owner ranges with two gradient buckets per baseline bit; equal payload
  bits cancel structurally. Shared payload classes reuse dot products under
  a fixed admission bound of at most 4D classes, with at most 4TD cached doubles.

Read the current English/Chinese explanations in ALGORITHM.md and ALGORITHM.zh-CN.md
alongside ``conv_rosa_transformer/csrc``. The implementation is split into small
files for the deletion index, suffix structures, accumulation, and orchestration;
there is one production credit engine. This teaching module re-exports it.

4. Complexity and numerical scope
=================================
Costs depend on actual ordinary edited-state visits, causal occurrences,
compressed deletion records, and the lifetimes of certified families. The sum
of family lifetimes can remain Theta(T**2). Compression substantially changes
work on repeated structure but does not establish a generally subquadratic
worst-case algorithm. A suffix state is not automatically O(1) work, and a
compact route family is not automatically O(1) numeric contraction work.

Routing uses exact unsigned symbols and integer comparisons. Numeric credits
use binary64 arithmetic, with error-free expansion accumulation for cancellation-
sensitive ranges and snapshots. Nonzero residuals are retained, growing beyond
inline storage when required. Already rounded dot-product events do not become
arbitrary-precision real arithmetic. Float outputs are compared with documented
tolerances; the implementation does not promise bitwise parity with another
summation order. Compilation preserves the required floating-point semantics.

The core is CPU code. PyTorch layers accept their normal tensor device and
explicitly transfer routing/credit inputs to CPU. GPU tensors incur transfers
and synchronization; CPU core measurements are not GPU or training throughput
measurements.

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
