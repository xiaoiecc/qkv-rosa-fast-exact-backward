"""Small independent definitions for auditing hard QKV-ROSA.

``forward_naive`` enumerates every causal endpoint and compares characters.
``backward_bruteforce`` flips each bit, reruns that definition, and projects the
output change. These deliberately slow functions are small-input test oracles.
They share no suffix indices, deletion formulas, or repair code with the engine.

For existing imports, layer names and ``exact_stream_bit_credits`` are re-exported
from the single production implementation. The credit function returns five
results: Q/K/V credits, match lengths, and routes. Diagnostics are available via
``rosa_core.credits_numpy``; there is no alternate optimized Python backend.
"""
from __future__ import annotations

import operator
from typing import Sequence

import torch

if __package__:
    from .rosa_core import exact_stream_bit_credits
    from .rosa_layer import (
        NBitQKVRosa, NBitQKVRosaCpp, NBitQKVRosaForward, NBitQKVRosaForwardCpp,
    )
else:
    from rosa_core import exact_stream_bit_credits
    from rosa_layer import (
        NBitQKVRosa, NBitQKVRosaCpp, NBitQKVRosaForward, NBitQKVRosaForwardCpp,
    )


def matching_stats_naive(q: Sequence[int], k: Sequence[int]):
    """Return (lengths, routes) by the literal longest/latest suffix rule.

    A route is a K endpoint r strictly before t. Empty matches have route -1,
    including t=0. A later endpoint replaces an equal positive match length.
    Time O(T**3) and auxiliary space O(T).
    """
    if len(q) != len(k):
        raise ValueError("q and k must have equal lengths")
    lengths, routes = [0] * len(q), [-1] * len(q)
    for t in range(len(q)):
        for r in range(t):
            length = 0
            while length <= r and q[t - length] == k[r - length]:
                length += 1
            if length > 0 and length >= lengths[t]:
                lengths[t], routes[t] = length, r
    return lengths, routes


def _validate(q, k, v_bits, emb0, emb1, D):
    D = operator.index(D)
    if not 1 <= D <= 64:
        raise ValueError("D must be in [1,64]")
    q = [operator.index(x) for x in q]
    k = [operator.index(x) for x in k]
    if len(q) != len(k) or any(x < 0 or x >= 1 << D for x in q + k):
        raise ValueError("q and k must be equal-length unsigned D-bit streams")
    if v_bits.shape != (len(q), D) or emb0.shape != (D,) or emb1.shape != (D,):
        raise ValueError("v_bits and embeddings must have shapes [T,D] and [D]")
    if bool(((v_bits != 0) & (v_bits != 1)).any()):
        raise ValueError("v_bits must contain only 0/1")
    if emb0.device != emb1.device or v_bits.device != emb0.device:
        raise ValueError("v_bits and embeddings must share a device")
    return q, k, D


@torch.no_grad()
def forward_naive(q: Sequence[int], k: Sequence[int], v_bits: torch.Tensor,
                  emb0: torch.Tensor, emb1: torch.Tensor, D: int):
    """Literal hard forward: return (y, lengths, routes).

    Output t selects V[r+1] for its winning endpoint r; an unmatched row emits
    emb0. The payload is (2*V[r+1]-1)*emb1. Output dtype promotes the two
    embedding dtypes, and its device is the embedding device.

    Time O(T**3 + T*D), auxiliary space O(T*D).
    """
    q, k, D = _validate(q, k, v_bits, emb0, emb1, D)
    lengths, routes = matching_stats_naive(q, k)
    y = torch.empty((len(q), D), dtype=torch.promote_types(emb0.dtype, emb1.dtype),
                    device=emb0.device)
    for t, r in enumerate(routes):
        if r == -1:
            y[t] = emb0
        else:
            y[t] = (v_bits[r + 1].to(y.dtype) * 2.0 - 1.0) * emb1
    return y, lengths, routes


@torch.no_grad()
def backward_bruteforce(q: Sequence[int], k: Sequence[int], v_bits: torch.Tensor,
                        grad_y: torch.Tensor, emb0: torch.Tensor,
                        emb1: torch.Tensor, D: int):
    """Independent definition of all Q/K/V one-bit counterfactual credits.

    Computes sum_t grad_y[t] dot (y_flip[t]-y_base[t]) for each flipped bit.
    Float inputs are promoted to CPU binary64 for this audit; returned credits
    are CPU float64. Routes use exact integer comparisons, while numerical
    projections still have ordinary floating-point rounding.

    Time O(D*T**4 + D**2*T**2), space O(T*D). Use only on small streams.
    """
    q, k, D = _validate(q, k, v_bits, emb0, emb1, D)
    if grad_y.shape != (len(q), D):
        raise ValueError("grad_y must have shape [T,D]")
    v = v_bits.detach().to(device="cpu", dtype=torch.uint8)
    gy = grad_y.detach().to(device="cpu", dtype=torch.float64)
    e0 = emb0.detach().to(device="cpu", dtype=torch.float64)
    e1 = emb1.detach().to(device="cpu", dtype=torch.float64)
    y0, _, _ = forward_naive(q, k, v, e0, e1, D)
    cq, ck, cv = [torch.zeros((len(q), D), dtype=torch.float64) for _ in range(3)]
    for s in range(len(q)):
        for j in range(D):
            q1 = q.copy()
            q1[s] ^= 1 << j
            y1, _, _ = forward_naive(q1, k, v, e0, e1, D)
            cq[s, j] = ((y1 - y0) * gy).sum()
            k1 = k.copy()
            k1[s] ^= 1 << j
            y1, _, _ = forward_naive(q, k1, v, e0, e1, D)
            ck[s, j] = ((y1 - y0) * gy).sum()
            v1 = v.clone()
            v1[s, j] ^= 1
            y1, _, _ = forward_naive(q, k, v1, e0, e1, D)
            cv[s, j] = ((y1 - y0) * gy).sum()
    return cq, ck, cv


__all__ = [
    "matching_stats_naive", "forward_naive", "backward_bruteforce",
    "exact_stream_bit_credits", "NBitQKVRosa", "NBitQKVRosaCpp",
    "NBitQKVRosaForward", "NBitQKVRosaForwardCpp",
]
