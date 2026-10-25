"""Batched hard QKV-ROSA with exact single-bit counterfactual credits.

The suffix matcher and credit engine are the single CPU implementation in
``rosa_core``. Tensor projections, payload selection, and the explicit
credit-to-logit rule run on the input device. CUDA tensors therefore incur CPU
transfers and synchronization for routing and credits; this is not a GPU kernel.

The Q/K/V/O projection and embedding parameter names preserve existing state
dicts. All historical layer class names below alias this same implementation.
"""
from __future__ import annotations

import math
import operator

import torch
import torch.nn as nn

if __package__:
    from . import rosa_core
else:  # ``python conv_rosa_transformer/<script>.py``
    import rosa_core


def _flip_credit_to_logit_grad(z, bits, flip_credit, tau, mode):
    """Map the finite flip credit to the user-selected continuous update rule.

    The finite credit is L(1-b)-L(b), where L is the upstream linearization.
    Multiplication by 1-2b gives L(1)-L(0). The Bernoulli mode multiplies
    this difference by the derivative of sigmoid(z/tau); identity uses it
    directly. This rule is separate from differentiation of the hard step.
    """
    orient = 1.0 - 2.0 * bits.to(dtype=z.dtype, device=z.device)
    l1_minus_l0 = orient * flip_credit.to(dtype=z.dtype, device=z.device)
    if mode == "bernoulli":
        p = torch.sigmoid(z / tau)
        return (p * (1.0 - p) / tau) * l1_minus_l0
    if mode == "identity":
        return l1_minus_l0
    raise ValueError(f"unknown credit_to_logit mode: {mode}")


class _HardQKVRosaFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, emb0, emb1, n_bits, tau, credit_to_logit):
        if q.ndim != 3 or k.shape != q.shape or v.shape != q.shape:
            raise ValueError("q, k, v must have the same [B,T,C] shape")
        if not all(z.is_floating_point() for z in (q, k, v)):
            raise TypeError("q, k, v must be floating-point tensors")
        if any(z.device != q.device or z.dtype != q.dtype for z in (k, v)):
            raise ValueError("q, k, v must share a device and dtype")
        B, T, C = q.shape
        D = int(n_bits)
        if not 1 <= D <= 64 or C == 0 or C % D:
            raise ValueError("n_bits must be in [1,64] and divide positive C")
        if emb0.shape != (1, 1, C) or emb1.shape != (1, 1, C):
            raise ValueError("emb0 and emb1 must have shape [1,1,C]")
        if emb0.device != q.device or emb1.device != q.device:
            raise ValueError("embeddings must be on the q/k/v device")
        G = C // D
        qb = (q > 0).to(torch.uint8).reshape(B, T, G, D)
        kb = (k > 0).to(torch.uint8).reshape(B, T, G, D)
        vb = (v > 0).to(torch.uint8).reshape(B, T, G, D)
        qc, kc = qb.detach().cpu(), kb.detach().cpu()
        route = torch.full((B, T, G), -1, dtype=torch.long)
        q_all = [[None] * G for _ in range(B)]
        k_all = [[None] * G for _ in range(B)]
        for b in range(B):
            for g in range(G):
                qs = rosa_core.pack_bits(qc[b, :, g, :])
                ks = rosa_core.pack_bits(kc[b, :, g, :])
                _, rr = rosa_core.matching_stats(qs, ks)
                q_all[b][g], k_all[b][g] = qs, ks
                route[b, :, g] = torch.tensor(rr, dtype=torch.long)
        route = route.to(q.device)
        indices = (route + 1).clamp_min(0)[..., None].expand(B, T, G, D)
        selected = torch.gather(vb, dim=1, index=indices)
        e0 = emb0.expand(B, T, C).reshape(B, T, G, D)
        e1 = emb1.expand(B, T, C).reshape(B, T, G, D)
        sign = selected.to(e1.dtype) * 2.0 - 1.0
        y = torch.where((route >= 0)[..., None], sign * e1, e0)
        ctx.n_bits, ctx.groups = D, G
        ctx.tau, ctx.credit_to_logit = float(tau), str(credit_to_logit)
        ctx.q_all, ctx.k_all = q_all, k_all
        ctx.save_for_backward(q, k, v, qb, kb, vb, route, emb0, emb1)
        return y.reshape(B, T, C)

    @staticmethod
    def backward(ctx, grad_y):
        q, k, v, qb, kb, vb, route, emb0, emb1 = ctx.saved_tensors
        B, T, C = q.shape
        D, G = ctx.n_bits, ctx.groups
        gy4 = grad_y.reshape(B, T, G, D)
        vbc = vb.detach().cpu()
        # The core uses binary64 accumulation. Convert before NumPy access,
        # including for bfloat16/autocast inputs whose .numpy() is unsupported.
        gyc = gy4.detach().to(device="cpu", dtype=torch.float64)
        e0c = emb0.detach().reshape(G, D).to(device="cpu", dtype=torch.float64)
        e1c = emb1.detach().reshape(G, D).to(device="cpu", dtype=torch.float64)
        qfc = torch.zeros((B, T, G, D), dtype=torch.float64)
        kfc, vfc = torch.zeros_like(qfc), torch.zeros_like(qfc)
        for b in range(B):
            for g in range(G):
                qa, ka, va, _, _ = rosa_core.exact_stream_bit_credits(
                    ctx.q_all[b][g], ctx.k_all[b][g], vbc[b, :, g, :],
                    gyc[b, :, g, :], e0c[g], e1c[g], D,
                )
                qfc[b, :, g, :], kfc[b, :, g, :], vfc[b, :, g, :] = qa, ka, va
        grad_q, grad_k, grad_v = (
            _flip_credit_to_logit_grad(
                z, bits.reshape(B, T, C), credit.reshape(B, T, C),
                ctx.tau, ctx.credit_to_logit,
            )
            for z, bits, credit in ((q, qb, qfc), (k, kb, kfc), (v, vb, vfc))
        )
        matched = (route >= 0)[..., None]
        indices = (route + 1).clamp_min(0)[..., None].expand(B, T, G, D)
        selected = torch.gather(vb, dim=1, index=indices)
        sign = selected.to(gy4.dtype) * 2.0 - 1.0
        # These are ordinary derivatives of the selected embedding values.
        ge0 = torch.where(~matched, gy4, torch.zeros_like(gy4)).sum(dim=(0, 1))
        ge1 = torch.where(matched, gy4 * sign, torch.zeros_like(gy4)).sum(dim=(0, 1))
        ge0 = ge0.reshape(1, 1, C).to(dtype=emb0.dtype, device=emb0.device)
        ge1 = ge1.reshape(1, 1, C).to(dtype=emb1.dtype, device=emb1.device)
        return grad_q, grad_k, grad_v, ge0, ge1, None, None, None


class NBitQKVRosa(nn.Module):
    """Hard grouped QKV-ROSA with one exact CPU credit engine.

    ``x`` has shape [B,T,C]. Each contiguous ``n_bits`` channels form one
    symbol; bit j has unsigned weight 2**j, including j=63. Each group uses
    strictly causal K endpoints r<t, with longest match then latest endpoint.

    ``credit_to_logit`` is ``bernoulli`` (default) or ``identity``. The former
    uses sigmoid(z/tau)'s derivative with the exact one-bit credit. Hard
    quantization in the forward pass is always z>0.
    """

    def __init__(self, channels: int, n_bits: int, init_emb1: float = 1.0,
                 bias: bool = True, tau: float = 1.0,
                 credit_to_logit: str = "bernoulli"):
        super().__init__()
        channels, n_bits = operator.index(channels), operator.index(n_bits)
        if channels <= 0 or not 1 <= n_bits <= 64 or channels % n_bits:
            raise ValueError("n_bits must be in [1,64] and divide positive channels")
        tau = float(tau)
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and positive")
        if credit_to_logit not in ("bernoulli", "identity"):
            raise ValueError(f"unknown credit_to_logit mode: {credit_to_logit}")
        self.channels, self.n_bits = channels, n_bits
        self.groups = channels // n_bits
        self.tau, self.credit_to_logit = tau, str(credit_to_logit)
        self.q_proj = nn.Linear(channels, channels, bias=bias)
        self.k_proj = nn.Linear(channels, channels, bias=bias)
        self.v_proj = nn.Linear(channels, channels, bias=bias)
        self.o_proj = nn.Linear(channels, channels, bias=bias)
        self.emb0 = nn.Parameter(torch.zeros(1, 1, channels))
        self.emb1 = nn.Parameter(torch.full((1, 1, channels), float(init_emb1)))

    def _hard_rosa_from_qkv(self, q, k, v):
        return _HardQKVRosaFn.apply(
            q, k, v, self.emb0, self.emb1,
            self.n_bits, self.tau, self.credit_to_logit,
        )

    def forward(self, x):
        if x.ndim != 3 or x.size(-1) != self.channels:
            raise ValueError("x must have shape [B,T,C]")
        return self.o_proj(self._hard_rosa_from_qkv(
            self.q_proj(x), self.k_proj(x), self.v_proj(x),
        ))


NBitQKVRosaCpp = NBitQKVRosa
NBitQKVRosaForward = NBitQKVRosa
NBitQKVRosaForwardCpp = NBitQKVRosa
__all__ = [
    "NBitQKVRosa", "NBitQKVRosaCpp", "NBitQKVRosaForward", "NBitQKVRosaForwardCpp",
]
