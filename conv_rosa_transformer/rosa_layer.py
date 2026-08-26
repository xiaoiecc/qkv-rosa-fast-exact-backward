"""Hard n-bit QKV-ROSA layer backed by the C++ extension.

Parameter names identical to the Python reference; backward calls the C++
exact_stream_bit_credits (bitwise identical to the bundled Python reference).
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import build

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = build()
    return _ext


def _flip_credit_to_logit_grad(z, bits, flip_credit, tau, mode):
    orient = 1.0 - 2.0 * bits.to(dtype=z.dtype)
    l1_minus_l0 = orient * flip_credit.to(dtype=z.dtype, device=z.device)
    if mode == "bernoulli":
        p = torch.sigmoid(z / tau)
        return (p * (1.0 - p) / tau) * l1_minus_l0
    if mode == "identity":
        return l1_minus_l0
    raise ValueError(f"unknown credit_to_logit mode: {mode}")


class _HardQKVRosaCppFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, emb0, emb1, n_bits: int, tau: float, credit_to_logit: str):
        ext = _get_ext()
        B, T, C = q.shape
        D = int(n_bits)
        if k.shape != q.shape or v.shape != q.shape or C % D:
            raise ValueError("invalid q/k/v shape or n_bits")
        G = C // D
        qb = (q > 0).to(torch.uint8).reshape(B, T, G, D)
        kb = (k > 0).to(torch.uint8).reshape(B, T, G, D)
        vb = (v > 0).to(torch.uint8).reshape(B, T, G, D)
        qc = qb.detach().cpu()
        kc = kb.detach().cpu()
        route = torch.full((B, T, G), -1, dtype=torch.long)
        mlen = torch.zeros((B, T, G), dtype=torch.long)
        q_all = [[None] * G for _ in range(B)]
        k_all = [[None] * G for _ in range(B)]
        for b in range(B):
            for g in range(G):
                qs = ext.pack_bits(qc[b, :, g, :])
                ks = ext.pack_bits(kc[b, :, g, :])
                ell, rr = ext.matching_stats(qs, ks)
                q_all[b][g] = qs
                k_all[b][g] = ks
                route[b, :, g] = torch.tensor(rr, dtype=torch.long)
                mlen[b, :, g] = torch.tensor(ell, dtype=torch.long)
        route = route.to(q.device)
        mlen = mlen.to(q.device)
        vidx = (route + 1).clamp_min(0)
        gidx = vidx[..., None].expand(B, T, G, D)
        selected = torch.gather(vb, dim=1, index=gidx)
        matched = route >= 0
        e0 = emb0.expand(B, T, C).reshape(B, T, G, D)
        e1 = emb1.expand(B, T, C).reshape(B, T, G, D)
        sign = selected.to(e1.dtype) * 2.0 - 1.0
        y = torch.where(matched[..., None], sign * e1, e0).reshape(B, T, C)
        ctx.n_bits = D
        ctx.groups = G
        ctx.tau = float(tau)
        ctx.credit_to_logit = str(credit_to_logit)
        ctx.q_all = q_all
        ctx.k_all = k_all
        ctx.save_for_backward(q, k, v, qb, kb, vb, route, mlen, emb0, emb1)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        ext = _get_ext()
        q, k, v, qb, kb, vb, route, _, emb0, emb1 = ctx.saved_tensors
        B, T, C = q.shape
        D, G = ctx.n_bits, ctx.groups
        gy4 = grad_y.reshape(B, T, G, D)
        vbc = vb.detach().cpu()
        gyc = gy4.detach().cpu()
        e0c = emb0.detach().reshape(G, D).cpu()
        e1c = emb1.detach().reshape(G, D).cpu()
        qfc = torch.zeros((B, T, G, D), dtype=gyc.dtype)
        kfc = torch.zeros_like(qfc)
        vfc = torch.zeros_like(qfc)
        for b in range(B):
            for g in range(G):
                qa, ka, va, _, _ = ext.exact_stream_bit_credits(
                    ctx.q_all[b][g], ctx.k_all[b][g],
                    vbc[b, :, g, :], gyc[b, :, g, :], e0c[g], e1c[g], D)
                qfc[b, :, g, :] = qa
                kfc[b, :, g, :] = ka
                vfc[b, :, g, :] = va
        qfc = qfc.reshape(B, T, C)
        kfc = kfc.reshape(B, T, C)
        vfc = vfc.reshape(B, T, C)
        grad_q = _flip_credit_to_logit_grad(q, qb.reshape(B, T, C), qfc, ctx.tau, ctx.credit_to_logit)
        grad_k = _flip_credit_to_logit_grad(k, kb.reshape(B, T, C), kfc, ctx.tau, ctx.credit_to_logit)
        grad_v = _flip_credit_to_logit_grad(v, vb.reshape(B, T, C), vfc, ctx.tau, ctx.credit_to_logit)
        matched = (route >= 0).reshape(B, T, G, 1)
        vidx = (route + 1).clamp_min(0)
        gidx = vidx[..., None].expand(B, T, G, D)
        selected = torch.gather(vb, dim=1, index=gidx)
        sign = selected.to(gy4.dtype) * 2.0 - 1.0
        ge0 = torch.where(~matched, gy4, torch.zeros_like(gy4)).sum(dim=(0, 1)).reshape(1, 1, C).to(
            dtype=emb0.dtype, device=emb0.device)
        ge1 = torch.where(matched, gy4 * sign, torch.zeros_like(gy4)).sum(dim=(0, 1)).reshape(1, 1, C).to(
            dtype=emb1.dtype, device=emb1.device)
        return grad_q, grad_k, grad_v, ge0, ge1, None, None, None


class NBitQKVRosaCpp(nn.Module):
    """Drop-in C++-accelerated hard n-bit QKV-ROSA layer (same params, same math)."""

    def __init__(self, channels: int, n_bits: int, init_emb1: float = 1.0, bias: bool = True,
                 tau: float = 1.0, credit_to_logit: str = "bernoulli"):
        super().__init__()
        if n_bits <= 0 or channels % n_bits:
            raise ValueError("n_bits must be positive and divide channels")
        self.channels = int(channels)
        self.n_bits = int(n_bits)
        self.groups = channels // n_bits
        self.tau = float(tau)
        self.credit_to_logit = str(credit_to_logit)
        self.q_proj = nn.Linear(channels, channels, bias=bias)
        self.k_proj = nn.Linear(channels, channels, bias=bias)
        self.v_proj = nn.Linear(channels, channels, bias=bias)
        self.o_proj = nn.Linear(channels, channels, bias=bias)
        self.emb0 = nn.Parameter(torch.zeros(1, 1, channels))
        self.emb1 = nn.Parameter(torch.full((1, 1, channels), float(init_emb1)))

    def _hard_rosa_from_qkv(self, q, k, v):
        return _HardQKVRosaCppFn.apply(q, k, v, self.emb0, self.emb1,
                                            self.n_bits, self.tau, self.credit_to_logit)

    def forward(self, x):
        if x.ndim != 3 or x.size(-1) != self.channels:
            raise ValueError("x must have shape [B,T,C]")
        return self.o_proj(self._hard_rosa_from_qkv(self.q_proj(x), self.k_proj(x), self.v_proj(x)))


NBitQKVRosaForwardCpp = NBitQKVRosaCpp
__all__ = ["NBitQKVRosaCpp", "NBitQKVRosaForwardCpp"]
