"""Batched hard QKV-ROSA with bounded joint-bit training credits.

The suffix matcher and credit engine are the single CPU implementation in
``rosa_core``. Projections, payload selection, and final tensor gradients run
on the input device; structural credits and selected-block arithmetic run on
CPU. CUDA tensors therefore incur transfers and synchronization for routing
and credits; this is not a GPU kernel.

The Q/K/V/O projection and embedding parameter names preserve existing state
dicts. All historical layer class names below alias this same implementation.
"""
from __future__ import annotations

import math
import operator

import numpy as np
import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

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


def _joint_rows(length, count, gradient=None, seed=0):
    if gradient is not None:
        active = np.flatnonzero(np.any(gradient != 0, axis=-1))
        if len(active) <= count:
            return active.tolist()
        # Weighted sampling without replacement over exactly nonzero rows.
        # Normalize once for stable norms, and use exponential-race keys.
        # Positive rescaling of the upstream loss preserves this distribution
        # and, up to roundoff, this fixed-seed realization.
        rows = gradient[active]
        scale = np.max(np.abs(rows))
        norm = np.sqrt(np.sum((rows / scale)**2, axis=-1))
        rng = np.random.default_rng(seed)
        keys = np.log1p(-rng.random(len(active))) / norm
        selected = np.argpartition(keys, -count)[-count:]
        return sorted(active[selected].tolist())
    if length <= count:
        return list(range(length))
    # Only inputs/positions choose proposals; losses and incoming gradients do
    # not influence the frozen surrogate Jacobian.
    return np.linspace(0, length - 1, count, dtype=np.intp).tolist()


def _payload_radius(emb0, emb1):
    """Half the diameter of {emb0} union the signed-payload hypercube.

    The farthest signed pair has length 2*||emb1||. The farthest vertex
    from emb0 has coordinatewise distance |emb0|+|emb1|. Including the
    unmatched value prevents a distant learned emb0 from understating the
    maximum finite jump. The small floor only protects the all-zero set.
    """
    signed_radius = np.linalg.norm(emb1, axis=-1)
    empty_radius = .5*np.linalg.norm(np.abs(emb0)+np.abs(emb1), axis=-1)
    return np.maximum(np.maximum(signed_radius, empty_radius), 1e-12)


class _HardQKVRosaFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, emb0, emb1, n_bits, tau, credit_to_logit,
                repair_samples, joint_pairs, joint_rows, sample_values, trust_diagonal,
                joint_row_selection):
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
        if sample_values:
            # Sampling changes only training payload bits. Routing remains
            # hard Q/K, and each realized payload/credit is evaluated exactly.
            # FP32 probabilities avoid bfloat16's particularly coarse RNG grid.
            probability_dtype = torch.float64 if v.dtype == torch.float64 else torch.float32
            probability = torch.sigmoid(v.to(probability_dtype) / tau)
            vb = (torch.rand_like(probability) < probability).to(torch.uint8).reshape(B, T, G, D)
        else:
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
        ctx.repair_samples = repair_samples
        ctx.trust_diagonal = float(trust_diagonal)
        ctx.joint_pairs, ctx.joint_rows = int(joint_pairs), int(joint_rows)
        ctx.joint_row_selection = str(joint_row_selection)
        # Freeze independent proposal/repair RNG streams in this forward.
        # Conditional on forced pair lanes, repair selection still needs its
        # own random draw for the inverse-probability estimator. Repeated
        # backward calls through the same graph reuse both realizations.
        randomized = (repair_samples is not None and repair_samples < D) or joint_pairs > 0
        ctx.seed, ctx.repair_seed = torch.randint(0, 2**63 - 1, (2,), device="cpu").tolist() if randomized else (0, 0)
        ctx.q_all, ctx.k_all = q_all, k_all
        ctx.save_for_backward(q, k, v, qb, kb, vb, route, emb0, emb1)
        return y.reshape(B, T, C)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y):
        q, k, v, qb, kb, vb, route, emb0, emb1 = ctx.saved_tensors
        B, T, C = q.shape
        D, G = ctx.n_bits, ctx.groups
        gy4 = grad_y.reshape(B, T, G, D)
        vbc = vb.detach().cpu()
        # The core uses binary64 accumulation. Convert before NumPy access,
        # including for bfloat16/autocast inputs whose .numpy() is unsupported.
        gyc = gy4.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(gyc).all()):
            # A regular GradScaler must be able to detect an overflow and skip
            # the optimizer step. The strict numerical core rejects nonfinite
            # inputs, so propagate a nonfinite backward result at the tensor
            # boundary instead of raising during the scaled backward pass.
            gradients = tuple(torch.full_like(z, float("nan"))
                              for z in (q, k, v, emb0, emb1))
            return gradients + (None,) * 9
        e0c = emb0.detach().reshape(G, D).to(device="cpu", dtype=torch.float64)
        e1c = emb1.detach().reshape(G, D).to(device="cpu", dtype=torch.float64)
        radii = _payload_radius(e0c.numpy(), e1c.numpy()) if ctx.trust_diagonal else None
        qfc = torch.zeros((B, T, G, D), dtype=torch.float64)
        kfc, vfc = torch.zeros_like(qfc), torch.zeros_like(qfc)
        replacements = []
        for b in range(B):
            for g in range(G):
                qs, ks = ctx.q_all[b][g], ctx.k_all[b][g]
                seed = (ctx.seed + 0x9e3779b97f4a7c15 * (b * G + g + 1)) % (2**64)
                repair_seed = (ctx.repair_seed + 0x9e3779b97f4a7c15 * (b * G + g + 1)) % (2**64)
                pairs = []
                if ctx.joint_pairs and T:
                    if __package__:
                        from .joint_credit import propose_pairs
                    else:
                        from joint_credit import propose_pairs
                    row_gradient = gyc[b, :, g, :].numpy() if ctx.joint_row_selection == "active" else None
                    pairs, _ = propose_pairs(qs, ks, D, max_pairs=ctx.joint_pairs,
                        rows=_joint_rows(T, ctx.joint_rows, row_gradient, seed), seed=seed)
                args = (qs, ks, vbc[b, :, g, :], gyc[b, :, g, :], e0c[g], e1c[g])
                diagonal = None
                if ctx.trust_diagonal:
                    # A homogeneous local trust metric, not an estimate of
                    # the loss Hessian. It bounds large route-output jumps
                    # relative to incoming gradient and payload scale.
                    radius = float(radii[g])
                    row_norm = np.linalg.norm(gyc[b, :, g, :].numpy(), axis=-1, keepdims=True)
                    diagonal = np.broadcast_to(ctx.trust_diagonal * row_norm / radius, (T, D)).copy()
                if ctx.repair_samples is not None and ctx.repair_samples < D:
                    forced = [bit.bit for pair in pairs for bit in pair]
                    result = rosa_core.adaptive_repair_sampled_credits_numpy(
                        *args, ctx.repair_samples, repair_seed,
                        forced_bits=sorted(set(forced)), D=D, diag_y=diagonal)
                elif diagonal is not None:
                    result = rosa_core.quadratic_credits_numpy(
                        *args[:4], diagonal, *args[4:], D)
                else:
                    result = rosa_core.credits_numpy(*args, D)
                qa, ka, va = result[:3]
                if diagonal is not None:
                    # The trust metric protects route changes, while V keeps
                    # the ordinary linear estimator. Its exact route scatter
                    # is O(TD), without a second structural-credit traversal.
                    rr = np.asarray(result[4], dtype=np.intp)
                    active = rr >= 0
                    va = np.zeros((T, D), dtype=np.float64)
                    source = rr[active] + 1
                    change = (1 - 2*vbc[b, source, g, :].numpy().astype(float)) * (2*e1c[g].numpy())
                    np.add.at(va, source, gyc[b, active, g, :].numpy() * change)
                qfc[b, :, g, :] = torch.from_numpy(qa)
                kfc[b, :, g, :] = torch.from_numpy(ka)
                vfc[b, :, g, :] = torch.from_numpy(va)
                if pairs:
                    if __package__:
                        from .joint_credit import evaluate_pairs, pair_logit_gradient
                    else:
                        from joint_credit import evaluate_pairs, pair_logit_gradient
                    rr = route[b, :, g].detach().cpu().numpy()
                    baseline = np.broadcast_to(e0c[g].numpy(), (T, D)).copy()
                    active = rr >= 0
                    baseline[active] = (2 * vbc[b, rr[active] + 1, g, :].numpy().astype(float) - 1) * e1c[g].numpy()
                    evaluations, _ = evaluate_pairs(*args[:4], diagonal,
                        e0c[g], e1c[g], pairs, single_credits=(qa, ka, va), baseline_y=baseline)
                    locations = [bit for record in evaluations for bit in (record.left, record.right)]
                    selected_bits = np.array([((qs if bit.stream == "Q" else ks)[bit.position] >> bit.bit) & 1
                        for bit in locations], dtype=np.uint8).reshape(-1, 2)
                    replacements.append((b, g, locations, selected_bits, evaluations))
        grad_q, grad_k, grad_v = (
            _flip_credit_to_logit_grad(
                z, bits.reshape(B, T, C), credit.reshape(B, T, C),
                ctx.tau, ctx.credit_to_logit,
            )
            for z, bits, credit in ((q, qb, qfc), (k, kb, kfc), (v, vb, vfc))
        )
        gradients = {"Q": grad_q, "K": grad_k, "V": grad_v}
        if replacements:
            # Gather only the selected 2J coordinates per batch/group. All
            # groups share one device->CPU logit transfer, then one batched
            # scatter per stream. No complete Q/K logit copies are needed.
            coordinates = {stream: [[], [], [], []] for stream in ("Q", "K")}
            count = 0
            for b, group, locations, _, _ in replacements:
                for bit in locations:
                    batches, positions, columns, destinations = coordinates[bit.stream]
                    batches.append(b); positions.append(bit.position)
                    columns.append(group*D+bit.bit); destinations.append(count)
                    count += 1
            chunks = []
            tensors = {"Q": q, "K": k}
            for stream, (batches, positions, columns, _) in coordinates.items():
                if batches:
                    chunks.append(tensors[stream][batches, positions, columns].detach())
            joined = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
            packed_logits = joined.to(device="cpu", dtype=torch.float64).numpy()
            logits = np.empty(count, dtype=np.float64)
            offset = 0
            for _, _, _, destinations in coordinates.values():
                size = len(destinations)
                logits[destinations] = packed_logits[offset:offset+size]
                offset += size
            values = np.empty(count, dtype=np.float64)
            offset = 0
            for _, _, locations, selected_bits, evaluations in replacements:
                size = len(locations)
                selected_logits = logits[offset:offset+size].reshape(-1, 2)
                result = pair_logit_gradient(selected_logits, selected_bits,
                                             evaluations, tau=ctx.tau)
                if ctx.credit_to_logit == "identity":
                    zz = selected_logits / ctx.tau
                    p = 1 / (1 + np.exp(-np.clip(zz, -700, 700)))
                    orient = 1 - 2 * selected_bits.astype(float)
                    flip = np.where(selected_bits != 0, 1 - p, p)
                    result = np.array([[record.single_left + flip[i, 1] * record.interaction,
                                        record.single_right + flip[i, 0] * record.interaction]
                                       for i, record in enumerate(evaluations)]) * orient
                values[offset:offset+size] = result.reshape(-1)
                offset += size
            for stream, (batches, positions, columns, destinations) in coordinates.items():
                if batches:
                    dst = gradients[stream]
                    dst[batches, positions, columns] = torch.as_tensor(
                        values[destinations], dtype=dst.dtype, device=dst.device)
        matched = (route >= 0)[..., None]
        indices = (route + 1).clamp_min(0)[..., None].expand(B, T, G, D)
        selected = torch.gather(vb, dim=1, index=indices)
        sign = selected.to(gy4.dtype) * 2.0 - 1.0
        # These are ordinary derivatives of the selected embedding values.
        ge0 = torch.where(~matched, gy4, torch.zeros_like(gy4)).sum(dim=(0, 1))
        ge1 = torch.where(matched, gy4 * sign, torch.zeros_like(gy4)).sum(dim=(0, 1))
        ge0 = ge0.reshape(1, 1, C).to(dtype=emb0.dtype, device=emb0.device)
        ge1 = ge1.reshape(1, 1, C).to(dtype=emb1.dtype, device=emb1.device)
        return grad_q, grad_k, grad_v, ge0, ge1, None, None, None, None, None, None, None, None, None


class NBitQKVRosa(nn.Module):
    """Hard grouped QKV-ROSA with one shared CPU credit engine.

    ``x`` has shape [B,T,C]. Each contiguous ``n_bits`` channels form one
    symbol; bit j has unsigned weight 2**j, including j=63. Each group uses
    strictly causal K endpoints r<t, with longest match then latest endpoint.

    Default training uses at most one independent joint-bit block per stream,
    selected at active-gradient rows, and a diameter-scaled local trust metric
    with coefficient 0.5. This is a finite-output surrogate, not a recovered
    downstream Hessian. Q/K/V forward quantization remains z>0; V credits and
    repair enumeration remain exact. A selected block adds one internal route
    replay, while callers still use one model forward and loss.backward().

    ``joint_pairs=0, trust_diagonal=0`` is the original singleton ablation in
    the same engine. ``repair_samples`` optionally samples certified repair
    residuals; it is disabled by default. ``credit_to_logit`` is ``bernoulli``
    (sigmoid derivative) or ``identity``. Experimental stochastic V and local
    output mixtures are opt-in and are not the selected training recipe.
    """

    def __init__(self, channels: int, n_bits: int, init_emb1: float = 1.0,
                 bias: bool = True, tau: float = 1.0,
                 credit_to_logit: str = "bernoulli", *,
                 repair_samples: int | None = None, joint_pairs: int = 1,
                 joint_rows: int = 32, stochastic_values: bool = False,
                 trust_diagonal: float = 0.5,
                 training_rule: str = "hard_credit",
                 joint_row_selection: str = "active"):
        super().__init__()
        channels, n_bits = operator.index(channels), operator.index(n_bits)
        if channels <= 0 or not 1 <= n_bits <= 64 or channels % n_bits:
            raise ValueError("n_bits must be in [1,64] and divide positive channels")
        tau = float(tau)
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and positive")
        if credit_to_logit not in ("bernoulli", "identity"):
            raise ValueError(f"unknown credit_to_logit mode: {credit_to_logit}")
        if repair_samples is not None:
            repair_samples = operator.index(repair_samples)
            if not 1 <= repair_samples <= n_bits:
                raise ValueError("repair_samples must lie in [1,n_bits], or be None")
        joint_pairs, joint_rows = operator.index(joint_pairs), operator.index(joint_rows)
        if joint_pairs < 0 or joint_rows <= 0:
            raise ValueError("joint_pairs must be nonnegative and joint_rows positive")
        if not isinstance(stochastic_values, bool):
            raise TypeError("stochastic_values must be bool")
        trust_diagonal = float(trust_diagonal)
        if not math.isfinite(trust_diagonal) or trust_diagonal < 0:
            raise ValueError("trust_diagonal must be finite and nonnegative")
        if training_rule not in ("hard_credit", "local_mixture"):
            raise ValueError("training_rule must be hard_credit or local_mixture")
        if training_rule == "local_mixture" and joint_pairs == 0:
            raise ValueError("local_mixture requires joint_pairs > 0")
        if joint_row_selection not in ("input", "active"):
            raise ValueError("joint_row_selection must be input or active")
        if training_rule == "local_mixture" and joint_row_selection != "input":
            raise ValueError("local_mixture selects candidates before an upstream gradient exists")
        self.channels, self.n_bits = channels, n_bits
        self.groups = channels // n_bits
        self.tau, self.credit_to_logit = tau, str(credit_to_logit)
        self.repair_samples = repair_samples
        self.joint_pairs, self.joint_rows = joint_pairs, joint_rows
        self.stochastic_values = stochastic_values
        self.trust_diagonal, self.training_rule = trust_diagonal, training_rule
        self.joint_row_selection = joint_row_selection
        self.last_forward_work = None
        self.q_proj = nn.Linear(channels, channels, bias=bias)
        self.k_proj = nn.Linear(channels, channels, bias=bias)
        self.v_proj = nn.Linear(channels, channels, bias=bias)
        self.o_proj = nn.Linear(channels, channels, bias=bias)
        self.emb0 = nn.Parameter(torch.zeros(1, 1, channels))
        self.emb1 = nn.Parameter(torch.full((1, 1, channels), float(init_emb1)))

    def _hard_rosa_from_qkv(self, q, k, v):
        if self.training and self.training_rule == "local_mixture":
            if __package__:
                from .local_mixture import local_mixture
            else:
                from local_mixture import local_mixture
            y, self.last_forward_work = local_mixture(
                q, k, v, self.emb0, self.emb1, self.n_bits, self.tau,
                max_pairs=self.joint_pairs, max_rows=self.joint_rows)
            return y
        self.last_forward_work = None
        return _HardQKVRosaFn.apply(
            q, k, v, self.emb0, self.emb1,
            self.n_bits, self.tau, self.credit_to_logit,
            self.repair_samples if self.training else None,
            self.joint_pairs if self.training else 0, self.joint_rows,
            self.training and self.stochastic_values,
            self.trust_diagonal if self.training else 0.0,
            self.joint_row_selection,
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
