"""Budgeted joint-bit proposals with exact, conditional block credits.

Randomness chooses which structural candidates to inspect. Full uint64 symbols
are compared, and every selected two-bit edit is rerouted exactly. This augments the
single-bit credit engine with optional conditional blocks.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import operator
import time
from collections.abc import Collection, Mapping, Sequence

import numpy as np


STREAMS = ("Q", "K", "V")
MASK64 = (1 << 64) - 1


@dataclass(frozen=True, order=True)
class Bit:
    stream: str
    position: int
    bit: int

    def index(self):
        return STREAMS.index(self.stream), self.position, self.bit


@dataclass(frozen=True)
class PairCredit:
    left: Bit
    right: Bit
    joint: float
    interaction: float
    single_left: float
    single_right: float


def _words(values, name, d):
    try:
        out = [operator.index(x) for x in values]
    except TypeError as error:
        raise ValueError(f"{name} must contain integer packed symbols") from error
    if any(x < 0 or x >= 1 << d for x in out):
        raise ValueError(f"{name} must fit the declared D-bit unsigned alphabet")
    return out


def _bit(value, n, d):
    value = value if isinstance(value, Bit) else Bit(*value)
    if value.stream not in STREAMS:
        raise ValueError("bit stream must be Q, K, or V")
    if not 0 <= value.position < n or not 0 <= value.bit < d:
        raise ValueError("bit index is outside the stream")
    return value


def _pair(value, n, d):
    if len(value) != 2:
        raise ValueError("a pair must have exactly two bits")
    left, right = sorted((_bit(value[0], n, d), _bit(value[1], n, d)))
    if left == right:
        raise ValueError("a joint edit requires two distinct bits")
    return left, right


def _mix64(x):
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def _priority(pair, seed, n, d):
    result = seed & MASK64
    for b in pair:
        key = ((STREAMS.index(b.stream) * n + b.position) * d + b.bit)
        result = _mix64(result ^ _mix64(key))
    return result


def propose_pairs(q, k, D, *, max_pairs=16, seed=0, rows=None,
                  allowed_streams=("Q", "K"), allowed_owners=None,
                  bucket_cap=8, occurrence_cap=2, scan=8, candidate_cap=None,
                  candidate_trace=False):
    """Budgeted exact-verified repairs from static causal band sampling.

    The single proposal implementation lives in causal_candidates. Sampling
    can omit candidates; full words and every selected joint edit are exact.
    This replaces the initial per-token Python reservoir prototype.
    """
    if __package__:
        from .causal_candidates import propose_pairs as sample
    else:
        from causal_candidates import propose_pairs as sample
    return sample(q, k, D, max_pairs=max_pairs, seed=seed, rows=rows,
                  allowed_streams=allowed_streams, allowed_owners=allowed_owners,
                  bucket_cap=bucket_cap, occurrence_cap=occurrence_cap,
                  scan=scan, candidate_cap=candidate_cap,
                  candidate_trace=candidate_trace)


def _real(value, name, shape):
    value = np.asarray(value)
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    out = np.asarray(value, dtype=np.float64)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(f"{name} requires finite values with shape {shape}")
    return out


def bits_from_streams(q, k, v):
    v = np.asarray(v)
    if v.ndim != 2 or not 1 <= v.shape[1] <= 64 or np.any((v != 0) & (v != 1)):
        raise ValueError("V must have binary shape [T,D], D in [1,64]")
    n, d = v.shape
    q, k = _words(q, "Q", d), _words(k, "K", d)
    if len(q) != n or len(k) != n:
        raise ValueError("Q/K/V lengths differ")
    out = np.empty((3, n, d), dtype=np.uint8)
    shifts = np.arange(d, dtype=np.uint64)
    for stream, words in enumerate((q, k)):
        # Only the low bit of each shifted uint64 is needed. Writing its low
        # byte directly avoids an additional [T,D] uint64 temporary, while the
        # unsigned input preserves bit 63 and every D=64 packed symbol.
        np.right_shift(np.asarray(words, dtype=np.uint64)[:, None], shifts,
                       out=out[stream], casting="unsafe")
        out[stream] &= 1
    out[2] = v
    return out


def forward_payload(q, k, v, e0, e1, *, matcher=None):
    """Materialize exact hard outputs; matcher returns (lengths, routes)."""
    if matcher is None:
        if __package__:
            from .rosa_core import matching_stats
        else:
            from rosa_core import matching_stats
        matcher = matching_stats
    v = np.asarray(v, dtype=np.int8)
    _, routes = matcher(q, k)
    routes = np.asarray(routes, dtype=np.int64)
    n, d = v.shape
    if routes.shape != (n,) or np.any(routes < -1) or np.any(routes >= np.arange(n)):
        raise ValueError("matcher returned a noncausal route")
    y = np.broadcast_to(e0, (n, d)).copy()
    active = routes >= 0
    y[active] = (2 * v[routes[active] + 1] - 1) * e1
    return y, routes


def evaluate_pairs(q, k, v, g, h, e0, e1, pairs, *, single_credits=None,
                   baseline_y=None, matcher=None):
    """Evaluate every selected joint edit exactly under one frozen quadratic.

    Q(g,h;delta)=g^T delta+0.5*sum(h*delta^2). With complete singleton credits
    supplied, only one baseline and one joint forward per selected pair are
    needed. Without them, the two singleton corners are computed with a cache.
    Curvature must be nonnegative here. Neither targets nor true-loss reruns
    enter proposal selection or this frozen-output-model calculation.

    ``baseline_y`` is an advanced cache interface: the caller guarantees that it
    is exactly the hard output for these Q/K/V and embeddings. Its shape and
    finiteness are checked, but semantic equality cannot be checked without the
    forward we are deliberately saving. Never reuse it after changing any bit
    or either embedding. g/h must describe the intended model at that baseline.
    """
    started = time.perf_counter()
    v = np.asarray(v)
    if v.ndim != 2 or not 1 <= v.shape[1] <= 64 or np.any((v != 0) & (v != 1)):
        raise ValueError("V must have binary shape [T,D], D in [1,64]")
    n, d = v.shape
    # Retain packed NumPy storage throughout all corner replays. Passing a
    # copied Python list to the matcher would validate/repack every word for
    # every corner, although the input was already checked here.
    if __package__:
        from .rosa_core import _words as packed_words, matching_stats
    else:
        from rosa_core import _words as packed_words, matching_stats
    q, k = packed_words(q, "Q"), packed_words(k, "K")
    if q.shape != (n,) or k.shape != (n,):
        raise ValueError("Q/K/V lengths differ")
    if d < 64 and (np.any(q >> np.uint64(d)) or np.any(k >> np.uint64(d))):
        raise ValueError("Q/K must fit the declared D-bit unsigned alphabet")
    v = np.asarray(v, dtype=np.uint8)
    g = _real(g, "g", (n, d))
    h = None if h is None else _real(h, "h", (n, d))
    e0, e1 = _real(e0, "e0", (d,)), _real(e1, "e1", (d,))
    if h is not None and np.any(h < 0):
        raise ValueError("this block model requires nonnegative diagonal curvature")
    if h is not None and not np.any(h):
        h = None
    pairs = [_pair(pair, n, d) for pair in pairs]
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate pair")
    singles = None if single_credits is None else _real(single_credits, "single_credits", (3, n, d))
    stats = dict(baseline_forwards=0, singleton_forwards=0, joint_forwards=0,
                 evaluated_pairs=len(pairs), baseline_reused=baseline_y is not None,
                 baseline_seconds=0.0, singleton_evaluation_seconds=0.0,
                 joint_evaluation_seconds=0.0)
    if not pairs:
        stats["elapsed_seconds"] = time.perf_counter() - started
        return [], stats
    if baseline_y is None:
        baseline_started = time.perf_counter()
        y, _ = forward_payload(q, k, v, e0, e1, matcher=matcher)
        stats["baseline_forwards"] += 1
        stats["baseline_seconds"] = time.perf_counter() - baseline_started
    else:
        y = _real(baseline_y, "baseline_y", (n, d))
    single_cache = {}
    route_matcher = matching_stats if matcher is None else matcher
    # Payload values do not change for a Q/K edit. Cache this conversion once
    # instead of materializing signed V bits for each exact route replay.
    payload = (2 * v.astype(np.int8) - 1) * e1
    times = np.arange(n)

    def credit(edits):
        a, b = q.copy(), k.copy()
        changed_payload = payload.copy() if any(bit.stream == "V" for bit in edits) else payload
        for bit in edits:
            if bit.stream == "Q":
                a[bit.position] ^= np.uint64(1 << bit.bit)
            elif bit.stream == "K":
                b[bit.position] ^= np.uint64(1 << bit.bit)
            else:
                changed_payload[bit.position, bit.bit] = -changed_payload[bit.position, bit.bit]
        _, rr = route_matcher(a, b)
        rr = np.asarray(rr, dtype=np.int64)
        if rr.shape != (n,) or np.any(rr < -1) or np.any(rr >= times):
            raise ValueError("matcher returned a noncausal route")
        yp = np.broadcast_to(e0, (n, d)).copy()
        active = rr >= 0
        yp[active] = changed_payload[rr[active] + 1]
        delta = yp - y
        if h is None:
            return float(np.sum(g * delta))
        return float(np.sum(g * delta + 0.5 * h * delta * delta))

    def single(bit):
        if singles is not None:
            return float(singles[bit.index()])
        if bit not in single_cache:
            single_started = time.perf_counter()
            single_cache[bit] = credit([bit])
            stats["singleton_forwards"] += 1
            stats["singleton_evaluation_seconds"] += time.perf_counter() - single_started
        return single_cache[bit]

    result = []
    for left, right in pairs:
        ci, cj = single(left), single(right)
        joint_started = time.perf_counter()
        joint = credit((left, right))
        stats["joint_forwards"] += 1
        stats["joint_evaluation_seconds"] += time.perf_counter() - joint_started
        result.append(PairCredit(left, right, joint, joint - ci - cj, ci, cj))
    stats["elapsed_seconds"] = time.perf_counter() - started
    return result, stats


def _block_arrays(logits, bits, single_credits, tau):
    logits = np.asarray(logits)
    if logits.ndim != 3 or logits.shape[0] != 3 or not 1 <= logits.shape[2] <= 64:
        raise ValueError("logits must have shape [3,T,D]")
    logits = _real(logits, "logits", logits.shape)
    bits = np.asarray(bits)
    if bits.shape != logits.shape or np.any((bits != 0) & (bits != 1)):
        raise ValueError("bits must have the same binary shape as logits")
    singles = _real(single_credits, "single_credits", logits.shape)
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    exponential = np.exp(-np.abs(logits / tau))
    p = np.where(logits >= 0, 1 / (1 + exponential), exponential / (1 + exponential))
    flip = np.where(bits != 0, 1 - p, p)
    return logits, bits, singles, p, flip, tau


def _validated_blocks(evaluations, n, d):
    used = set()
    for record in evaluations:
        pair = _pair((record.left, record.right), n, d)
        if any(bit in used for bit in pair):
            raise ValueError("conditional pair blocks must be disjoint")
        used.update(pair)
        if not math.isfinite(record.interaction):
            raise ValueError("interaction must be finite")
        yield pair, record.interaction


def block_logit_gradient(logits, bits, single_credits, evaluations, *, tau=1.0):
    """Exact gradient of the selected conditional block interpolation.

    With flip probabilities alpha_i, the frozen model is
    sum_i alpha_i*C_i + sum_(i,j) alpha_i*alpha_j*(C_ij-C_i-C_j).
    Other bits are held at their current hard values when each corner is
    evaluated. This is not the gradient of a full joint Bernoulli network.
    Selection is held fixed when differentiating; no gradient is claimed for
    changing the candidate set or crossing a hard-bit boundary.
    """
    logits, bits, singles, p, flip, tau = _block_arrays(logits, bits, single_credits, tau)
    corrected = singles.copy()
    _, n, d = bits.shape
    for (left, right), interaction in _validated_blocks(evaluations, n, d):
        i, j = left.index(), right.index()
        corrected[i] += flip[j] * interaction
        corrected[j] += flip[i] * interaction
    return (1 - 2 * bits.astype(np.float64)) * p * (1 - p) * corrected / tau


def block_objective(logits, bits, single_credits, evaluations, *, tau=1.0):
    """The same frozen surrogate, useful for finite-difference verification."""
    _, bits, singles, _, flip, _ = _block_arrays(logits, bits, single_credits, tau)
    value = float(np.sum(flip * singles))
    _, n, d = bits.shape
    for (left, right), interaction in _validated_blocks(evaluations, n, d):
        value += float(flip[left.index()] * flip[right.index()] * interaction)
    return value


def correlated_block_objective(logits, bits, single_credits, evaluations, *, tau=1.0):
    """Lower convex envelope of each frozen, disjoint four-corner block.

    For flip marginals a,b the joint-flip probability rho lies in
    [max(0,a+b-1), min(a,b)]. Minimize a*C_i+b*C_j+rho*interaction
    over that interval: use its upper endpoint for a negative interaction and
    its lower endpoint for a positive one. This optimizes the coupling rather
    than imposing independent flips. All four already-evaluated corner costs
    are reused; this function performs no forwards or random sampling.

    The hard baseline, selected blocks, and output quadratic model are frozen.
    This is not an unbiased full-network stochastic-loss objective.
    """
    _, bits, singles, _, flip, _ = _block_arrays(logits, bits, single_credits, tau)
    value = float(np.sum(flip * singles))
    _, n, d = bits.shape
    for (left, right), interaction in _validated_blocks(evaluations, n, d):
        a, b = float(flip[left.index()]), float(flip[right.index()])
        rho = min(a, b) if interaction < 0 else max(0.0, a + b - 1.0)
        value += rho * interaction
    return value


def correlated_block_logit_gradient(logits, bits, single_credits, evaluations, *, tau=1.0):
    """Gradient, or a deterministic minimum-norm block subgradient at a kink.

    For a negative interaction, min(a,b) assigns its derivative to the smaller
    marginal. At a=b the derivative of that term can be split with any weight
    theta in [0,1]; choose theta minimizing the squared norm of the COMPLETE
    two marginal derivatives, including C_i and C_j. For a positive interaction
    at a+b=1, similarly minimize over the shared weight lambda in [0,1].

    Ties use exact floating-point equality, not a tolerance that would alter a
    differentiable branch. At either kink the two sigmoid derivative magnitudes
    agree, so this also minimizes the norm of the block's logit subgradient.
    The returned vector applies the ordinary sigmoid chain rule exactly once.
    Candidate selection and hard-baseline changes are not differentiated.
    """
    logits, bits, singles, p, flip, tau = _block_arrays(logits, bits, single_credits, tau)
    corrected = singles.copy()
    _, n, d = bits.shape
    for (left, right), interaction in _validated_blocks(evaluations, n, d):
        i, j = left.index(), right.index()
        a, b = float(flip[i]), float(flip[j])
        if interaction < 0:
            if a < b:
                corrected[i] += interaction
            elif a > b:
                corrected[j] += interaction
            else:
                theta = float(np.clip(
                    (singles[j] + interaction - singles[i]) / (2.0 * interaction), 0.0, 1.0))
                corrected[i] += theta * interaction
                corrected[j] += (1.0 - theta) * interaction
        elif interaction > 0:
            if a + b > 1.0:
                corrected[i] += interaction
                corrected[j] += interaction
            elif a + b == 1.0:
                weight = float(np.clip(
                    -(singles[i] + singles[j]) / (2.0 * interaction), 0.0, 1.0))
                corrected[i] += weight * interaction
                corrected[j] += weight * interaction
    return (1 - 2 * bits.astype(np.float64)) * p * (1 - p) * corrected / tau


def _pair_arrays(pair_logits, pair_bits, evaluations, tau):
    """Validate only the selected 2J coordinates; no full-tensor scans."""
    records = list(evaluations)
    shape = (len(records), 2)
    logits = _real(pair_logits, "pair_logits", shape)
    bits = np.asarray(pair_bits)
    if bits.shape != shape or np.any((bits != 0) & (bits != 1)):
        raise ValueError("pair_bits must have binary shape [J,2]")
    singles = np.empty(shape, dtype=np.float64)
    interactions = np.empty(len(records), dtype=np.float64)
    canonical_order = []
    used = set()
    for row, record in enumerate(records):
        left, right = record.left, record.right
        if left == right or left in used or right in used:
            raise ValueError("conditional pair blocks must be disjoint")
        for bit in (left, right):
            if bit.stream not in STREAMS or bit.position < 0 or not 0 <= bit.bit < 64:
                raise ValueError("invalid bit in pair evaluation")
        used.update((left, right))
        canonical_order.append((0, 1) if left < right else (1, 0))
        singles[row] = record.single_left, record.single_right
        interactions[row] = record.interaction
    if not np.isfinite(singles).all() or not np.isfinite(interactions).all():
        raise ValueError("pair single credits and interactions must be finite")
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    exponential = np.exp(-np.abs(logits / tau))
    p = np.where(logits >= 0, 1 / (1 + exponential), exponential / (1 + exponential))
    flip = np.where(bits != 0, 1 - p, p)
    return bits, singles, interactions, p, flip, tau, canonical_order


def pair_logit_gradient(pair_logits, pair_bits, evaluations, *, tau=1.0):
    """Independent block gradients for only the selected 2J coordinates.

    Inputs/outputs have shape [J,2], ordered as each evaluation's left/right.
    Start from the caller's existing singleton logit gradient, then REPLACE
    those coordinates with this result. Returning complete replacement values
    preserves the full reference API's arithmetic order, including exact zeros.

    Cache contract: evaluations must come from the same bits, logits' current
    hard baseline, and quadratic model as the caller's singleton gradient.
    Its single_left/right fields are authoritative here; unlike the full API,
    this function does not receive or scan a separate [3,T,D] singleton tensor.
    The caller gathers matching logits/bits and is responsible for tensor bounds.
    Pair blocks must be disjoint. Work and extra memory are O(J).
    """
    bits, singles, interactions, p, flip, tau, _ = _pair_arrays(
        pair_logits, pair_bits, evaluations, tau)
    corrected = singles.copy()
    for row, interaction in enumerate(interactions):
        corrected[row, 0] += flip[row, 1] * interaction
        corrected[row, 1] += flip[row, 0] * interaction
    return (1 - 2 * bits.astype(np.float64)) * p * (1 - p) * corrected / tau


def correlated_pair_logit_gradient(pair_logits, pair_bits, evaluations, *, tau=1.0):
    """Optimal-coupling replacements on 2J coordinates, using O(J) work.

    Same [J,2] ordering and cache contract as ``pair_logit_gradient``. The
    result replaces selected coordinates in an existing singleton gradient;
    it is not an increment. The full ``correlated_block_logit_gradient`` API
    remains the reference. Both use the same minimum-norm kink choice and
    arithmetic order, without recomputing full-tensor sigmoids or unpacking bits.
    """
    bits, singles, interactions, p, flip, tau, canonical_order = _pair_arrays(
        pair_logits, pair_bits, evaluations, tau)
    corrected = singles.copy()
    for row, interaction in enumerate(interactions):
        i, j = canonical_order[row]
        a, b = float(flip[row, i]), float(flip[row, j])
        if interaction < 0:
            if a < b:
                corrected[row, i] += interaction
            elif a > b:
                corrected[row, j] += interaction
            else:
                theta = float(np.clip(
                    (singles[row, j] + interaction - singles[row, i]) / (2.0 * interaction), 0.0, 1.0))
                corrected[row, i] += theta * interaction
                corrected[row, j] += (1.0 - theta) * interaction
        elif interaction > 0:
            if a + b > 1.0:
                corrected[row, i] += interaction
                corrected[row, j] += interaction
            elif a + b == 1.0:
                weight = float(np.clip(
                    -(singles[row, i] + singles[row, j]) / (2.0 * interaction), 0.0, 1.0))
                corrected[row, i] += weight * interaction
                corrected[row, j] += weight * interaction
    return (1 - 2 * bits.astype(np.float64)) * p * (1 - p) * corrected / tau
