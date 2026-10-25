"""One CPU implementation of the exact hard QKV-ROSA single-bit VJP.

``credits_numpy`` returns float64 credits and work counters. The five-result
``exact_stream_bit_credits`` interface preserves a PyTorch gradient's dtype.
The shared C++17 core also supplies the layer's causal forward matcher.
See ALGORITHM.md for the derivation and the limits of floating-point exactness.
"""
from __future__ import annotations

import ctypes
import hashlib
import operator
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "csrc"
_LIB = None
_LOAD_LOCK = threading.Lock()
_MAX_T = 1 << 28
_MAX_ELEMENTS = (1 << 31) - 1
_MAX_WORD = (1 << 64) - 1

COUNTER_NAMES = (
    "payload_classes", "visited_alignment_pairs", "one_mismatch_candidates",
    "touched_owner_bits", "payload_dot_components", "reported_occurrence_entries",
    "sam_active_advances", "deletion_ranges", "sam_transition_probes",
    "sam_suffix_fallbacks", "delete_a_records", "delete_r_records",
    "sam_endpoint_queries", "delete_a_expanded_records", "deletion_owner_ranges",
    "deletion_owner_covered_records", "materialized_index_flags",
    "singleton_corridors", "singleton_represented_rows", "singleton_run_steps",
    "singleton_k_key_rows", "singleton_dot_components", "singleton_heap_pops",
)


def build(verbose: bool = False) -> Path:
    """Build the versionless native core; cache by source and compiler options.

    Linux/macOS need a C++17 compiler. Windows needs an x64 Visual Studio
    developer shell (``cl``), or a compatible compiler supplied through CXX.
    No PyTorch C++ headers, CUDA toolkit or ninja installation is needed.
    """
    compiler = os.environ.get("CXX", "cl" if os.name == "nt" else "c++")
    command = shlex.split(compiler, posix=os.name != "nt")
    if os.name == "nt":
        command = [part[1:-1] if len(part) >= 2 and part[0] == part[-1] == '"' else part
                   for part in command]
    if not command or shutil.which(command[0]) is None:
        raise RuntimeError("A C++17 compiler is required. Set CXX or use a Visual Studio x64 developer shell.")
    is_msvc = Path(command[0]).name.lower() in {"cl", "cl.exe", "clang-cl", "clang-cl.exe"}
    flags = (["/nologo", "/O2", "/DNDEBUG", "/std:c++17", "/EHsc", "/fp:strict", "/LD"]
             if is_msvc else
             ["-O3", "-DNDEBUG", "-std=c++17", "-fPIC", "-shared", "-ffp-contract=off"])
    digest = hashlib.sha256()
    digest.update(repr((command, flags, sys.platform)).encode())
    for path in sorted(SOURCE.glob("*.h")) + [SOURCE / "unified_core.cpp"]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    suffix = ".dll" if os.name == "nt" else (".dylib" if sys.platform == "darwin" else ".so")
    cache = HERE / "build" / "exact"
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / ("rosa_exact_" + digest.hexdigest()[:16] + suffix)
    if output.exists():
        return output
    # Compile to a private directory and publish only a complete library.
    # Concurrent processes never observe a half-written shared object.
    with tempfile.TemporaryDirectory(prefix="compile-", dir=cache) as temporary:
        target = Path(temporary) / output.name
        args = command + flags + [str(SOURCE / "unified_core.cpp")]
        args += (["/Fe:" + str(target)] if is_msvc else ["-o", str(target)])
        if verbose:
            print("Building QKV-ROSA CPU core with", command[0])
        subprocess.run(args, cwd=temporary, check=True)
        if not output.exists():
            os.replace(target, output)
    return output


def _library():
    global _LIB
    if _LIB is None:
        with _LOAD_LOCK:
            if _LIB is None:
                library = ctypes.CDLL(str(build()))
                library.rosa_uniform.restype = ctypes.c_int
                library.rosa_uniform.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 12
                library.rosa_quadratic.restype = ctypes.c_int
                library.rosa_quadratic.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 13
                library.rosa_selected.restype = ctypes.c_int
                library.rosa_selected.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64] + [ctypes.c_void_p] * 16
                library.rosa_budgeted.restype = ctypes.c_int
                library.rosa_budgeted.argtypes = ([ctypes.c_int] * 3 + [ctypes.c_uint64] * 2
                                                  + [ctypes.c_void_p] * 16)
                library.rosa_pair_candidates.restype = ctypes.c_int
                library.rosa_pair_candidates.argtypes = (
                    [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 3 + [ctypes.c_int]
                    + [ctypes.c_void_p] * 2 + [ctypes.c_int, ctypes.c_uint64]
                    + [ctypes.c_int] * 4 + [ctypes.c_void_p] * 4)
                library.rosa_matching_stats.restype = ctypes.c_int
                library.rosa_matching_stats.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 4
                _LIB = library
    return _LIB


def _array(value, dtype=None):
    if hasattr(value, "detach"):
        if value.device.type != "cpu":
            raise ValueError("The stream core accepts CPU tensors; move inputs explicitly.")
        value = value.detach()
        # NumPy has no native bfloat16 dtype; conversion is lossless here.
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    raw = np.asarray(value)
    if dtype is not None and np.issubdtype(np.dtype(dtype), np.floating) and np.iscomplexobj(raw):
        raise ValueError("Gradients and embeddings must be real-valued")
    return np.ascontiguousarray(raw, dtype=dtype)


def _words(value, name):
    if isinstance(value, (list, tuple)):
        try:
            items = [operator.index(item) for item in value]
        except TypeError as exc:
            raise ValueError(f"{name} must contain integer packed words") from exc
        if any(item < 0 or item > _MAX_WORD for item in items):
            raise ValueError(f"{name} words must fit unsigned 64-bit integers")
        return np.asarray(items, dtype=np.uint64)
    raw = _array(value)
    if raw.dtype.kind not in "iub":
        raise ValueError(f"{name} must contain integer packed words")
    if raw.dtype.kind == "i" and np.any(raw < 0):
        raise ValueError(f"{name} words must be nonnegative")
    return np.ascontiguousarray(raw, dtype=np.uint64)


def _bits(value):
    raw = _array(value)
    if raw.ndim != 2 or not 1 <= raw.shape[1] <= 64:
        raise ValueError("bits must have shape [T,D] with 1 <= D <= 64")
    if np.any((raw != 0) & (raw != 1)):
        raise ValueError("bits must contain only 0/1")
    return np.ascontiguousarray(raw, dtype=np.uint8)


def pack_bits(bits):
    """Pack [T,D] bits, least-significant bit first, into Python unsigned ints."""
    values = _bits(bits).astype(np.uint64)
    shifts = np.arange(values.shape[1], dtype=np.uint64)
    return (values << shifts).sum(axis=1, dtype=np.uint64).tolist()


def matching_stats(q, k):
    """Return causal (lengths, routes) with longest/latest tie-breaking."""
    qn, kn = _words(q, "Q"), _words(k, "K")
    if qn.ndim != 1 or kn.shape != qn.shape or len(qn) >= _MAX_T:
        raise ValueError("Q/K must have equal shape [T], with T < 2**28")
    n = len(qn)
    lengths = np.zeros(n, dtype=np.int32)
    routes = np.full(n, -1, dtype=np.int32)
    arrays = (qn, kn, lengths, routes)
    code = _library().rosa_matching_stats(n, *(a.ctypes.data_as(ctypes.c_void_p) for a in arrays))
    if code:
        raise RuntimeError(f"Causal matcher failed with code {code}")
    return lengths.tolist(), routes.tolist()


def credits_numpy(q, k, v_bits, grad_y, emb0, emb1, D=None):
    """Return (cq, ck, cv, lengths, routes, counters), with float64 credits.

    Q/K are packed words. V and grad_y have shape [T,D]; embeddings have
    shape [D]. This is the finite single-bit counterfactual credit, not the
    ordinary derivative of threshold quantization. Inputs must be finite;
    IEEE nearest-even arithmetic and no internal overflow are required.
    """
    return _credits_numpy(q, k, v_bits, grad_y, emb0, emb1, D)


def quadratic_credits_numpy(q, k, v_bits, grad_y, diag_y, emb0, emb1, D=None):
    """Return finite-flip credits of a diagonal quadratic output-loss model.

    For each Q/K/V bit i, the returned credit is
    ``sum(grad_y * delta_i + 0.5 * diag_y * delta_i**2)``. ``delta_i`` is
    the complete hard ROSA output change after only that bit is flipped.
    ``diag_y`` has shape [T,D] and must be finite and nonnegative. It is a
    caller-supplied output-space metric, not a curvature estimate produced by
    this function. Cross-output curvature and multi-bit interactions are not
    included. The tuple shape matches ``credits_numpy``.

    Routing is identical to the exact first-order engine. The baseline output
    is obtained inside its existing row scan, so this API adds no forward pass.
    Two possible signed-payload costs per output cell require O(TD)
    preparation work. Extra working storage is O(D) unless deferred intervals
    require historical rows, when an O(TD) table is allocated. Baseline-relative
    costs avoid subtracting large common scores. A zero metric uses the
    ordinary numerical path.
    """
    return _credits_numpy(q, k, v_bits, grad_y, emb0, emb1, D, diag_y)


def selected_credits_numpy(q, k, v_bits, grad_y, emb0, emb1,
                           repair_bits, D=None, diag_y=None):
    """Compute exact deletion baselines and selected complete repair lanes.

    Returns the usual six credit results, followed by ``q_deletion[T]`` and
    ``k_deletion[T]``. Selected Q/K columns equal the full finite credits;
    unselected columns contain only their exact deletion baseline. All V
    credits and baseline routes remain exact. ``repair_bits`` is an iterable
    of distinct bit indices. An empty selection computes the control variate.

    A lane includes every alternative for every owner of that bit. In
    particular, this preserves K's maximum over competing repair routes.
    Selecting individual route alternatives instead would not preserve it.
    This function itself is deterministic and does not weight the results.
    """
    return _credits_numpy(q, k, v_bits, grad_y, emb0, emb1, D, diag_y,
                          repair_bits=repair_bits)


def repair_sampled_credits_numpy(q, k, v_bits, grad_y, emb0, emb1,
                                repair_bits, inclusion_probability, D=None,
                                diag_y=None):
    """Unbiased repair-residual estimator with exact deletion and V credits.

    The caller supplies a realized subset of complete bit lanes and their
    *conditional marginal inclusion probabilities*. A scalar gives every lane
    the same probability. A length-D vector supports forced lanes (p=1) and
    nonuniform sampling. Every probability must be in (0,1]; every p=1 lane
    must be selected. For a uniform r-of-D subset, pass ``r / D``.

    Writing the full Q/K credit as C=D+R, this returns D+I*R/p. The expectation
    equals C in exact arithmetic, conditional on fixed inputs and sampling
    design. This is a claim about finite-credit estimation, not an unbiased
    gradient of the complete discrete network. The caller must really sample
    with the stated probabilities; supplying arbitrary weights cannot ensure
    unbiasedness. Loss-scaled/diagonal credits follow the same algebra.

    Uniform sampling without replacement gives coordinate MSE
    (1-p)/p * R**2, as against (1-p)/p * C**2 for dropping entire credits.
    Deletion and V add no noise. Full inclusion is numerically identical to
    the exact selected engine and requires no extra forward.
    """
    result = selected_credits_numpy(q, k, v_bits, grad_y, emb0, emb1,
                                   repair_bits, D, diag_y)
    cq, ck, cv, lengths, routes, stats, qb, kb = result
    d = cq.shape[1]
    prob = _array(inclusion_probability, np.float64)
    if np.ndim(inclusion_probability) == 0:
        prob = np.full(d, float(prob.reshape(-1)[0]), dtype=np.float64)
    if prob.shape != (d,) or not np.isfinite(prob).all() or np.any(prob <= 0) or np.any(prob > 1):
        raise ValueError("inclusion_probability must be in (0,1], scalar or shape [D]")
    selected = stats["repair_bits"]
    present = np.zeros(d, dtype=bool)
    present[selected] = True
    if np.any((prob == 1) & ~present):
        raise ValueError("Every lane with inclusion probability 1 must be selected")
    for bit in selected:
        p = prob[bit]
        if p != 1:
            cq[:, bit] = qb + (cq[:, bit] - qb) / p
            ck[:, bit] = kb + (ck[:, bit] - kb) / p
    stats["repair_inclusion_probabilities"] = prob.tolist()
    stats["repair_estimator"] = "exact_deletion_plus_inverse_probability_repair"
    return cq, ck, cv, lengths, routes, stats


def adaptive_repair_sampled_credits_numpy(q, k, v_bits, grad_y, emb0, emb1,
                                         repair_count, seed, forced_bits=(),
                                         D=None, diag_y=None):
    """Sample only certified possible repair lanes, with exact forced lanes.

    The root dictionary certifies lane j inactive if no causal pair satisfies
    Q[s] xor K[r] == 2**j for r<s. Such a lane has identically zero repair
    residual. It is kept at its exact deletion baseline, without sampling.
    Among supported lanes, ``forced_bits`` are evaluated exactly; a uniform
    subset of the rest is evaluated and inverse-probability weighted.

    ``repair_count`` is the target total number of active lanes. If forced
    active lanes consume that target, one additional optional lane is still
    sampled to keep every nonzero residual's probability positive. Actual
    selected lanes/probabilities and the effective budget are returned in
    counters. No more than D lanes can run. A budget >= the support size
    returns exact credits. The seed is an unsigned 64-bit sampling seed;
    dictionary placement uses an independent seed and cannot alter answers.

    The same native call builds the dictionary once, certifies support, and
    runs selected repairs. It does not call an extra forward. All root joins
    are retained: O(U_Q*D) preprocessing, then selected repair work. The
    stochastic estimate is unbiased for the fixed finite-credit vector in
    exact arithmetic, not for arbitrary downstream true-loss differences.
    """
    result = _credits_numpy(q, k, v_bits, grad_y, emb0, emb1, D, diag_y,
                            repair_count=repair_count, sampling_seed=seed,
                            forced_bits=forced_bits)
    cq, ck, cv, lengths, routes, stats, qb, kb = result
    d = cq.shape[1]
    support = stats["supported_repair_bits"]
    selected = stats["repair_bits"]
    forced = set(stats["forced_repair_bits"])
    optional = [bit for bit in support if bit not in forced]
    chosen_optional = [bit for bit in selected if bit not in forced]
    p = len(chosen_optional) / len(optional) if optional else 1.0
    probabilities = np.zeros(d, dtype=np.float64)
    for bit in support:
        probabilities[bit] = 1.0 if bit in forced else p
    for bit in chosen_optional:
        if p != 1:
            cq[:, bit] = qb + (cq[:, bit] - qb) / p
            ck[:, bit] = kb + (ck[:, bit] - kb) / p
    stats["repair_inclusion_probabilities"] = probabilities.tolist()
    stats["repair_estimator"] = "exact_deletion_plus_supported_inverse_probability_repair"
    stats["repair_sampling_exact"] = len(selected) == len(support)
    stats["repair_target_budget"] = operator.index(repair_count)
    stats["repair_sampling_seed"] = operator.index(seed)
    return cq, ck, cv, lengths, routes, stats


def _credits_numpy(q, k, v_bits, grad_y, emb0, emb1, D=None, diag_y=None,
                   repair_bits=None, repair_count=None, sampling_seed=0,
                   forced_bits=()):
    v = _bits(v_bits)
    n, d = v.shape
    if D is not None and operator.index(D) != d:
        raise ValueError("D does not match v_bits.shape[1]")
    if n >= _MAX_T or n * d > _MAX_ELEMENTS:
        raise ValueError("Requires T < 2**28 and T*D <= 2**31-1")
    qn, kn = _words(q, "Q"), _words(k, "K")
    grad = _array(grad_y, np.float64)
    e0, e1 = _array(emb0, np.float64), _array(emb1, np.float64)
    if qn.shape != (n,) or kn.shape != (n,) or grad.shape != (n, d) or e0.shape != (d,) or e1.shape != (d,):
        raise ValueError("Q/K, gradients, and embeddings have incompatible shapes")
    if d < 64 and (np.any(qn >> np.uint64(d)) or np.any(kn >> np.uint64(d))):
        raise ValueError("Q/K codes exceed the declared D-bit alphabet")
    if not all(np.isfinite(a).all() for a in (grad, e0, e1)):
        raise ValueError("Gradients and embeddings must be finite")
    diagonal = None
    if diag_y is not None:
        diagonal = _array(diag_y, np.float64)
        if diagonal.shape != (n, d):
            raise ValueError("diag_y must have the same [T,D] shape as grad_y")
        if not np.isfinite(diagonal).all() or np.any(diagonal < 0):
            raise ValueError("diag_y must be finite and nonnegative")
        if not np.any(diagonal):
            diagonal = None
    selection = None
    if repair_bits is not None:
        try:
            selection = [operator.index(bit) for bit in repair_bits]
        except (TypeError, ValueError) as exc:
            raise ValueError("repair_bits must be distinct integer indices") from exc
        if len(set(selection)) != len(selection) or any(bit < 0 or bit >= d for bit in selection):
            raise ValueError("repair_bits must be distinct indices in [0,D)")
        selection.sort()
    budget = None
    forced = []
    if repair_count is not None:
        if selection is not None:
            raise ValueError("Specify repair_bits or repair_count, not both")
        try:
            budget, sampling_seed = operator.index(repair_count), operator.index(sampling_seed)
            forced = [operator.index(bit) for bit in forced_bits]
        except (TypeError, ValueError) as exc:
            raise ValueError("Repair budget, seed and forced bits must be integers") from exc
        if not 1 <= budget <= d:
            raise ValueError("repair_count must be in [1,D]")
        if not 0 <= sampling_seed <= _MAX_WORD:
            raise ValueError("seed must fit an unsigned 64-bit integer")
        if len(set(forced)) != len(forced) or any(bit < 0 or bit >= d for bit in forced):
            raise ValueError("forced_bits must be distinct indices in [0,D)")
        forced.sort()
    cq, ck, cv = [np.zeros((n, d), dtype=np.float64) for _ in range(3)]
    lengths = np.zeros(n, dtype=np.int32)
    routes = np.full(n, -1, dtype=np.int32)
    counts = np.zeros(len(COUNTER_NAMES), dtype=np.uint64)
    arrays = (qn, kn, v, grad)
    if diagonal is not None:
        arrays += (diagonal,)
    arrays += (e0, e1, cq, ck, cv, lengths, routes, counts)
    if selection is None and budget is None:
        entry = _library().rosa_uniform if diagonal is None else _library().rosa_quadratic
        code = entry(n, d, *(a.ctypes.data_as(ctypes.c_void_p) for a in arrays))
    else:
        q_deletion = np.zeros(n, dtype=np.float64)
        k_deletion = np.zeros(n, dtype=np.float64)
        selection_counts = np.zeros(5, dtype=np.uint64)
        selected_arrays = (qn, kn, v, grad, diagonal, e0, e1, cq, ck, cv,
                           lengths, routes, counts, q_deletion, k_deletion, selection_counts)
        pointers = [ctypes.c_void_p() if a is None else a.ctypes.data_as(ctypes.c_void_p)
                    for a in selected_arrays]
        if budget is None:
            mask = sum(1 << bit for bit in selection)
            code = _library().rosa_selected(n, d, mask, *pointers)
        else:
            forced_mask = sum(1 << bit for bit in forced)
            code = _library().rosa_budgeted(n, d, budget, sampling_seed, forced_mask, *pointers)
            mask = int(selection_counts[4])
            selection = [bit for bit in range(d) if (mask >> bit) & 1]
    if code:
        raise RuntimeError(f"Exact credit core failed with code {code}")
    stats = dict(zip(COUNTER_NAMES, map(int, counts)))
    flags = stats["materialized_index_flags"]
    stats["lca_index_materialized"] = bool(flags & 1)
    stats["wavelet_index_materialized"] = bool(flags & 2)
    stats["possible_alignment_pairs"] = n * (n - 1) // 2
    stats["sum_match_lengths"] = int(lengths.sum(dtype=np.int64))
    if diag_y is not None:
        stats["curvature_model"] = "output_diagonal"
        stats["curvature_preprocess_elements"] = n * d if diagonal is not None else 0
        history = bool(stats["singleton_corridors"]) and diagonal is not None
        stats["curvature_history_materialized"] = history
        stats["curvature_working_bytes"] = (16 * d + (16 * n * d if history else 0)
                                             if diagonal is not None and n else 0)
        stats["curvature_avoided_payload_bytes"] = (8 * stats["payload_classes"] * d
                                                     if diagonal is not None else 0)
    result = (cq, ck, cv, lengths.tolist(), routes.tolist(), stats)
    if selection is not None:
        stats["repair_bits"] = selection
        stats["repair_lane_count"] = len(selection)
        stats["repair_lane_total"] = d
        stats["query_symbol_classes"] = int(selection_counts[0])
        stats["root_repair_entries"] = int(selection_counts[1])
        stats["transition_working_capacity_bytes"] = int(selection_counts[2])
        supported_mask = int(selection_counts[3])
        active_mask = int(selection_counts[4])
        stats["supported_repair_bits"] = [bit for bit in range(d) if (supported_mask >> bit) & 1]
        stats["active_repair_bits"] = [bit for bit in range(d) if (active_mask >> bit) & 1]
        stats["forced_repair_bits"] = forced
        return result + (q_deletion, k_deletion)
    return result


def pair_candidates_numpy(q, k, D, rows, q_allowed, k_allowed, max_pairs,
                          seed, bucket_cap, occurrence_cap, scan, candidate_cap,
                          candidate_trace=False):
    """Native static causal-band candidates as packed K-first bit codes.

    K(position,bit) is position*D+bit; Q adds T*D. Allowed-owner masks are
    binary [T] arrays. Candidate sampling uses mt19937_64 with exact integer
    uniform draws; its finite-budget samples differ from the old NumPy stream.
    Full-capacity pools and seeded priority selection retain the same meaning.
    Returned candidate_pairs in counters are packed integer pairs when asked.
    """
    import time
    started = time.perf_counter()
    try:
        d, max_pairs, seed, bucket_cap, occurrence_cap, scan, candidate_cap = map(
            operator.index, (D, max_pairs, seed, bucket_cap, occurrence_cap, scan, candidate_cap))
    except TypeError as exc:
        raise ValueError("Dimensions, budgets and seed must be integers") from exc
    if not 1 <= d <= 64 or max_pairs < 0 or min(bucket_cap, occurrence_cap, scan) < 1:
        raise ValueError("Invalid dimensions or candidate budgets")
    if candidate_cap < max_pairs or max(max_pairs, bucket_cap, occurrence_cap, scan, candidate_cap) > _MAX_ELEMENTS:
        raise ValueError("Candidate capacities must fit int32 and candidate_cap >= max_pairs")
    if not 0 <= seed <= _MAX_WORD:
        raise ValueError("seed must fit uint64")
    qn, kn = _words(q, "Q"), _words(k, "K")
    if qn.ndim != 1 or kn.shape != qn.shape:
        raise ValueError("Q/K must have matching [T] shape")
    n = len(qn)
    if n >= _MAX_T or n*d > _MAX_ELEMENTS:
        raise ValueError("Requires T < 2**28 and T*D <= 2**31-1")
    if d < 64 and (np.any(qn >> np.uint64(d)) or np.any(kn >> np.uint64(d))):
        raise ValueError("Q/K symbols exceed D bits")
    requested = _array(rows)
    if requested.ndim != 1 or (requested.size and requested.dtype.kind not in "iub") or np.any(requested < 0) or np.any(requested >= n):
        raise ValueError("rows must contain valid integer positions")
    requested = np.unique(requested).astype(np.int32)
    allowed = []
    for value in (q_allowed, k_allowed):
        raw = np.ones(n, dtype=np.uint8) if value is None else _array(value)
        if raw.shape != (n,) or np.any((raw != 0) & (raw != 1)):
            raise ValueError("Allowed-owner masks must have binary shape [T]")
        allowed.append(np.ascontiguousarray(raw, dtype=np.uint8))
    return _pair_candidates_prepared(qn, kn, d, requested, allowed[0], allowed[1],
        max_pairs, seed, bucket_cap, occurrence_cap, scan, candidate_cap,
        candidate_trace, started=started)


def _pair_candidates_prepared(qn, kn, d, requested, q_allowed, k_allowed,
                              max_pairs, seed, bucket_cap, occurrence_cap,
                              scan, candidate_cap, candidate_trace=False, *,
                              started=None):
    """Internal native call for already validated contiguous input arrays.

    The two public boundaries validate their own arguments once. This helper
    avoids repeating full Q/K scans and row sorting after the proposal wrapper
    has just checked and constructed the exact same arrays.
    """
    import time
    if started is None:
        started = time.perf_counter()
    n = len(qn)
    output_count = min(max_pairs, n*d)
    pairs = np.zeros((output_count, 2), dtype=np.uint64)
    trace_count = min(candidate_cap, n*d*max(0, 2*n*d-1))
    trace = np.zeros((trace_count, 2), dtype=np.uint64) if candidate_trace else None
    counts = np.zeros(12, dtype=np.uint64)
    timings = np.zeros(2, dtype=np.float64)
    ptr = lambda a: ctypes.c_void_p() if a is None else a.ctypes.data_as(ctypes.c_void_p)
    code = _library().rosa_pair_candidates(n, d, ptr(qn), ptr(kn), ptr(requested), len(requested),
        ptr(q_allowed), ptr(k_allowed), output_count, seed, bucket_cap, occurrence_cap,
        scan, candidate_cap, ptr(pairs), ptr(trace), ptr(counts), ptr(timings))
    if code:
        raise RuntimeError(f"Native causal candidate proposal failed with code {code}")
    names = ("bucket_probes", "examined_alignments", "symbol_comparisons", "candidate_events",
             "retained_candidates", "selected_pairs", "causal_symbols", "query_rows",
             "distinct_words", "sampled_word_events", "word_rng_calls", "occurrence_rng_calls")
    stats = dict(zip(names, map(int, counts)))
    stats.update(seed=seed, proposal_algorithm="static_causal_bands", proposal_implementation="native_cpp",
                 rng_algorithm="mt19937_64", preprocessing_seconds=float(timings[0]),
                 native_proposal_seconds=float(timings[1]), proposal_seconds=time.perf_counter()-started)
    if candidate_trace:
        stats["candidate_pairs"] = trace[:stats["retained_candidates"]].tolist()
    return pairs[:stats["selected_pairs"]], stats


def exact_stream_bit_credits(q, k, v_bits, grad_y, emb0_g, emb1_g, n_bits):
    """Five-result stream API; PyTorch input gradients produce CPU tensors."""
    if hasattr(grad_y, "detach") and not grad_y.is_floating_point():
        raise TypeError("The PyTorch stream API requires a real floating-point grad_y")
    cq, ck, cv, lengths, routes, _ = credits_numpy(q, k, v_bits, grad_y, emb0_g, emb1_g, n_bits)
    if hasattr(grad_y, "detach"):
        import torch
        cq, ck, cv = (torch.from_numpy(x).to(dtype=grad_y.dtype) for x in (cq, ck, cv))
    return cq, ck, cv, lengths, routes


if __name__ == "__main__":
    print(build(verbose=True))
