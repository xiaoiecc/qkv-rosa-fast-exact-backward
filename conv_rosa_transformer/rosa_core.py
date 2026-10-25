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
    cq, ck, cv = [np.zeros((n, d), dtype=np.float64) for _ in range(3)]
    lengths = np.zeros(n, dtype=np.int32)
    routes = np.full(n, -1, dtype=np.int32)
    counts = np.zeros(len(COUNTER_NAMES), dtype=np.uint64)
    arrays = (qn, kn, v, grad, e0, e1, cq, ck, cv, lengths, routes, counts)
    code = _library().rosa_uniform(n, d, *(a.ctypes.data_as(ctypes.c_void_p) for a in arrays))
    if code:
        raise RuntimeError(f"Exact credit core failed with code {code}")
    stats = dict(zip(COUNTER_NAMES, map(int, counts)))
    flags = stats["materialized_index_flags"]
    stats["lca_index_materialized"] = bool(flags & 1)
    stats["wavelet_index_materialized"] = bool(flags & 2)
    stats["possible_alignment_pairs"] = n * (n - 1) // 2
    stats["sum_match_lengths"] = int(lengths.sum(dtype=np.int64))
    return cq, ck, cv, lengths.tolist(), routes.tolist(), stats


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
