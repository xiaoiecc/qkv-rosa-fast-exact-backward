"""Compare complete native CPU forward matchers on identical packed inputs.

Both timers include SAM construction, all auxiliary indexes actually built,
and every causal longest/latest query. Python validation/list conversion and
neural projections are excluded. The old implementation comes from an explicit
external checkout; it is never a selectable runtime backend in this package.
"""
from pathlib import Path
import argparse
import ctypes
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_forward_lean import build_forward_probe
from conv_rosa_transformer import rosa_core as current_module


def load_baseline(root):
    source = root / "conv_rosa_transformer" / "rosa_core.py"
    spec = importlib.util.spec_from_file_location("_external_forward_baseline", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module._library().rosa_matching_stats


def cases(quick=False):
    if quick:
        return [(f, 8192) for f in ("all_same", "random2", "random16", "random256", "random64", "periodic", "shadow")]
    return ([(f, n) for f in ("all_same", "random2", "random16", "random256", "random64", "periodic", "shadow")
             for n in (128, 8192)] +
            [("random16", 1024), ("all_same", 65536), ("random2", 65536),
             ("random256", 65536), ("random64", 65536), ("periodic", 65536), ("shadow", 65536)])


def inputs(family, n):
    seed = int.from_bytes(hashlib.sha256(f"20261025:{family}:{n}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    if family.startswith("random"):
        if family == "random64":
            q = rng.bit_generator.random_raw(n)
            k = rng.bit_generator.random_raw(n)
            # Keep both actual uint64 matches and nonmatches represented.
            q[1::4] = k[:-1:4]
        else:
            alphabet = int(family.removeprefix("random"))
            q = rng.integers(alphabet, size=n, dtype=np.uint64)
            k = rng.integers(alphabet, size=n, dtype=np.uint64)
    elif family == "all_same":
        q = np.zeros(n, dtype=np.uint64); k = q.copy()
    elif family == "shadow":
        q = np.zeros(n, dtype=np.uint64); k = q.copy(); k[0] = 1
    else:
        q = np.arange(n, dtype=np.uint64) % 4; k = q.copy(); k[n//2] ^= np.uint64(1)
    return q, k


def measure(call, batches):
    begin = time.perf_counter_ns()
    for _ in range(batches):
        if call():
            raise RuntimeError("native matcher failed")
    return (time.perf_counter_ns()-begin) / batches / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/forward.json")
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    _, probe = build_forward_probe()
    baseline_module, older = load_baseline(args.baseline_dir.resolve())
    newer = current_module._library().rosa_matching_stats
    header = ROOT / "conv_rosa_transformer/csrc/forward.h"
    source_hash = hashlib.sha256(header.read_bytes()).hexdigest()
    report = dict(status="RUNNING", started_utc=datetime.now(timezone.utc).isoformat(),
                  scope="native full route matcher, includes construction; excludes Python validation/list conversion and neural projections",
                  baseline_directory=str(args.baseline_dir.resolve()),
                  baseline_library_sha256=hashlib.sha256(baseline_module.build().read_bytes()).hexdigest(),
                  current_library_sha256=hashlib.sha256(Path(current_module._library()._name).read_bytes()).hexdigest(),
                  measured_unified_core_sha256=hashlib.sha256((ROOT/"conv_rosa_transformer/csrc/unified_core.cpp").read_bytes()).hexdigest(),
                  forward_header_sha256=source_hash,
                  machine=platform.machine(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                  platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,
                  repeats=args.repeat, counter_probe_hash_seed=20261025,
                  timed_hash_seed_policy="production per-process random seed; complete key equality",
                  public_scope="matching_stats on contiguous uint64 arrays, includes validation, native matcher, and conversion of both output arrays to Python lists",
                  rows=[])
    if Path("/proc/cpuinfo").exists():
        report["cpu_model"] = next((line.split(":",1)[1].strip() for line in
                                   Path("/proc/cpuinfo").read_text().splitlines()
                                   if line.startswith("model name")), "unknown")
    gc.disable()
    try:
        for family, n in cases(args.quick):
            q, k = inputs(family, n)
            l = np.zeros(n, dtype=np.int32); r = np.zeros(n, dtype=np.int32)
            l0 = np.zeros(n, dtype=np.int32); r0 = np.zeros(n, dtype=np.int32)
            counts = np.zeros(10, dtype=np.uint64)
            ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)
            pq, pk, pl, pr, pl0, pr0, pc = map(ptr, (q,k,l,r,l0,r0,counts))
            old_call = lambda: older(n,pq,pk,pl0,pr0)
            new_call = lambda: newer(n,pq,pk,pl,pr)
            if probe(n,pq,pk,pl,pr,20261025,pc):
                raise RuntimeError("counter probe failed")
            def old_public():
                baseline_module.matching_stats(q,k)
                return 0
            def new_public():
                current_module.matching_stats(q,k)
                return 0
            old_warm = measure(old_call,1); new_warm = measure(new_call,1)
            if not (np.array_equal(l,l0) and np.array_equal(r,r0)):
                raise AssertionError((family,n,"routes differ"))
            if current_module.matching_stats(q,k)!=baseline_module.matching_stats(q,k):
                raise AssertionError((family,n,"public routes differ"))
            batches = max(1,min(1000,int(3/max(min(old_warm,new_warm),.001))))
            old_samples=[];new_samples=[]
            for repetition in range(args.repeat):
                order = ((old_call,old_samples),(new_call,new_samples)) if repetition%2==0 else ((new_call,new_samples),(old_call,old_samples))
                for call, samples in order:
                    samples.append(measure(call,batches))
            old_ms=statistics.median(old_samples);new_ms=statistics.median(new_samples)
            old_public_samples=[];new_public_samples=[]
            for repetition in range(args.repeat):
                order = ((old_public,old_public_samples),(new_public,new_public_samples)) if repetition%2==0 else ((new_public,new_public_samples),(old_public,old_public_samples))
                for call, samples in order:
                    samples.append(measure(call,batches))
            old_public_ms=statistics.median(old_public_samples)
            new_public_ms=statistics.median(new_public_samples)
            row=dict(family=family,T=n,baseline_ms=old_ms,current_ms=new_ms,speedup=old_ms/new_ms,
                     baseline_public_ms=old_public_ms,current_public_ms=new_public_ms,
                     public_speedup=old_public_ms/new_public_ms,
                     baseline_public_samples_ms=old_public_samples,current_public_samples_ms=new_public_samples,
                     tokens_per_second=1000*n/new_ms,batches_per_sample=batches,
                     baseline_samples_ms=old_samples,current_samples_ms=new_samples,
                     input_sha256=hashlib.sha256(q.tobytes()+k.tobytes()).hexdigest(),
                     routes_exact=True,transition_probes=int(counts[0]),suffix_fallbacks=int(counts[1]),
                     endpoint_queries=int(counts[2]),endpoint_updates=int(counts[3]),hash_probe_steps=int(counts[4]),
                     retained_vector_capacity_bytes=int(counts[5]),states=int(counts[6]),edges=int(counts[7]),
                     hash_slots=int(counts[8]),endpoint_index_materialized=bool(counts[9]))
            report["rows"].append(row)
            print(f"{family:12s} {n:6d}  {old_ms:9.4f} -> {new_ms:9.4f} ms  {old_ms/new_ms:6.2f}x",flush=True)
    finally:
        gc.enable()
    if hashlib.sha256(header.read_bytes()).hexdigest()!=source_hash:
        raise RuntimeError("forward.h changed during measurement")
    report["status"]="PASS";report["finished_utc"]=datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report,indent=2)+"\n");temporary.replace(args.output)
    print(json.dumps({"status":report["status"],"rows":len(report["rows"]),"output":str(args.output)}))


if __name__ == "__main__":
    main()
