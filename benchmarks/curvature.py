"""Measure the added cost of a supplied diagonal quadratic output metric.

Includes public NumPy argument validation, baseline routing, contraction and
returned arrays. Excludes estimating the metric: h is already available.
Alternating measurement order avoids systematically favoring one call.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from conv_rosa_transformer import rosa_core


def source_hashes():
    package = ROOT / "conv_rosa_transformer"
    files = [package / "rosa_core.py"]
    files += sorted((package / "csrc").glob("*.h"))
    files += sorted((package / "csrc").glob("*.cpp"))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files}


def make_case(family, n, d, seed):
    rng = np.random.default_rng(seed)
    if family == "random":
        q = rng.integers(0, 1 << d, n, dtype=np.uint64)
        k = rng.integers(0, 1 << d, n, dtype=np.uint64)
    elif family == "periodic":
        q = np.arange(n, dtype=np.uint64) % 4
        k = q.copy()
        k[n // 2] ^= np.uint64(1)
    else:
        q = np.zeros(n, dtype=np.uint64)
        k = q.copy()
        if family == "shadow":
            k[0] = 1
    v = rng.integers(0, 2, (n, d), dtype=np.uint8)
    if d == 64:
        bank = rng.integers(0, 2, (2, d), dtype=np.uint8)
        v = bank[np.arange(n) % 2]
    g = rng.normal(size=(n, d))
    h = rng.uniform(0.1, 1.5, (n, d))
    e0, e1 = rng.normal(size=(2, d))
    return q, k, v, g, h, e0, e1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmarks/results/curvature.json")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("At least three paired samples are required")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    rosa_core.build()
    hashes = source_hashes()
    cpu = "unknown"
    info = Path("/proc/cpuinfo")
    if info.exists():
        for line in info.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    report = {
        "status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
        "measurement": "public NumPy CPU full credit call; supplied metric; warm library",
        "curvature_estimation_included": False, "seed": 20261025,
        "repeats": args.repeats, "cpu": cpu, "platform": platform.platform(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        "source_sha256": hashes, "rows": [],
    }
    cases = (("random", 1024, 8), ("random", 8192, 8), ("random", 8192, 16),
             ("all_same", 8192, 1), ("shadow", 8192, 1),
             ("periodic", 4096, 4), ("shadow", 2048, 64))
    for index, (family, n, d) in enumerate(cases):
        q, k, v, g, h, e0, e1 = make_case(family, n, d, 20261025 + index)
        def first():
            return rosa_core.credits_numpy(q, k, v, g, e0, e1)
        def quadratic():
            return rosa_core.quadratic_credits_numpy(q, k, v, g, h, e0, e1)
        one, two = first(), quadratic()
        assert one[3:5] == two[3:5]
        for value in two[:3]:
            assert np.isfinite(value).all()
        # Route discovery is independent of the numerical output metric.
        for key in rosa_core.COUNTER_NAMES:
            assert one[5][key] == two[5][key], (family, key)
        samples = [[], []]
        calls = (first, quadratic)
        for repeat in range(args.repeats):
            for which in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                begin = time.perf_counter_ns()
                calls[which]()
                samples[which].append((time.perf_counter_ns() - begin) / 1e6)
        first_ms, quadratic_ms = map(statistics.median, samples)
        row = {
            "family": family, "T": n, "D": d,
            "first_order_ms": first_ms, "quadratic_ms": quadratic_ms,
            "quadratic_over_first_order": quadratic_ms / first_ms,
            "first_order_samples_ms": samples[0], "quadratic_samples_ms": samples[1],
            "routes_exact": True, "structural_counters_equal": True,
            "curvature_working_array_bytes": two[5]["curvature_working_bytes"],
            "curvature_avoided_payload_bytes": two[5]["curvature_avoided_payload_bytes"],
            "curvature_input_bytes": h.nbytes,
            "curvature_history_materialized": two[5]["curvature_history_materialized"],
            "returned_array_bytes": sum(a.nbytes for a in two[:3]),
        }
        report["rows"].append(row)
        print(json.dumps({key: row[key] for key in ("family", "T", "D", "first_order_ms",
                          "quadratic_ms", "quadratic_over_first_order",
                          "curvature_working_array_bytes")} ), flush=True)
    unchanged = hashes == source_hashes()
    report["source_unchanged"] = unchanged
    report["status"] = "PASS" if unchanged else "INVALID_SOURCE_CHANGED"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    if not unchanged:
        raise SystemExit("Runtime sources changed during measurement")
    print(json.dumps({"status": report["status"], "rows": len(report["rows"]),
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
