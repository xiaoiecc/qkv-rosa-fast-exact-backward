"""Interleave complete backward calls against an external, frozen checkout.

The installed package still has one engine. The reference is a separately
loaded checkout used only by this benchmark. Both receive identical inputs,
credit options and proposal seeds; compilation and forward are outside timing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from conv_rosa_transformer import native_training as current
from benchmarks.subquadratic_engine import errors, make_case


def hashes(root):
    package = root / "conv_rosa_transformer"
    paths = sorted(package.glob("*.py"))
    paths += sorted((package / "csrc").rglob("*.h"))
    paths += sorted((package / "csrc").glob("*.cpp"))
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def revision(root):
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                   text=True).strip()


def load_reference(root):
    package = types.ModuleType("rosa_benchmark_reference")
    package.__path__ = [str(root / "conv_rosa_transformer")]
    sys.modules[package.__name__] = package
    return importlib.import_module(package.__name__ + ".native_training")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkout", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--expanded", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/optimized_comparison.json")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("use at least three interleaved repetitions")
    reference_root = args.reference_checkout.resolve()
    if reference_root == ROOT:
        parser.error("reference must be an external frozen checkout")
    if hasattr(os, "sched_setaffinity"):
        cpu = args.cpu if args.cpu is not None else min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cpu})
    reference = load_reference(reference_root)
    modules = {"reference": reference, "current": current}
    roots = {"reference": reference_root, "current": ROOT}
    before = {name: hashes(root) for name, root in roots.items()}
    libraries = {name: module._library() for name, module in modules.items()}
    compiled = {name: hashes(root) for name, root in roots.items()}
    cases = [("random", 512, 1), ("random", 2048, 8), ("random", 8192, 16),
             ("random", 4096, 64), ("unary", 4096, 8), ("periodic", 4096, 4),
             ("shadow", 4096, 8), ("shadow", 8192, 8), ("skyline", 16, 8), ("skyline", 32, 8)]
    if args.expanded:
        cases += [("random", n, d) for n, d in
                  ((1024, 1), (2048, 1), (4096, 1), (1024, 8), (4096, 8),
                   (8192, 8), (1024, 16), (4096, 16), (8192, 64))]
        cases += [("unary", 1024, 8), ("periodic", 1024, 4), ("shadow", 1024, 8)]
    rows = []
    for case, (family, n, d) in enumerate(cases):
        arrays, gradient = make_case(family, n, d, 2026110600 + case)
        for order in ("first", "second"):
            caches, outputs, times = {}, {}, {name: [] for name in modules}
            forward = {}
            for name, module in modules.items():
                forward[name], caches[name], _ = module.training_forward_numpy(
                    *arrays, credit_order=order, randomized=True,
                    proposal_seed=719, repair_seed=817)
                outputs[name] = module.training_backward_numpy(caches[name], gradient, trace=True)
            forward_equal = forward["reference"].tobytes() == forward["current"].tobytes()
            for repeat in range(args.repeats):
                names = list(modules)
                if repeat & 1:
                    names.reverse()
                for name in names:
                    start = time.perf_counter_ns()
                    outputs[name] = modules[name].training_backward_numpy(caches[name], gradient, trace=True)
                    times[name].append((time.perf_counter_ns() - start) / 1e6)
            medians = {name: statistics.median(samples) for name, samples in times.items()}
            quality = errors(outputs["reference"], outputs["current"])
            numerical_close = all(np.allclose(a, b, rtol=2e-10, atol=2e-9)
                                  for a, b in zip(outputs["reference"][:5], outputs["current"][:5]))
            pair_codes_equal = (outputs["reference"][-1]["pair_codes"].tobytes()
                                == outputs["current"][-1]["pair_codes"].tobytes())
            row = dict(family=family, T=gradient.shape[2], D=d, order=order,
                       seed=2026110600 + case, median_ms=medians, samples_ms=times,
                       input_sha256=[hashlib.sha256(x.tobytes()).hexdigest()
                                     for x in (*arrays, gradient)],
                       speedup=medians["reference"] / medians["current"],
                       errors=quality, numerical_close=numerical_close,
                       forward_bytewise_equal=forward_equal, pair_codes_equal=pair_codes_equal)
            rows.append(row)
            print(json.dumps({key: row[key] for key in
                              ("family", "T", "D", "order", "median_ms", "speedup", "numerical_close")}), flush=True)
    unchanged = before == compiled == {name: hashes(root) for name, root in roots.items()}
    passed = unchanged and all(row["numerical_close"] and row["forward_bytewise_equal"]
                               and row["pair_codes_equal"] for row in rows)
    report = dict(status="PASS" if passed else "FAIL", source_unchanged=unchanged,
                  scope=__doc__, repeats=args.repeats, platform=platform.platform(),
                  numpy=np.__version__, affinity=sorted(os.sched_getaffinity(0))
                  if hasattr(os, "sched_getaffinity") else [],
                  checkouts={name: dict(path=str(root), revision=revision(root), source_sha256=before[name],
                                       library_sha256=hashlib.sha256(Path(libraries[name]._name).read_bytes()).hexdigest())
                             for name, root in roots.items()}, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
