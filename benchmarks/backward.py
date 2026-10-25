"""Measure complete CPU counterfactual VJP calls of the current algorithm.

The default matrix reports current absolute latency and throughput. Supply an
external --baseline-dir for a smaller paired comparison with a pinned original
upstream commit; its code is compiled in a temporary cache, never bundled.
No timings run on import and no input-dependent runtime backend is selected.
"""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from conv_rosa_transformer import rosa_core


def runtime_hashes():
    package = ROOT / "conv_rosa_transformer"
    files = [package / "rosa_core.py", package / "build.py"]
    files += sorted((package / "csrc").glob("*.h"))
    files += sorted((package / "csrc").glob("*.cpp"))
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def cases(paired=False, smoke=False):
    if smoke:
        shapes = [("random8", 64, 8, 0), ("shadow", 64, 1, 2), ("all_same", 64, 1, 2)]
    elif paired:
        shapes = [(family, n, d, v) for family, d, v in
                  (("random_binary", 1, 0), ("random8", 8, 0),
                   ("periodic", 4, 0), ("shadow", 1, 2), ("all_same", 1, 2))
                  for n in (256, 1024)]
        shapes += [("random16", 4096, 16, 0)]
    else:
        shapes = [(family, n, d, v) for family, d, v in
                  (("random_binary", 1, 0), ("random8", 8, 0),
                   ("periodic", 4, 0), ("shadow", 1, 2), ("all_same", 1, 2))
                  for n in (1024, 4096, 8192)]
        shapes += [("random_binary", 16384, 1, 0), ("random_binary", 65536, 1, 0),
                   ("random16", 8192, 16, 0), ("shadow", 2048, 64, 1),
                   ("shadow", 4096, 64, 2), ("shadow", 2048, 64, 0)]
    return [dict(case_id=f"{family}_T{n}_D{d}_U{v or 'random'}", family=family,
                 T=n, D=d, requested_payload_classes=v) for family, n, d, v in shapes]


def make_inputs(case, seed):
    text = f"{seed}:{case['case_id']}"
    case_seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") % (2 ** 63 - 1)
    gen = torch.Generator(device="cpu").manual_seed(case_seed)
    n, d, family = case["T"], case["D"], case["family"]
    if family.startswith("random"):
        q = torch.randint(0, 1 << d, (n,), generator=gen, dtype=torch.int64).tolist()
        k = torch.randint(0, 1 << d, (n,), generator=gen, dtype=torch.int64).tolist()
    elif family == "periodic":
        q = [i % 4 for i in range(n)]
        k = q.copy()
        k[n // 2] ^= 1
    elif family == "shadow":
        q, k = [0] * n, [1] + [0] * (n - 1)
    elif family == "all_same":
        q, k = [0] * n, [0] * n
    else:
        raise ValueError(family)
    classes = case["requested_payload_classes"]
    if classes:
        bank = torch.randint(0, 2, (classes, d), generator=gen, dtype=torch.uint8)
        if classes == 2:
            bank[1, 0] = 1 - bank[0, 0]
        # Scalar repeated cases retain alternating payloads; wide cases use
        # an independently sampled class at each position.
        indices = torch.arange(n) % classes if d == 1 else torch.randint(classes, (n,), generator=gen)
        v = bank[indices]
    else:
        v = torch.randint(0, 2, (n, d), generator=gen, dtype=torch.uint8)
    g = torch.randn(n, d, generator=gen, dtype=torch.float32)
    e0, e1 = (torch.randn(d, generator=gen, dtype=torch.float32) for _ in range(2))
    return (q, k, v, g, e0, e1, d), case_seed


def input_digest(inputs):
    q, k, v, g, e0, e1, d = inputs
    digest = hashlib.sha256(json.dumps(dict(q=q, k=k, D=d), separators=(",", ":")).encode())
    for tensor in (v, g, e0, e1):
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def compare(actual, expected, atol, rtol):
    if len(actual) != 5 or len(expected) != 5:
        raise AssertionError("Expected the public five-result interface")
    route_ok = list(actual[3]) == list(expected[3]) and list(actual[4]) == list(expected[4])
    gradients = {}
    for name, a, b in zip(("q", "k", "v"), actual[:3], expected[:3]):
        a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        error = float((a - b).abs().max()) if a.numel() else 0.
        gradients[name] = dict(max_abs_error=error, finite=finite,
                               allclose=finite and bool(torch.allclose(a, b, atol=atol, rtol=rtol)))
    return dict(pass_check=route_ok and all(x["allclose"] for x in gradients.values()),
                routes_exact=route_ok, atol=atol, rtol=rtol, gradients=gradients)


def hardware():
    result = {"system": platform.system(), "machine": platform.machine(),
              "logical_cpu_count": os.cpu_count()}
    if hasattr(os, "sched_getaffinity"):
        result["affinity_cpu_ids"] = sorted(os.sched_getaffinity(0))
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("model name"):
                result["cpu_model"] = line.partition(":")[2].strip()
                break
    return result


def save(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)
    csv_path = path.with_suffix(".csv")
    csv_temporary = csv_path.with_name(csv_path.name + ".tmp")
    with csv_temporary.open("w", newline="") as stream:
        fields = ["case_id", "T", "D", "current_ms", "tokens_per_second", "credit_values_per_second",
                  "upstream_ms", "upstream_over_current", "input_sha256"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["cases"]:
            timings = row["timings"]
            value = {key: row[key] for key in ("case_id", "T", "D", "input_sha256")}
            value.update(current_ms=timings["current"]["median_seconds"] * 1000,
                         tokens_per_second=row["tokens_per_second"],
                         credit_values_per_second=row["credit_values_per_second"])
            if "upstream" in timings:
                value.update(upstream_ms=timings["upstream"]["median_seconds"] * 1000,
                             upstream_over_current=row["upstream_over_current"])
            writer.writerow(value)
    csv_temporary.replace(csv_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--filter", default="")
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--baseline-revision", default="32e90e272929942a764d696b639409b783c61bb1")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "backward.json")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        parser.error("At least one timed repetition and one warmup are required")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rosa_core.build()
    functions = {"current": rosa_core.exact_stream_bit_credits}
    source_before = runtime_hashes()
    compiler = shlex.split(os.environ.get("CXX", "cl" if os.name == "nt" else "c++"),
                           posix=os.name != "nt")
    if os.name == "nt":
        compiler = [part.strip('"') for part in compiler]
    try:
        compiler_version = subprocess.run(compiler + ["--version"], capture_output=True,
                                          text=True, timeout=15, check=True).stdout.splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        compiler_version = " ".join(compiler)
    report = {"status": "RUNNING", "metadata": {
        "started_utc": datetime.now(timezone.utc).isoformat(), "hardware": hardware(),
        "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "compiler": compiler_version,
        "current_compile_flags": (["/nologo", "/O2", "/DNDEBUG", "/std:c++17", "/EHsc", "/fp:strict", "/LD"]
                                  if Path(compiler[0]).name.lower() in ("cl", "cl.exe", "clang-cl", "clang-cl.exe") else
                                  ["-O3", "-DNDEBUG", "-std=c++17", "-fPIC", "-shared", "-ffp-contract=off"]),
        "torch_threads": 1, "repeats": args.repeats, "warmup": args.warmup, "seed": args.seed,
        "timing": "complete public five-result CPU float32 counterfactual VJP call",
        "included": ["input validation and conversions", "per-call indices and baseline routes",
                     "Q/K/V credit calculation", "output allocation and conversion"],
        "excluded": ["compile/import", "input generation", "warmup", "correctness checks",
                     "dense projections", "autograd layer and end-to-end training"],
        "source_sha256": source_before,
        "memory_reporting": "returned credit tensor bytes only; no peak RSS claim"}, "cases": []}
    if args.baseline_dir:
        from upstream import load_upstream
        upstream, settings = load_upstream(args.baseline_dir, args.baseline_revision)
        functions["upstream"] = upstream.exact_stream_bit_credits
        report["metadata"]["upstream"] = settings
    for case in cases(paired=bool(args.baseline_dir), smoke=args.smoke):
        if args.filter not in case["case_id"]:
            continue
        inputs, seed = make_inputs(case, args.seed)
        q, k, v, g, e0, e1, d = inputs
        precise = (q, k, v, g.double(), e0.double(), e1.double(), d)
        current64 = rosa_core.credits_numpy(*precise)
        row = dict(case, seed=seed, input_sha256=input_digest(inputs), counters=current64[5])
        if args.baseline_dir:
            historical64 = upstream.exact_stream_bit_credits(*precise)
            row["float64_comparison"] = compare(current64[:5], historical64, 1e-8, 1e-10)
            if not row["float64_comparison"]["pass_check"]:
                raise AssertionError((case, row["float64_comparison"]))
        warmed = {}
        for _ in range(args.warmup):
            for name, function in functions.items():
                warmed[name] = function(*inputs)
        row["current_float32_vs_float64"] = compare(warmed["current"], current64[:5], 2e-4, 2e-5)
        if not row["current_float32_vs_float64"]["pass_check"]:
            raise AssertionError((case, row["current_float32_vs_float64"]))
        if args.baseline_dir:
            # The upstream deliberately uses float32 intermediate reductions.
            # Publish differences rather than equating real-arithmetic identity
            # with bitwise agreement across different reduction orders.
            row["float32_comparison"] = compare(warmed["current"], warmed["upstream"], 2e-4, 2e-5)
            if not row["float32_comparison"]["routes_exact"]:
                raise AssertionError("Baseline route mismatch in public float32 call")
        row["returned_credit_tensor_bytes"] = sum(t.numel() * t.element_size()
                                                     for t in warmed["current"][:3])
        del warmed
        samples = {name: [] for name in functions}
        gc.collect()
        gc.disable()
        try:
            names = list(functions)
            for repeat in range(args.repeats):
                for name in (names if repeat % 2 == 0 else names[::-1]):
                    start = time.perf_counter_ns()
                    output = functions[name](*inputs)
                    samples[name].append((time.perf_counter_ns() - start) * 1e-9)
                    del output
        finally:
            gc.enable()
        row["timings"] = {name: dict(samples_seconds=values,
                                     median_seconds=statistics.median(values),
                                     min_seconds=min(values), max_seconds=max(values))
                          for name, values in samples.items()}
        duration = row["timings"]["current"]["median_seconds"]
        row["tokens_per_second"] = case["T"] / duration
        row["credit_values_per_second"] = 3 * case["T"] * case["D"] / duration
        if args.baseline_dir:
            row["upstream_over_current"] = row["timings"]["upstream"]["median_seconds"] / duration
        report["cases"].append(row)
        save(report, args.output)
        print(f"{case['case_id']}: {duration * 1000:.3f} ms, "
              f"{row['tokens_per_second']:,.0f} token/s"
              + (f", upstream/current={row['upstream_over_current']:.3f}" if args.baseline_dir else ""),
              flush=True)
    if not report["cases"]:
        raise ValueError("No case matched the selected filter")
    if source_before != runtime_hashes():
        raise RuntimeError("Implementation changed during measurement; discard this run")
    report["status"] = "PASS"
    report["metadata"]["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save(report, args.output)


if __name__ == "__main__":
    main()
