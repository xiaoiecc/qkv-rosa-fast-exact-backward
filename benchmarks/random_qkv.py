"""Compare complete CPU backward on random Q, K and V against a frozen checkout.

Gaussian and Bernoulli logits are independent over position, channel and Q/K/V.
The held-out distribution shifts Gaussian logits so each packed bit is one
with probability 0.8. Forward, compilation, tracing and parity checks are
outside the timer. Include allocations, complete native backward, and the
ordinary NumPy wrapper. Both credit orders use the public defaults.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from compare_checkout import ROOT, current, hashes, load_reference, revision
from benchmarks.subquadratic_engine import errors


WIDTHS = (1, 2, 4, 8, 12, 16, 32, 64)
LENGTHS = {"gaussian": (1024, 4096, 8192), "bernoulli": (4096,), "biased": (4096,)}
SEEDS = (173431, 352117, 597353)


def make_inputs(family, n, width, seed):
    # Independent streams make each case reproducible in isolation and ensure
    # the additional distributions do not reuse the measured Gaussian draws.
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, width, tuple(LENGTHS).index(family)]))
    shape = (1, 1, n, width)
    if family == "bernoulli":
        logits = [(2 * rng.integers(0, 2, shape, dtype=np.int8) - 1) * .35 for _ in range(3)]
    else:
        shift = .8416212335729143 if family == "biased" else 0.
        logits = [rng.standard_normal(shape) + shift for _ in range(3)]
    embeds = [rng.standard_normal((1, width)) for _ in range(2)]
    return logits + embeds, rng.standard_normal(shape) / np.sqrt(width)


def input_digest(arrays):
    digest = hashlib.sha256()
    for value in arrays:
        digest.update(value.dtype.str.encode())
        digest.update(str(value.shape).encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def measure(modules, arrays, gradient, order, repeats, seed):
    caches, forward, traced = {}, {}, {}
    for name, module in modules.items():
        forward[name], caches[name], _ = module.training_forward_numpy(
            *arrays, credit_order=order, randomized=True, proposal_seed=seed+17, repair_seed=seed+31)
        traced[name] = module.training_backward_numpy(caches[name], gradient, trace=True)
    reference, current_result = traced["reference"], traced["current"]
    discrepancy = errors(reference, current_result)
    forward_equal = np.array_equal(forward["reference"].view(np.uint8), forward["current"].view(np.uint8))
    code_equal = np.array_equal(reference[-1]["pair_codes"], current_result[-1]["pair_codes"])
    cost_close = np.allclose(reference[-1]["pair_costs"], current_result[-1]["pair_costs"],
                             rtol=2e-10, atol=2e-9, equal_nan=True)
    close = all(np.allclose(a, b, rtol=2e-10, atol=2e-9)
                for a, b in zip(reference[:5], current_result[:5]))
    samples = {name: [] for name in modules}
    names = tuple(modules)
    for rep in range(-1, repeats):
        for name in (names if (rep + seed) % 2 else names[::-1]):
            start = time.perf_counter_ns()
            result = modules[name].training_backward_numpy(caches[name], gradient)
            elapsed = (time.perf_counter_ns() - start) * 1e-6
            if rep >= 0:
                samples[name].append(elapsed)
            # Exclude releasing the previous call's arrays from the timer.
            del result
    median = {name: statistics.median(values) for name, values in samples.items()}
    return dict(samples_ms=samples, median_ms=median, speedup=median["reference"]/median["current"],
                forward_equal=bool(forward_equal), pair_codes_equal=bool(code_equal),
                pair_costs_close=bool(cost_close), gradient_close=bool(close), errors=discrepancy,
                max_match_length=int(caches["current"].lengths.max()),
                workspace_estimate_bytes={name: value[-1]["workspace_estimate_bytes"]
                                          for name, value in traced.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkout", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=tuple(LENGTHS), default=tuple(LENGTHS))
    parser.add_argument("--widths", nargs="+", type=int, default=WIDTHS)
    parser.add_argument("--lengths", nargs="+", type=int, help="override the default lengths per distribution")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("use at least three repetitions")
    if any(not 1 <= w <= 64 for w in args.widths) or any(t < 1 for t in args.lengths or ()):
        parser.error("widths must be in [1,64] and lengths positive")
    reference = args.reference_checkout.resolve()
    if reference == ROOT:
        parser.error("use a separate frozen reference checkout")
    os.sched_setaffinity(0, {args.cpu})
    roots = {"reference": reference, "current": ROOT}
    modules = {"reference": load_reference(reference), "current": current}
    snapshots = {name: hashes(root) for name, root in roots.items()}
    before = {name: revision(root) for name, root in roots.items()}
    binaries = {name: hashlib.sha256(Path(module._library()._name).read_bytes()).hexdigest()
                for name, module in modules.items()}
    rows = []
    for family in args.families:
        for n in args.lengths or LENGTHS[family]:
            for width in args.widths:
                group = []
                for seed in args.seeds:
                    arrays, grad = make_inputs(family, n, width, seed)
                    digest = input_digest([*arrays, grad])
                    for order in ("first", "second"):
                        row = measure(modules, arrays, grad, order, args.repeats, seed)
                        row.update(family=family, T=n, bits=width, order=order, seed=seed,
                                   input_sha256=digest)
                        rows.append(row)
                        group.append(row)
                speeds = {order: statistics.geometric_mean(r["speedup"] for r in group if r["order"] == order)
                          for order in ("first", "second")}
                print(json.dumps(dict(family=family, T=n, bits=width, speedup=speeds)), flush=True)
    unchanged = all(snapshots[name] == hashes(root) and before[name] == revision(root)
                    for name, root in roots.items())
    passed = unchanged and all(row["gradient_close"] and row["forward_equal"] and row["pair_codes_equal"]
                               and row["pair_costs_close"] and row["errors"]["v_embeddings_bitwise_equal"]
                               for row in rows)
    report = dict(status="PASS" if passed else "FAIL", source_unchanged=unchanged, scope=__doc__,
                  cpu=args.cpu, platform=platform.platform(), numpy=np.__version__, repeats=args.repeats,
                  thread_settings={key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                  checkouts={name: dict(revision=before[name], source_sha256=snapshots[name],
                                       binary_sha256=binaries[name]) for name in roots}, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
