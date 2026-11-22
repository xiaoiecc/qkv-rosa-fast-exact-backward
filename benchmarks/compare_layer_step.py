"""Interleaved CPU measurements of one PyTorch mixer update from identical state."""
import argparse
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import torch

from compare_checkout import ROOT, hashes, load_reference, revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkout", required=True, type=Path)
    parser.add_argument("--cpu", required=True, type=int)
    parser.add_argument("--repeats", default=11, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed-offsets", type=int, nargs="+", default=[0],
                        help="repeat each shape with independently seeded weights and data")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("at least three interleaved measurements are required")
    os.sched_setaffinity(0, {args.cpu})
    torch.set_num_threads(1)
    reference = args.reference_checkout.resolve()
    if reference == ROOT:
        parser.error("use a separate checkout as the reference")
    roots = {"reference": reference, "current": ROOT}
    before = {name: hashes(root) for name, root in roots.items()}
    load_reference(reference)
    modules = {name: importlib.import_module(package + ".rosa_layer")
               for name, package in (("reference", "rosa_benchmark_reference"),
                                     ("current", "conv_rosa_transformer"))}
    rows = []
    shapes = ((2048, 8, 1), (4096, 32, 8), (4096, 32, 16), (2048, 64, 64))
    for (n, channels, bits), offset in itertools.product(shapes, args.seed_offsets):
        for order in ("first", "second"):
            seed = 701 + n + bits + offset
            torch.manual_seed(seed)
            x = torch.randn(1, n, channels)
            target = torch.randn_like(x)
            models = {name: module.NBitQKVRosa(channels, bits, credit_order=order)
                      for name, module in modules.items()}
            initial = {k: t.clone() for k, t in models["current"].state_dict().items()}
            optimizers = {name: torch.optim.SGD(model.parameters(), lr=.001)
                          for name, model in models.items()}
            times = {name: [] for name in models}
            maximum = {"output": 0., "gradient": 0., "updated_parameter": 0.}
            passed = True
            for rep in range(-1, args.repeats):
                observations = {}
                for name in (list(models) if rep % 2 == 0 else reversed(models)):
                    model = models[name]
                    model.load_state_dict(initial)
                    torch.manual_seed(seed + 100)
                    start = time.perf_counter_ns()
                    optimizers[name].zero_grad(set_to_none=True)
                    y = model(x)
                    loss = (y - target).square().mean()
                    loss.backward()
                    optimizers[name].step()
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    if rep >= 0:
                        times[name].append(elapsed)
                    observations[name] = (y.detach(), torch.cat([p.grad.flatten() for p in model.parameters()]),
                                          torch.cat([p.detach().flatten() for p in model.parameters()]))
                for key, old, new in zip(maximum, observations["reference"], observations["current"]):
                    maximum[key] = max(maximum[key], (old - new).abs().max().item())
                    passed &= torch.allclose(old, new, rtol=2e-5, atol=2e-7)
            med = {name: statistics.median(values) for name, values in times.items()}
            row = dict(T=n, channels=channels, bits=bits, order=order, seed=seed,
                       median_ms=med, speedup=med["reference"] / med["current"],
                       samples_ms=times, numerical_close=passed, max_abs=maximum,
                       input_sha256=hashlib.sha256(x.numpy().tobytes()+target.numpy().tobytes()).hexdigest())
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("T", "channels", "bits", "order", "median_ms",
                                                  "speedup", "numerical_close")}), flush=True)
    unchanged = before == {name: hashes(root) for name, root in roots.items()}
    passed = unchanged and all(row["numerical_close"] for row in rows)
    report = dict(status="PASS" if passed else "FAIL", scope=__doc__, optimizer="SGD(lr=.001)",
                  batch=1, dtype="float32", repeats=args.repeats, cpu=args.cpu, threads=1,
                  torch=torch.__version__, platform=platform.platform(), source_unchanged=unchanged,
                  checkouts={name: revision(root) for name, root in roots.items()}, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
