"""Paired small-model training checks; not evidence of a general loss advantage.

Each update is the ordinary zero_grad / model(x) / loss.backward / step loop.
All methods share initial parameters, data and framework RNG. Every method
uses the sole production engine. Evaluation is outside timed steps.
The two tasks deliberately stress repeated routes; they are not language-model
benchmarks. The independent C++ and NumPy oracles carry the correctness claims.
The inherited LR .03 destabilized the exact baselines. One lower-LR check used
only legacy/complete; neither approximate method receives its own tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import training_round_tasks as tasks
from conv_rosa_transformer import rosa_layer as layer_module
from conv_rosa_transformer import rosa_core


METHODS = {
    "complete": dict(randomized=False, backward_mode="exact"),
    "verified_cache": dict(randomized=True, backward_mode="exact"),
    "row_sample": dict(randomized=True,
                       backward_mode="approximate", approximation_rows=8),
    "channel_sample": dict(randomized=True,
                           backward_mode="approximate", approximation_rows=8,
                           approximation_channels=4),
}
TASKS = (("periodic_copy", 64), ("prefix_copy", 128))


def hash_arrays(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        if isinstance(array, torch.Tensor):
            array = array.detach().contiguous().cpu().numpy()
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def source_hashes():
    paths = sorted((ROOT / "conv_rosa_transformer").glob("*.py"))
    paths += sorted((ROOT / "conv_rosa_transformer/csrc").rglob("*.h"))
    paths += sorted((ROOT / "conv_rosa_transformer/csrc").glob("*.cpp"))
    paths += [Path(__file__), ROOT / "research/training_round_tasks.py",
              ROOT / "research/native_random_toy.py"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def run_one(method, order, task, length, seed, steps, examples, lr):
    torch.manual_seed(seed)
    settings = dict(METHODS[method], credit_order=order)
    model = tasks.make_model(layer_module.NBitQKVRosa, settings, task)
    initial_hash = hash_arrays(model.state_dict().values())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x, y = tasks.make_data(seed, task, examples, queries=length)
    xt, yt = tasks.make_data(seed, task, max(8, 4 * examples), split=1, queries=length)
    data_hash = hash_arrays((x, y, xt, yt))
    framework_rng = torch.get_rng_state().clone()
    counts = dict(forward=0, backward=0)
    native_work = []
    original_backward = layer_module.training_backward_numpy

    def observed_backward(cache, gradient):
        counts["backward"] += 1
        result = original_backward(cache, gradient)
        native_work.append({k: v for k, v in result[-1].items()
                            if isinstance(v, (int, float, bool, str))})
        return result

    handle = model.rosa.register_forward_hook(
        lambda *unused: counts.__setitem__("forward", counts["forward"] + 1))

    def evaluation():
        model.eval()
        with torch.no_grad():
            return dict(fit=tasks.metrics(model(x), y, task),
                        test=tasks.metrics(model(xt), yt, task))

    losses, timings, parameter_hashes, first_gradient, first_qk = [], [], [], None, None
    with patch.object(layer_module, "training_backward_numpy", observed_backward):
        initial = evaluation()
        for step in range(steps):
            model.train()
            torch.set_rng_state(framework_rng)
            before = counts.copy()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = tasks.loss_value(model(x), y, task)
            loss.backward()
            if step == 0:
                first_gradient = torch.cat([p.grad.detach().reshape(-1)
                                            for p in model.parameters() if p.grad is not None]).double().numpy()
                first_qk = torch.cat([p.grad.detach().reshape(-1)
                                     for module in (model.rosa.q_proj, model.rosa.k_proj)
                                     for p in module.parameters() if p.grad is not None]).double().numpy()
            optimizer.step()
            timings.append(time.perf_counter() - start)
            framework_rng = torch.get_rng_state().clone()
            if counts["forward"] - before["forward"] != 1 or counts["backward"] - before["backward"] != 1:
                raise AssertionError("training update used more than one model forward/backward")
            losses.append(float(loss.detach()))
            parameter_hashes.append(hash_arrays(model.state_dict().values()))
        final = evaluation()
    handle.remove()
    record = dict(method=method, credit_order=order, settings=settings, task=task,
                  length=length, seed=seed, steps=steps, examples=examples, learning_rate=lr,
                  initial_state_sha256=initial_hash, input_sha256=data_hash,
                  initial=initial, final=final, training_losses=losses,
                  training_seconds=sum(timings), median_step_seconds=statistics.median(timings),
                  step_seconds=timings, every_update_parameter_sha256=parameter_hashes,
                  training_model_forwards=steps, training_backward_calls=steps,
                  native_work_first=native_work[0], native_work_last=native_work[-1],
                  final_framework_rng_sha256=hash_arrays((framework_rng,)))
    return record, first_gradient, first_qk


def discrepancy(actual, expected):
    norm = float(np.linalg.norm(expected))
    other = float(np.linalg.norm(actual))
    return dict(relative_l2=float(np.linalg.norm(actual - expected)) / max(norm, 1e-30),
                maximum_absolute=float(np.max(np.abs(actual - expected), initial=0)),
                cosine=float(np.dot(actual, expected) / (norm * other)) if norm and other else None,
                bitwise_identical=actual.tobytes() == expected.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/subquadratic_toy.json")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1101, 2203, 3307])
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--lr", type=float, default=.005)
    parser.add_argument("--pilot-reports", type=Path, nargs="*", default=[])
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=list(METHODS))
    parser.add_argument("--orders", nargs="+", choices=("first", "second"), default=["first", "second"])
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    if any(args.output.resolve() == path.resolve() for path in args.pilot_reports):
        raise ValueError("historical pilot receipts are read-only")
    if "complete" not in args.methods:
        raise ValueError("include complete for paired gradient diagnostics")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    before = source_hashes()
    library = rosa_core._library()
    after_build = source_hashes()
    records = []
    for task, length in TASKS:
        for order in args.orders:
            for index, seed in enumerate(args.seeds):
                batch = {}
                # Alternate order to reduce systematic cache/thermal advantage.
                for method in (args.methods if index % 2 == 0 else args.methods[::-1]):
                    row, gradient, qk = run_one(method, order, task, length, seed,
                                               args.steps, args.examples, args.lr)
                    batch[method] = (row, gradient, qk)
                    print(json.dumps(dict(task=task, order=order, seed=seed, method=method,
                                          loss=row["final"]["test"]["loss"], seconds=row["training_seconds"])), flush=True)
                reference, base_gradient, base_qk = batch["complete"]
                for method, (row, gradient, qk) in batch.items():
                    if row["initial_state_sha256"] != reference["initial_state_sha256"] or row["input_sha256"] != reference["input_sha256"]:
                        raise AssertionError("paired methods did not share model/data")
                    row["first_gradient_vs_complete"] = discrepancy(gradient, base_gradient)
                    row["first_qk_gradient_vs_complete"] = discrepancy(qk, base_qk)
                    row["parameter_trajectory_identical_to_complete"] = (
                        row["every_update_parameter_sha256"] == reference["every_update_parameter_sha256"])
                    records.append(row)
    after = source_hashes()
    summary = []
    for task, length in TASKS:
        for order in args.orders:
            for method in args.methods:
                rows = [r for r in records if (r["task"], r["credit_order"], r["method"]) == (task, order, method)]
                summary.append(dict(task=task, length=length, credit_order=order, method=method,
                                    test_loss_mean=float(np.mean([r["final"]["test"]["loss"] for r in rows])),
                                    test_loss_std=float(np.std([r["final"]["test"]["loss"] for r in rows])),
                                    train_loss_mean=float(np.mean([r["final"]["fit"]["loss"] for r in rows])),
                                    median_step_ms=1000 * statistics.median([r["median_step_seconds"] for r in rows]),
                                    first_qk_relative_error_mean=float(np.mean([r["first_qk_gradient_vs_complete"]["relative_l2"] for r in rows])),
                                    all_parameter_trajectories_identical=all(r["parameter_trajectory_identical_to_complete"] for r in rows)))
    unchanged = before == after_build == after
    pilots = []
    for path in args.pilot_reports:
        pilot = json.loads(path.read_text())
        pilots.append(dict(report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           status=pilot["status"], native_library=pilot["native_library"],
                           shared_learning_rates=sorted(set(row["learning_rate"] for row in pilot["records"])),
                           summary=pilot["summary"]))
    report = dict(status="COMPLETE" if unchanged else "SOURCE_CHANGED",
                  source_before=before, source_after_build=after_build, source_after=after,
                  source_unchanged=unchanged,
                  numpy=np.__version__, torch=torch.__version__, python=sys.version,
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  native_library=dict(path=library._name,
                                      sha256=hashlib.sha256(Path(library._name).read_bytes()).hexdigest()),
                  protocol="fixed small training sets; paired seeds; Adam; 1 forward/1 backward per step",
                  learning_rate_selection="Inherited LR .03 destabilized exact baselines; one .005 check used only legacy/complete, then .005 was fixed for all methods.",
                  pilot_reports=pilots,
                  limitations="Structured T=64/128 toy tasks, three seeds and 30 steps do not establish general training quality or large-model speed.",
                  summary=summary, records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
