# Measuring the current C++ engine

[Results](../docs/subquadratic-results.md) · [中文结果](../docs/subquadratic-results.zh-CN.md)

The repository contains one production backward engine. Its runtime choices
are first/second credit, verified-cache on/off, and explicit statistical
approximation. The optional paired comparison loads a separate, user-supplied
frozen checkout; no previous engine is shipped with this repository.

## Complete backward timing

```bash
python benchmarks/subquadratic_engine.py --expanded --repeats 9 --cpu 4 --output build/backward.json
```

Use an available logical CPU or omit `--cpu`. The runner alternates method
order and includes validation, allocation, index construction, all singleton
credits, trust/joint work, V/embedding gradients and logit mapping. It excludes
compilation, dense projections, the model forward, loss and optimizer.

`--expanded` includes additional lengths from 1024 to 8192 and narrow
alphabets, so small or unfavorable cases stay visible. `--quick` checks setup
on small inputs. `--historical-baseline` can read an
explicitly labeled previous JSON timing receipt; it does not load its code.
The final [current receipt](results/optimized_engine.json) contains all raw
samples, input digests, source hashes, counters and numerical comparisons.

The cases include random symbols, repeated symbols, periodic strings, a long
single-boundary match and an irregular skyline of runs. These labels belong
to the benchmark generator; the engine has no such dispatch.

## Paired comparison with a frozen checkout

```bash
python benchmarks/compare_checkout.py --reference-checkout /path/to/frozen/checkout \
    --expanded --repeats 9 --cpu 4 --output build/comparison.json
```

Both implementations receive byte-identical arrays, forward options and joint
proposal seeds. Calls are interleaved in alternating order on the same logical
CPU; compilation and the model forward stay outside the timer. The receipt
retains every sample, input/source/library hash, forward parity, joint-pair
parity and gradient discrepancies. The reference checkout is read-only and
must be supplied externally. [optimized_comparison.json](results/optimized_comparison.json)
is the current comparison; no result is selected by input-family dispatch in
the production engine.

## Ordinary training checks

```bash
python benchmarks/subquadratic_toy.py --cpu 3 --steps 30 --seeds 1101 1102 1103 --lr .005 --output build/toy.json
```

The [current toy receipt](results/optimized_toy.json) records paired data,
initial parameters, per-step parameter hashes, loss, gradient errors and wall
clock samples. Every update is a normal `zero_grad / model / loss.backward /
optimizer.step` loop. The runner asserts one model forward and one component
backward per update. It uses two short structured tasks, not a language-model
quality or large-model throughput benchmark. The two task/model definitions
in `research/` contain no alternative backward engine.

## Forward-only measurement

```bash
python benchmarks/forward.py --baseline-dir /path/to/external/checkout --cpu 4 --repeat 7 --output build/forward.json
```

This optional comparison uses the specified external checkout for its previous
forward matcher. The current forward shares its cursor with bounded Q scans
and has bounded hash probes with an ordered overflow map. [forward.json](results/forward.json) is an earlier
measurement with its own source identity; it is not a new backward timing.

## Correctness and provenance

```bash
python tests/run.py --exhaustive --output build/validation.json
```

The current gate is [optimized_validation.json](results/optimized_validation.json).
The independent native memory-safety checks are in
[optimized_sanitizers.json](results/optimized_sanitizers.json).

[subquadratic_comparison.json](results/subquadratic_comparison.json) and
[subquadratic_toy_comparison.json](results/subquadratic_toy_comparison.json)
were frozen before removing the previous engine. Their bytes and source hashes
are preserved as historical comparison evidence. They do not describe the
final source tree, and the old implementation is not retained as a runtime.
Other old JSON/CSV files in `results/` likewise refer to
their recorded revisions, not the current engine.
