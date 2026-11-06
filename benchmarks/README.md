# Measuring the current C++ engine

[Results](../docs/subquadratic-results.md) · [中文结果](../docs/subquadratic-results.zh-CN.md)

The repository contains one production backward engine. Its runtime choices
are first/second credit, verified-cache on/off, and explicit statistical
approximation. No runner imports an old optimized backward implementation.

## Complete backward timing

```bash
python benchmarks/subquadratic_engine.py --repeats 7 --cpu 4 --output build/backward.json
```

Use an available logical CPU or omit `--cpu`. The runner alternates method
order and includes validation, allocation, index construction, all singleton
credits, trust/joint work, V/embedding gradients and logit mapping. It excludes
compilation, dense projections, the model forward, loss and optimizer.

`--quick` checks setup on small inputs. `--historical-baseline` can read an
explicitly labeled previous JSON timing receipt; it does not load its code.
The final [current receipt](results/subquadratic_engine.json) contains all raw
samples, input digests, source hashes, counters and numerical comparisons.

The cases include random symbols, repeated symbols, periodic strings, a long
single-boundary match and an irregular skyline of runs. These labels belong
to the benchmark generator; the engine has no such dispatch.

## Ordinary training checks

```bash
python benchmarks/subquadratic_toy.py --cpu 3 --steps 30 --seeds 1101 1102 1103 --lr .005 --output build/toy.json
```

The [current toy receipt](results/subquadratic_toy.json) records paired data,
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
forward matcher. The current lean forward implementation is unchanged by the
backward replacement. [forward.json](results/forward.json) is an earlier
measurement with its own source identity; it is not a new backward timing.

## Correctness and provenance

```bash
python tests/run.py --exhaustive --output build/validation.json
```

The current gate is [subquadratic_validation.json](results/subquadratic_validation.json).
The independent C ABI memory-safety check is
[subquadratic_sanitizers.json](results/subquadratic_sanitizers.json).

[subquadratic_comparison.json](results/subquadratic_comparison.json) and
[subquadratic_toy_comparison.json](results/subquadratic_toy_comparison.json)
were frozen before removing the previous engine. Their bytes and source hashes
are preserved as historical comparison evidence. They do not describe the
final source tree, and the old implementation is not retained as a runtime or
benchmark backend. Other old JSON/CSV files in `results/` likewise refer to
their recorded revisions, not the current engine.
