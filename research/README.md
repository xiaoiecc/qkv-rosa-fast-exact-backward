# Task definitions and historical evidence

The production implementation is the single C++ engine under
`conv_rosa_transformer/csrc`. This directory contains no old optimized
backward engine and is not imported by production code.

`native_random_toy.py` and `training_round_tasks.py` define small datasets and
ordinary model scaffolding used by [the current toy runner](../benchmarks/subquadratic_toy.py).
They use the current public layer; they do not implement a credit engine.

[Current results](../docs/subquadratic-results.md) ·
[中文结果](../docs/subquadratic-results.zh-CN.md) ·
[Reproduction commands](../benchmarks/README.md)

Files under `results/` are preserved historical evidence from earlier studies.
Their source identities, dates, failures, and scopes remain as recorded.
Statements such as “current” inside those frozen files refer to the measured
snapshot. The previous research drivers and bindings have been removed with
the old engine; these files are not claims about the current code or executable
alternative backends.
