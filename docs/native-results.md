# Historical native-integration evidence

[中文](native-results.zh-CN.md) · [Current results](subquadratic-results.md) · [Current training interface](native-training.md)

The current measurements of the repository's only backward engine are in [subquadratic-results.md](subquadratic-results.md). The files below record earlier integration studies, under the source and binary identities stored in those records. Their timings, defaults, algorithms, and training trajectories are **historical evidence**, not measurements or usage instructions for the current implementation.

| Preserved raw record | Historical scope |
|---|---|
| [native_component.json](../benchmarks/results/native_component.json) | Component/layer timings and the recorded numerical comparisons |
| [exact_randomized.json](../benchmarks/results/exact_randomized.json) | Earlier search-acceleration experiment; not the current query-cache mechanism |
| [exact_randomized_provenance.json](../benchmarks/results/exact_randomized_provenance.json) | Source/probe identities for that experiment |
| [native_random_training_summary.json](../research/results/native_random_training_summary.json) | Training and held-out results, including regressions |
| [native_random_roundoff_trace.json](../research/results/native_random_roundoff_trace.json) | First-divergence numerical trace |
| [native_cpp_validation.json](../research/results/native_cpp_validation.json) | Validation receipt for its recorded historical build |

The old runtime, probe snapshot, and executable benchmark scripts are not retained as alternative backends. Reproducing an old result requires its recorded historical source and environment; running the current checkout is a new experiment. Raw records are kept so a favorable training or speed result cannot erase its counterexamples or provenance.

For today's algorithm, read [the derivation](../ALGORITHM.md), [the exact-cache contract](exact-randomized.md), and [the approximation tutorial](randomized-backward.md). Use [the current benchmark instructions](../benchmarks/README.md) and [tests](../tests/README.md) for this checkout.
