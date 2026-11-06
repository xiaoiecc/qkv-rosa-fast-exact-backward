# Historical approximation evidence

[中文](randomized-results.zh-CN.md) · [Current results](subquadratic-results.md) · [Current estimator](randomized-backward.md)

Current backward latency, workspace, and approximation error are reported in [subquadratic-results.md](subquadratic-results.md). The earlier results below belong to the source/binary hashes recorded in their JSON files. They do not describe the current runtime, its defaults, or its cache and numerical-channel implementation.

| Preserved raw record | Historical scope |
|---|---|
| [random_final_native.json](../benchmarks/results/random_final_native.json) | Earlier complete/approximate backward timings and gradient-error samples |
| [randomized_final_manifest.json](../research/results/randomized_final_manifest.json) | Evidence manifest and implementation identities |
| [training_round_final_summary.json](../research/results/training_round_final_summary.json) | Structured prefix-copy training runs and their recorded outcomes |
| [training_round_final_channel_controls.json](../research/results/training_round_final_channel_controls.json) | Channel-estimation controls, including first-order cancellation failures |
| [training_round_final_channel_controls_full.json](../research/results/training_round_final_channel_controls_full.json) | Corresponding complete-channel controls |
| [native_random_noise_ceiling.json](../research/results/native_random_noise_ceiling.json) | Earlier whole-lane noise experiment, not a lower bound for the current estimators |

The old executable implementations and reproduction scripts are not shipped as alternate algorithms. These raw records preserve successes, slowdowns, estimation failures, and seed-specific outcomes; they are not a new performance claim. Historical labels such as `exact`, `final`, or `A8` have the meanings recorded by those experiments and should not be silently mapped onto a current API call.

The current engine offers verified full-key caching and explicit residual-row or centered-channel sampling. Their contracts are derived in [exact-randomized.md](exact-randomized.md) and [randomized-backward.md](randomized-backward.md). For this checkout, use the [current benchmark instructions](../benchmarks/README.md) and [training interface](native-training.md).
