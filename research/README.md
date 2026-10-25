# Reproducible native-training study

The retained study fixes the previous selected learning rule, then compares
native integration and an exact randomized search toggle. It preserves every
seed and negative result; it does not select a favorable training checkpoint.
[English report](../docs/native-results.md) · [中文报告](../docs/native-results.zh-CN.md).

| Script | Purpose |
|---|---|
| native_random_toy.py | Shared task/model definitions; no alternate runtime |
| native_random_training.py | Fresh-seed ordinary model/loss/backward/Adam runs |
| native_random_summary.py | Aggregate all saved seeds and compare every on/off update |
| native_random_mode_contract.py | First/second, Dropout and framework-RNG invariance |
| native_random_noise_ceiling.py | Oracle best importance weights for residual thinning; analytical error, not claimed speed |
| native_random_roundoff_trace.py | Trace the worst held-out MSE seed to first hard-bit divergence |

The external frozen baseline is f6d7c4b73d7040b1b33860f5a2649c3cd3036cc5, included
as `baseline-f6` in the companion download. It is not a public upstream commit.
For example, with this checkout next to that directory:

```bash
python research/native_random_training.py --implementation baseline --baseline-root ../baseline-f6 --cpu 0 --output build/training_baseline.json
python research/native_random_training.py --implementation native --baseline-root ../baseline-f6 --cpu 0 --output build/training_native.json
python research/native_random_mode_contract.py --cpu 0 --output build/mode_contract.json
python research/native_random_noise_ceiling.py --baseline-root ../baseline-f6 --cpu 0 --output build/noise_ceiling.json
python research/native_random_roundoff_trace.py --baseline-root ../baseline-f6 --cpu 0 --output build/roundoff_trace.json
```

Use `--help` for explicit seeds, tasks and outputs. The summary script consumes
the checked-in final report names in `research/results`; its stored summary
records raw-file hashes. Do not replace a measured report with a new run while
continuing to present its old summary as the same evidence.

The small `joint_bits.py` and `causal_candidates.py` modules are compatibility
imports used by tests, not runtime implementations. The full earlier research
archive remains in the downloadable companion package rather than adding all
its historical trials to the PR. Production paths import no research code.
