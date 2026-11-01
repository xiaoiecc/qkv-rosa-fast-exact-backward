# Reproducing the randomized-backward study

[Results](../docs/randomized-results.md) · [中文结果](../docs/randomized-results.zh-CN.md)
· [Derivation](../docs/randomized-backward.md)
· [Final manifest](results/randomized_final_manifest.json)

The PR contains one final runtime. The larger research tree, old runtime
snapshots, captures and rejected candidate helpers are supplied in a separate
study archive. They are evidence and reproduction inputs, not selectable
production backends. Production code does not import `research`.

The online reference is `b510fd9479946f10e46ced1a6265695cbc131d0a`; the intermediate
row sampler is `5e3e9f81965242492ae5a3870efd3cfbbca4bf9e` (P1). Final measured native
SHA-256 is `a732d50fd3f12342320f19306b6a5292b3d6aa829ff2c3a8b43d64e28dfc9cb5`.
The manifest distinguishes final measurements from historical runs. Do not
rename an earlier report as a result of the current runtime.

## Entry points and measurement scopes

| Script | Purpose and scope |
|---|---|
| `benchmarks/random_final_native.py` | All 26 cases, five methods, nine interleaved repetitions of complete public native backward; includes validation, metric/proposals, credits, joint replays, mapping, V/embedding accumulation and trace reporting |
| `training_round_phase2.py` | Paired ordinary model forward/loss/backward/Adam; one fresh process per method, shared seeds/data/settings, source and loaded-library hashes |
| `training_round_channel_native.py` | Public-API channel controls: raw logits, Q-only/shared QK parameter gradients and input gradients; diagnostic call times are not the formal performance benchmark |
| `training_round_final_report.py` | Summarizes the four retained final training reports, including every failure; runs no new training |

Private credit-core timings exclude surrounding native work. The native timer
excludes model projections/forward, loss, optimizer and compilation. Whole-step
training includes those model/optimizer operations, but excludes data generation,
evaluation, compilation and diagnostic logging. These scopes must not be mixed.

## Environment

Use Python, NumPy, PyTorch and a GCC/Clang-compatible C++17 compiler. Run commands
from the repository root. The recorded runs used Linux CPU affinity; select a
logical CPU available on your machine. Limit numerical-library threads before
starting Python:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

The training scripts also set Torch intra/inter-op threads to one. Main native
runs used CPU 8; training and channel controls used CPU 2. These CPU results do
not measure GPU kernels. Compilation happens outside the reported timers.

## Complete native backward

Restore the frozen online and P1 trees from the study archive. Restore captured
inputs under the recorded `research/results/training_round_captures_context512`,
`training_round_captures_prefix4096` and `training_round_captures_prefix64_2048`
directories, preserving their `index.json`, `.npz` and metadata `.json` files.
Then, replacing checkout paths and CPU as needed:

```bash
python benchmarks/random_final_native.py --online-root ../online-b510fd9 --phase1-root ../phase1-5e3e9f8 --cpu 8 --repeats 9 --output build/random_final_native_rerun.json
```

The retained [native report](../benchmarks/results/random_final_native.json)
contains all samples, per-draw errors, selected-joint comparisons, source hashes,
library hashes and the unchanged-source check. It compares online exact, current
exact, P1 row32, current row32/full channels and current row32/eight channels.
Large errors and slower cases remain in the table.

## Ordinary training with the current public API

This command needs only the current runtime and shared task scripts; explicitly
listing methods avoids the driver's default historical P1 comparison:

```bash
python research/training_round_phase2.py --methods exact final final_channels --channels 8 --rows 32 --order second --tasks prefix_copy64_fresh --queries 2048 --examples 1 --steps 20 --lr .003 --seeds 2401 2402 --cpu 2 --output build/final_training_rerun.json
```

`exact` uses default exact randomized search. `final` uses explicit approximation
with 32 row samples and all spectral channels. `final_channels` additionally
sets the channel budget to 8. Use `--order first` to test the first-order rule.
To include the historical method, add `row_phase1` to `--methods` and provide
`--phase1-root ../phase1-5e3e9f8`. To repeat the longer structural stress, use
`--queries 4096 --seeds 1101 --methods exact final_channels`.

Each update still has one model forward and one backward. The reported loss is
the row mean of `0.5 * sum(error**2)`, not per-coordinate MSE. Twenty-update prefix
runs test a persistent expensive structure, not general convergence. The final
report preserves the first-order failure and the fresh seed where eight-channel
sampling was slower and less accurate than full channels.

## Channel controls and report generation

The capture below and its adjacent metadata JSON are supplied in the study
archive. The public probe covers balanced, dominant-channel and tangent adjoints
in both credit orders. It reports Q alone as well as Q/K together, since an exact
large K gradient can hide a poor Q estimate:

```bash
python research/training_round_channel_native.py --capture research/results/training_round_captures_prefix64_2048/prefix_copy64_fresh_second_s1101_p0.npz --budgets 0 8 --draws 16 --cpu 2 --output build/final_channel_controls_rerun.json
python research/training_round_final_report.py
```

The second command reads the four fixed `training_round_final_channels_*.json`
reports named in that script and rewrites their summary, not their measurements.
Keep reruns under new output paths. Do not overwrite a measured file and keep
presenting its old summary as the same evidence.

For packaging, the pure NumPy `adjoint` and `summary` functions were moved
**without changing their bodies** into `training_round_probe_utils.py`. This
removes the public channel probe's import dependency on private prototypes.
The measured reports retain the original script hashes; the study archive
preserves the original entry point at
`research/snapshots/reproduction-original/training_round_channel_native.py`.
No production source or measured numerical function changed in this relocation.

## Retained history

The P1 59-run study and nonlinear-stem failure remain historical evidence in the
study archive. The intermediate 600-update context comparison matched P1; its
aggregate has `SOURCE_CHANGED` and is not a final frozen timing result. Older
native-migration reports likewise retain their own source identities. The
manifest and [results report](../docs/randomized-results.md) explain these limits.
Small bindings used by independent oracle tests remain in the PR; they are not
alternative runtime implementations. See [test instructions](../tests/README.md).
