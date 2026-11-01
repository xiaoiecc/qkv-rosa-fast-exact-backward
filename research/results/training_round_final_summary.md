# Final native training checks

These are deliberately structured prefix-copy stress tasks. Each run uses B1, D64, Adam at learning rate 0.003 and 20 ordinary updates. The timer covers forward, loss, backward and Adam; data generation, evaluation, compilation and logging are excluded. Every update has exactly one model forward and one component backward. OpenBLAS, OMP, MKL and Torch threads were all limited to one, with process affinity CPU2.

The regression objective is the mean over rows of `0.5 * sum(error**2)`. Seed 1101 was used in discovery; seeds 2401 and 2402 were held out from it. Twenty updates test early training and the lifetime of the costly structure, not convergence or generalization to arbitrary networks.

Final native SHA256: `a732d50fd3f12342320f19306b6a5292b3d6aa829ff2c3a8b43d64e28dfc9cb5`. Every report below is marked `COMPLETE`; the historical row implementation has its own frozen binary. There are 15 runs and 300 updates.

| T | Rule | Seed | Method | Train-step seconds | Speedup vs exact | Initial loss | Final loss | First changed parameter update | First changed hard forward |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | second | 1101 | exact | 5.379006 | 1.000 | 27.0626984 | 14.8342638 | none | none |
| 2048 | second | 1101 | row_phase1 | 0.829930 | 6.481 | 27.0626984 | 13.7741413 | 1 | 20 |
| 2048 | second | 1101 | final | 0.830443 | 6.477 | 27.0626984 | 14.7566032 | 13 | 20 |
| 2048 | second | 1101 | final_channels | 0.652209 | 8.247 | 27.0626984 | 14.5978823 | 1 | 20 |
| 2048 | second | 2401 | exact | 5.346104 | 1.000 | 27.9437485 | 9.3580217 | none | none |
| 2048 | second | 2401 | final | 0.866877 | 6.167 | 27.9437485 | 9.3580217 | 18 | none |
| 2048 | second | 2401 | final_channels | 0.760651 | 7.028 | 27.9437485 | 9.3538885 | 1 | none |
| 2048 | second | 2402 | final_channels | 0.883773 | 6.063 | 27.2220917 | 26.7526684 | 1 | none |
| 2048 | second | 2402 | final | 0.792692 | 6.759 | 27.2220917 | 26.3919792 | 20 | none |
| 2048 | second | 2402 | exact | 5.357976 | 1.000 | 27.2220917 | 26.3919792 | none | none |
| 2048 | first | 1101 | exact | 4.009273 | 1.000 | 27.0626984 | 43.1005936 | none | none |
| 2048 | first | 1101 | final | 0.902561 | 4.442 | 27.0626984 | 43.5106049 | 11 | 16 |
| 2048 | first | 1101 | final_channels | 0.842973 | 4.756 | 27.0626984 | 43.9109993 | 1 | 13 |
| 4096 | second | 1101 | exact | 19.501015 | 1.000 | 27.0771446 | 15.5468969 | none | none |
| 4096 | second | 1101 | final_channels | 1.530265 | 12.744 | 27.0771446 | 15.5468969 | 1 | none |

`exact` is the default exact randomized mode. `final` is approximate mode with 32 row samples and full spectral channels. `final_channels` sets the optional channel budget to 8. `row_phase1` is the archived row-only implementation.

At T4096, the optional eight-channel mode took 1.53027 seconds against 19.5010 seconds for exact: 12.7436 times less complete train-step time. The 20 training losses and hard forwards matched exact on this run; its Q/K parameter updates did not. The same discovery seed at T2048 gave 8.2474 times less time. This is a size trend on a particular structural stress case, not a universal complexity benchmark.

Failures and regressions remain visible. With the first-order rule, all three final-runtime runs increased loss from about 27.06 to 43–44 under this learning-rate configuration. On second-order seed 2402, training made little progress, and eight-channel sampling was both slower and slightly worse in loss than full channels. Do not use the successful second-order seed to claim that first-order learning or every seed is equally reliable.

The broader 300-update tasks, equal-time comparisons, and nonlinear-stem failure are archived separately in `training_round_frozen_results.md`; they used the earlier frozen row runtime. The final spectral path was assessed here, and the unchanged row path had an additional 600-update equivalence check in `training_round_phase2_findings.md`.

Raw source and binary hashes, per-step timings, losses, hard-forward hashes, parameter hashes and native work counters remain in the JSON reports.
