# Correctness gates

Run the sole production engine against independent mathematical definitions:

```bash
python tests/run.py
python tests/run.py --exhaustive --output build/validation.json
```

The exhaustive command is the CI gate. It adds standalone C++ structural
oracles and longer binary enumeration. The JSON receipt records test results,
coverage counters, elapsed time and production source hashes; changing sources
during the run makes the gate fail. NumPy, a C++17 compiler and CPU PyTorch are
sufficient. The optional CUDA transfer check is skipped when CUDA is absent.

## What the tests establish

| Test | Independent check |
|---|---|
| `test_core` | All-bit finite credits against literal hard rerouting, exhaustive short binary strings, floating cases and bit 63 |
| `test_curvature` | A supplied diagonal row cost on Q/K, with the native linear V convention; direct edited-output loss differences |
| `test_subquadratic_engine` | Literal native training gradients, selected joint corners, every bit width 1–64, long structured inputs, exact-cache bitwise parity, approximate coverage and estimator means |
| `test_forward_lean` | Every causal endpoint against direct suffix comparisons; longest/latest ties and full-width words |
| `test_layer` | Grouped/noncontiguous tensor layouts, literal gradients, embedding scatters, state dictionaries, bfloat16 and transformer use |
| `test_native_training` | One ordinary forward/backward, owned forward caches, checkpointing, saved configuration, RNG, CPU autocast and nonfinite adjoints |
| `test_native_approximation` | Preserved hard forward, V/embedding gradients, joint corners and ordinary autograd in sampled modes |
| `test_approx_row_certificate` | Exhaustive exact-zero certificates against independent edited routes; no research backend is imported |
| `test_expansion` | Exact dyadic expansion and rounding against Python rational arithmetic |
| `test_validation` | Input conversion boundaries and rejected complex/integer adjoints |
| `test_structures` | C++ deletion, Q/long-K repair, compact short-K repair and complete C ABI oracles |

The C++ structural oracles include `subquadratic_delete.cpp`,
`subquadratic_repair.cpp`, `subquadratic_short.cpp` and
`subquadratic_training.cpp`. `subquadratic_correlation.cpp` independently tests
shifted interval contractions, dimension 65, multiple FFT block sizes and severe
cancellation. Its result is also checked by `test_subquadratic_engine`.

## Numerical and randomization contracts

Routes require exact equality. Turning the verified query cache on or off
requires byte-for-byte identical numeric outputs. Complete mode includes all
finite-credit terms, but regrouping and FFT do not preserve an older sequence
of rounded binary64 operations. Numeric oracle checks therefore state their
absolute/relative tolerances. Ill-conditioned sums are also checked against the
sum of product magnitudes and their absolute errors are reported explicitly.

Approximate mode samples complete row residuals or selected numeric channels;
it never samples candidates before taking their route maximum. The tests check
full coverage, fixed-seed repeatability, positive-probability estimator means,
constant-output invariance and exact restoration of selected joint corners.
These are implementation contracts, not a promise that every noisy gradient
lowers a model's loss.

There is one production engine. Obsolete optimized implementations and their
private-kernel tests are removed. Earlier comparison JSON files are retained as
historical measurements; they are never loaded as executable test backends.
