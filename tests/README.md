# Correctness tests

## Native training and exact randomized roots

Run `python tests/run.py --exhaustive` from the repository root. The native
validation snapshot records 88 tests with one CUDA skip, including exhaustive
route/credit oracles; later randomized-join coverage was run separately.
[Validation report](../research/results/native_cpp_validation.json) and
[curated-tree audit](../research/results/native_curated_audit.json) retain
commands, source hashes and exact scopes. Counts overlap.

New tests cover both credit orders, both logit mappings, empty inputs, bit63,
first/second × exact randomization off/on, independent random streams, selected
joint corners, the real two-bit zero-signal barrier, native versus frozen
Python arithmetic with explicit tolerances, ordinary optimizers, checkpoint,
GradScaler and strict weight restoration. Native off/on equality is bitwise;
old Python migration equality is tolerance-based.

`tests/frozen_training_oracle.py` loads the test-only frozen layer snapshot.
It is never a runtime backend. Production code has no dependency on research;
two small research compatibility shims remain solely for existing oracle tests.
The newest root tests exercise direct/two-block/four-block plans and complete
selected/sampled/forced trajectories. The exact random flag never enables
approximate repair sampling.

The ordinary training evidence separately checks every update and downstream
Dropout RNG. It also retains the native migration's held-out-loss regression:
[results](../docs/native-results.md). CUDA-specific integration remains skipped
in this CPU environment. Passing API checks is not evidence of native GPU
kernels or model generalization.

## Original independent oracle design


Run from the repository root with Python, NumPy, PyTorch, and a C++17 compiler:

```bash
python tests/run.py
python tests/run.py --exhaustive
```

The default mode performs 341 exhaustive binary Q/K inputs, floating and
64-bit all-flip checks, constant-forward identities, saved cancellation
witnesses, exact dyadic expansion checks, and layer tests. `--exhaustive`
extends the binary enumeration to 5,461 inputs and enables three standalone
C++ structural oracles. The report defaults to `build/validation.json`;
`--output` changes its destination. GCC/Clang are supported by the standalone
oracle build helper, including inside WSL; select one with `CXX`.

`oracle.py` implements the hard definition directly. It tries each causal
endpoint, compares suffix characters, applies the longest-then-latest rule,
flips each Q/K/V bit, and reruns the full forward. It does not share the
production deletion/repair decomposition. Integer cases require exact
equality. Ordinary floating cases use explicit absolute and relative
tolerances, while routes always require exact equality.

The C++ oracles test the actual production headers: deletion routes and
interval contraction, causal occurrence membership and latest endpoints,
and the union of ordinary repairs with singleton continuations. They are
test programs, not alternative production backends.

The numerical tests distinguish three contracts:

- Constant hard outputs must have strictly zero Q/K credits, even when
  causally irrelevant gradients are very large.
- Saved nonconstant cancellation fixtures assert independently computed
  integer answers for their documented target owner. Other ill-conditioned
  ordinary accumulators are not covered by that stronger assertion.
- ExactExpansion is compared with Python `Fraction` both for represented
  binary64-event sums and final round-to-nearest conversion. These tests
  require IEEE round-to-nearest, gradual underflow, no intermediate overflow,
  and no fast-math compiler transformations.

The report includes hashes of the implementation actually loaded and verifies
that those sources did not change during the run. Test counts at different
layers can overlap and should not be added as independent whole-core cases.
