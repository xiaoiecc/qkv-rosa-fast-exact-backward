# Correctness tests

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
