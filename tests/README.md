# Tests

Run the correctness suite from the repository root:

```bash
python tests/run.py               # Fast Python unit tests
python tests/run.py --exhaustive  # Full CI gate (adds standalone C++ oracles)
```

For AddressSanitizer and UndefinedBehaviorSanitizer checks (Linux/macOS with GCC/Clang):

```bash
python tests/sanitize.py --jobs 2 --no-leaks
```

## Coverage overview

| File | What it verifies |
|---|---|
| `test_core.py` | Single-stream exact credits against brute-force flips, exhaustive short binary strings, bit 63 |
| `test_curvature.py` | Diagonal quadratic trust scores on Q/K with linear V convention |
| `test_engine.py` | Batched native training gradients, joint corners, all bit widths 1–64, structured inputs, cache parity, approximate modes |
| `test_forward_lean.py` | Causal SAM forward against naive suffix scans, tie-breaking, fallback bounds |
| `test_layer.py` | PyTorch `NBitQKVRosa` layouts, autograd, embeddings, `bfloat16`, `RosaTransformer`, demo checkpointing |
| `test_native_training.py` | Forward cache ownership, gradient checkpointing, RNG isolation, CPU autocast, non-finite adjoint propagation |
| `test_native_approximation.py` | Residual-row and channel PPS estimators, exact V/embedding/joint corner preservation |
| `test_approx_row_certificate.py` | Exhaustive zero-residual row certificates against naive routes |
| `test_bounded_hash.py` | Adversarial hash collisions forcing bounded-probe ordered fallback |
| `test_expansion.py` | Exact dyadic floating-point expansion against Python `Fraction` |
| `test_validation.py` | Input shape/dtype validation and rejection of complex/integer adjoints |
| `test_structures.py` | Standalone C++ oracles (`interval_credit.cpp`, `subquadratic_delete.cpp`, `subquadratic_indexed_row.cpp`, `subquadratic_training.cpp`) |
