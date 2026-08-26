# Conv∥ROSA Transformer Package

A parallel-mixer Transformer: **hard n-bit QKV-ROSA** as the long-range retrieval
mixer + a **short convolution** as the local mixer, with a SwiGLU FFN. The
backward pass is the exact one-bit counterfactual VJP credit, implemented as a
C++ extension (see `csrc/`).

For a fully commented, single-file pure-Python explanation of the same algorithm,
see [`hard_qkv_rosa_explained.py`](../hard_qkv_rosa_explained.py) at the
repository root.

## Contents

| File | Description |
|---|---|
| `csrc/suffix, ortho, index` | foundations: suffix array / LCE, orthogonal range oracles, bidirectional position-suffix index |
| `csrc/oracles` | A/H deletion-context oracles + affine deletion-run merging |
| `csrc/qrepair` | Q-side repair-term compilation + zero-baseline surfaces |
| `csrc/bridges` | K-deletion threshold surfaces (with polarity merging) + shared one-bit repair bridges (sparse/diagonal materialization) |
| `csrc/certificates` | Equality-Shadow: maximal-run right-boundary square repair certificates in K + static interval index |
| `csrc/krepair` | K-side surface compiler (single threshold / boundary repair / certificate query / exact fallback) |
| `csrc/pipeline` | adaptive backend selection + pipeline entry `exact_stream_bit_credits` |
| `csrc/contract` | numerical contraction (DRS / sweep line / contraction; replicates torch's summation order at IEEE level) |
| `csrc/binding` | compact API: `pack_bits` / `matching_stats` / `repair_backend` / `exact_stream_bit_credits` |
| `rosa_reference.py` | reference Python implementation (the bitwise parity baseline for the C++ port) |
| `self_test.py` | self-check: 25 streams × four backends, bitwise against the reference implementation |
| `build.py` / `build.bat` | torch cpp_extension JIT build (`build.bat` sets up the MSVC environment; usage: `build.bat your_script.py`) |
| `rosa_layer.py` | `NBitQKVRosaCpp` layer wrapper (bitwise identical to the Python reference; compatible parameter names) |
| `rosa_transformer.py` | model definition: `RosaTransformer` stacking `ConvRosaBlock` (`x += ROSA(norm x) + ShortConv(norm x)`; `x += SwiGLU(norm x)`) |
| `multiply_demo.py` | 10-digit × 10-digit multiplication: train → export `multiply_ckpt.pt` → dual-ACC evaluation |
| `multiply_ckpt.pt` | trained example weights (see `multiply_metrics.json`) |
| `multiply_metrics.json` | dual-ACC metrics of the bundled training run |
| `train_log.txt` | training log of the bundled run |

## Design notes

- **Parallel mixers**: hard suffix-match retrieval (ROSA) excels at long-range
  exact copying (induction-style), but cannot express smooth local statistics of
  the form "use the most recent k tokens directly"; a depthwise causal conv with
  K=4 covers exactly that. In toy experiments, the parallel combination beats
  either mixer alone on a hybrid task (loss 2.152 vs 2.237 / 2.253).
- **Q/K weight sharing** (`tie_qk=True` by default): hard threshold codes require
  the same input to be encoded identically in the Q and K streams, otherwise
  matches collapse within tens of training steps. This is a key difference
  between hard routing and the bilinear scoring of self-attention.
- **Output format**: the demo emits products **least-significant digit first**
  (carry propagation then aligns with the generation direction, greatly improving
  learnability) and reverses back to natural order for evaluation.

## Multiplication demo

```bat
build.bat multiply_demo.py                 :: default 1500 steps, C=128, 3 layers, D=32, conv kernel 32
build.bat multiply_demo.py --steps 5000 --layers 4 --n-bits 16
```

About the conv kernel: the model default is `conv_kernel=4` (general local
mixing), while the multiplication demo defaults to `--conv-kernel 32` — because
output digit k (in reverse order) depends on the low k digits of both operands,
and the first answer digit is 13 tokens away from the units digit of `a`;
K=32 places all low-half product digits inside the local window (making the
units-digit rule purely locally learnable).

Metrics:

- `digit_acc`: per-digit accuracy of the answer (left-aligned to the true answer
  length; missing/wrong digits count as errors)
- `exact_acc`: the whole product string is exactly correct
- a per-position breakdown (units → high digits) is included, showing the
  learning gradient across the algorithm's structure

Results of the bundled run (1500 steps, CPU): digit_acc 0.173 / exact_acc 0.000;
units digit 0.94, highest digit 0.80 (Benford-margin predictable), middle digits
still at chance level — the model did learn the lowest-digit local rule, and the
middle digits need longer training (`--steps 10000+`). The overfitting smoke test
`overfit_test.py` (single sample, 300 steps, loss → 0.0007, a 20-digit product
fully reproduced) shows the pipeline and capacity are sound.

## Correctness verification

The C++ extension is bitwise identical to the reference Python implementation:
47 stream-level parity cases (full coverage of the four backends:
shared_sparse / shared_diagonal / surface_run_certified / none) plus 3
layer-level forward and full-gradient parity cases. In-package self-check:
`build.bat self_test.py` (25 streams × four backends, bitwise parity).
