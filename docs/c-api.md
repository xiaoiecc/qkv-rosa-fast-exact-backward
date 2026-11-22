# Standalone C/C++ API & ABI Reference

[简体中文](c-api.zh-CN.md) · [Home](../README.md) · [Algorithm](algorithm.md) · [Training Guide](training.md)

The core engine exposes a zero-dependency C ABI in [`qkv_rosa/csrc/training.h`](../qkv_rosa/csrc/training.h). You can compile and link it directly into C/C++17 programs without Python, NumPy, PyTorch headers, or CUDA.

---

## 1. Building the Standalone Example

[`examples/native_training.cpp`](../examples/native_training.cpp) is a complete runnable example:

```bash
c++ -O3 -std=c++17 -ffp-contract=off \
    examples/native_training.cpp qkv_rosa/csrc/unified_core.cpp \
    -o build/rosa-native

./build/rosa-native                          # Default: second-order exact
./build/rosa-native --first --deterministic  # First-order exact, cache disabled
./build/rosa-native --approximate            # Opt-in approximate backward
```

> Compile with `-ffp-contract=off` (or `/fp:strict` on MSVC) to keep floating-point rounding consistent across platforms.

---

## 2. Public C Functions

Declarations: [`qkv_rosa/csrc/training.h`](../qkv_rosa/csrc/training.h). Implementation: [`qkv_rosa/csrc/training_impl.h`](../qkv_rosa/csrc/training_impl.h) and [`qkv_rosa/csrc/unified_core.cpp`](../qkv_rosa/csrc/unified_core.cpp).

| Function | Purpose |
|---|---|
| `rosa_training_defaults` | Initialize `RosaTrainingOptions` with ABI v1 second-order exact defaults |
| `rosa_approximation_defaults` | Initialize `RosaApproximationOptions` with ABI v1 defaults |
| `rosa_training_forward` | Pack hard `Q/K/V`, compute causal routes, and write `output`, `lengths`, `routes` |
| `rosa_training_backward` | Full exact counterfactual backward (including trust model, joint replays, and logit/embedding gradients) |
| `rosa_training_backward_approx` | Explicit approximate backward (residual-row or channel PPS sampling); requires `saved_lengths`/`saved_routes` from the matching forward |
| `rosa_training_backward_subquadratic` | Unified diagnostic entry point supporting both exact (`approximation == NULL`) and approximate modes, optional saved routes, and optional 22-element structural counters |

---

## 3. Options Structs (ABI v1)

Always initialize option structs through the defaults functions rather than hardcoding `abi_version` or `struct_size`.

### `RosaTrainingOptions`

| Field | Default | Meaning |
|---|---:|---|
| `credit_order` | `2` | `1` = first-order linear credit; `2` = diagonal trust + joint two-bit blocks |
| `randomized` | `1` | `1` = verified full-key query cache enabled; `0` = disabled (identical outputs) |
| `logit_mode` | `0` | `0` = Bernoulli sigmoid derivative; `1` = oriented identity |
| `joint_pairs` | `1` | Max disjoint two-bit blocks per stream |
| `joint_rows` | `32` | Max proposal rows inspected |
| `active_rows` | `1` | `1` = nonzero upstream-gradient rows; `0` = evenly spaced rows |
| `tau` | `1.0` | Positive sigmoid temperature |
| `trust_diagonal` | `0.5` | Diagonal trust scale (0.5 yields $\|g_t\|_2 / \Delta_Y$) |
| `proposal_seed` | `0` | Joint proposal RNG seed |
| `repair_seed` | `0` | Independent cache layout / approximation sampling salt |
| `repair_budget` / `reserved` | `0` | Reserved for ABI compatibility; **must be 0** |

To select first-order mode in C, clear both second-order fields consistently (otherwise status `1` is returned):

```cpp
RosaTrainingOptions options;
rosa_training_defaults(&options);
options.credit_order = 1;
options.joint_pairs = 0;
options.trust_diagonal = 0.0;
```

### `RosaApproximationOptions`

| Field | Default | Meaning |
|---|---:|---|
| `sample_rows` | `32` | Positive residual-row budget per stream |
| `importance` | `1` | `1` = gradient-norm probabilities with 10% uniform floor; `0` = uniform strata |
| `sample_channels` | `0` | `0` = row estimator (all channels); `1..64` = fixed-size PPS channel estimator (`sample_rows` unused) |
| `seed` | `0` | Sampling salt independent of joint proposals |

---

## 4. Array Layouts & Bounds

All buffers must be **row-major contiguous**. Floating arrays use IEEE `double` (`float64`).

| Buffer | Shape | Type |
|---|---|---|
| `q_logits`, `k_logits`, `v_logits`, `output`, `grad_y`, `grad_q`, `grad_k`, `grad_v` | `[B, G, T, D]` | `double` |
| `emb0`, `emb1`, `grad_emb0`, `grad_emb1` | `[G, D]` | `double` |
| Packed `q`, `k` | `[B, G, T]` | `uint64_t` |
| `v_bits` | `[B, G, T, D]` | `uint8_t` (0 or 1 only) |
| `lengths`, `routes` | `[B, G, T]` | `int32_t` |
| Forward / training / approx / engine counters | `[B*G, K]` | `uint64_t` (header macros: 4 / 35 / 40 / 22) |

**Supported shapes:** `B >= 0`, `G > 0`, `0 <= T < 2^28`, `1 <= D <= 64`, `T * D <= INT32_MAX`, `B * G <= INT32_MAX`. Empty streams (`T == 0`) and empty batches (`B == 0`) are supported; bit 63 is a normal unsigned bit.

**Ownership:** The caller allocates all buffers and keeps inputs alive for the duration of the call. Output buffers must not alias inputs. `grad_emb0` and `grad_emb1` sum across batch streams within each group.

**Optional joint trace buffers:** `pair_codes` (`[B*G, joint_pairs, 2]`) and `pair_costs` (`[B*G, joint_pairs, 4]`) may be `NULL`. Codes `< T*D` denote K bits; codes `>= T*D` subtract `T*D` for Q bits (`pos = code / D`, `bit = code % D`). Costs store `C_i, C_j, C_ij, kappa`.

---

## 5. Return Status Codes

| Code | Meaning |
|---:|---|
| `0` | Success |
| `1` | Invalid shape, conflicting options, ABI mismatch, or missing required pointer |
| `2` | Allocation failure (`std::bad_alloc`) |
| `3` | Internal numerical or execution error |
| `4` | Non-finite (`NaN`/`Inf`) value in a required floating input |

Authoritative counter names live in [`qkv_rosa/native.py`](../qkv_rosa/native.py) (`TRAINING_COUNTER_NAMES`, `APPROXIMATION_COUNTER_NAMES`) and [`qkv_rosa/core.py`](../qkv_rosa/core.py) (`COUNTER_NAMES`).
