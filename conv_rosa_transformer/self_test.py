"""Independent hard-forward/one-bit-credit audit of the single CPU engine.

Run from the repository root as ``python -m conv_rosa_transformer.self_test``
or directly as ``python conv_rosa_transformer/self_test.py``. This audits exact
integer routing and tolerance-based binary64 credits; it does not select a
reference backend or require bitwise floating-point agreement.
"""
from __future__ import annotations

import random

import torch

if __package__:
    from . import rosa_core
    from .rosa_reference import backward_bruteforce, forward_naive
else:
    import rosa_core
    from rosa_reference import backward_bruteforce, forward_naive


def self_test(n_cases: int = 20, seed: int = 43, verbose: bool = True):
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    cases = []
    for trial in range(n_cases):
        T, D = rng.choice([0, 1, 2, 5, 8, 12]), rng.choice([1, 2, 4, 8])
        alphabet = 1 << D
        q = [rng.randrange(alphabet) for _ in range(T)]
        k = [rng.randrange(alphabet) for _ in range(T)]
        cases.append((f"random-{trial}", q, k, D))
    cases.extend([
        ("empty", [], [], 3),
        ("single", [1], [1], 1),
        ("equal", [0] * 24, [0] * 24, 1),
        ("mismatch", [0] * 24, [3] * 24, 2),
        ("shadow", [0] * 24, [1] + [0] * 23, 1),
        ("periodic", [0, 1, 2, 3] * 6, [0, 1, 2, 3] * 6, 3),
        ("bit-63", [1 << 63, 0, 1 << 63, 1, 1 << 63, 0],
         [1 << 63, 0, 1, 0, 1 << 63, (1 << 64) - 1], 64),
    ])
    max_abs = 0.0
    for tag, q, k, D in cases:
        T = len(q)
        vb = torch.randint(0, 2, (T, D), generator=generator, dtype=torch.uint8)
        gy = torch.randn(T, D, generator=generator, dtype=torch.float64)
        e0 = torch.randn(D, generator=generator, dtype=torch.float64)
        e1 = torch.randn(D, generator=generator, dtype=torch.float64)
        _, expected_lengths, expected_routes = forward_naive(q, k, vb, e0, e1, D)
        assert rosa_core.matching_stats(q, k) == (expected_lengths, expected_routes), tag
        actual = rosa_core.exact_stream_bit_credits(q, k, vb, gy, e0, e1, D)
        assert actual[3] == expected_lengths and actual[4] == expected_routes, tag
        expected = backward_bruteforce(q, k, vb, gy, e0, e1, D)
        for name, a, b in zip(("Q", "K", "V"), actual[:3], expected):
            torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11,
                                       msg=lambda message: f"{tag} {name}: {message}")
            if a.numel():
                max_abs = max(max_abs, (a - b).abs().max().item())
        if verbose:
            print(f"PASS {tag}: T={T} D={D}")
    result = {"streams": len(cases), "max_abs_credit_difference": max_abs}
    print(f"SELF-TEST PASSED: {len(cases)} streams; exact routes; "
          f"max absolute credit difference {max_abs:.3g}")
    return result


if __name__ == "__main__":
    self_test()
