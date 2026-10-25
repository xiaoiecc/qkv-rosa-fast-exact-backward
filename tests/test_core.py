"""All-bit counterfactual checks against direct character comparisons."""
import itertools
import random
import unittest

import numpy as np

from conv_rosa_transformer import rosa_core
from tests._support import COVERAGE, EXHAUSTIVE
from tests.oracle import brute_credits


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rosa_core.build()
        cls.maximum_error = 0.0
        cls.counts = {}

    @classmethod
    def tearDownClass(cls):
        COVERAGE["all_bit_core"] = dict(cls.counts, max_abs_error=cls.maximum_error,
                                        floating_atol=1e-10, floating_rtol=1e-10)

    def check_input(self, q, k, d, rng, category, integer=False):
        n = len(q)
        v = [[rng.randrange(2) for _ in range(d)] for _ in range(n)]
        sample = (lambda: rng.randrange(-3, 4)) if integer else (lambda: rng.uniform(-2, 2))
        g = [[sample() for _ in range(d)] for _ in range(n)]
        e0, e1 = ([sample() for _ in range(d)] for _ in range(2))
        got = rosa_core.credits_numpy(
            q, k, np.asarray(v, dtype=np.uint8).reshape(n, d),
            np.asarray(g, dtype=np.float64).reshape(n, d),
            np.asarray(e0, dtype=np.float64), np.asarray(e1, dtype=np.float64), d)
        want = brute_credits(q, k, v, g, e0, e1, integer=integer)
        self.assertEqual(got[3:5], want[3:5])
        for actual, expected in zip(got[:3], want[:3]):
            expected = np.asarray(expected, dtype=np.float64).reshape(n, d)
            if n:
                type(self).maximum_error = max(self.maximum_error,
                                               float(np.max(np.abs(actual - expected))))
            if integer:
                np.testing.assert_array_equal(actual, expected)
            else:
                np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
        if n:
            self.assertFalse(got[0][0].any(), "Q[0] cannot affect a causal match")
            self.assertFalse(got[1][-1].any(), "last K is never a causal endpoint")
            self.assertFalse(got[2][0].any(), "payload 0 is never selected")
        self.assertLessEqual(got[5]["visited_alignment_pairs"], n * (n - 1) // 2)
        self.counts[category] = self.counts.get(category, 0) + 1

    def test_binary_integer_exhaustive(self):
        rng = random.Random(20261018)
        for n in range(7 if EXHAUSTIVE else 5):
            words = list(itertools.product((0, 1), repeat=n))
            for q in words:
                for k in words:
                    self.check_input(q, k, 1, rng, "binary_integer_inputs", integer=True)

    def test_floating_all_flips(self):
        rng = random.Random(2026101801)
        for trial in range(160 if EXHAUSTIVE else 48):
            d, n = rng.choice((1, 2, 3, 4, 8)), rng.randrange(1, 25)
            q = [rng.randrange(1 << d) for _ in range(n)]
            k = [rng.randrange(1 << d) for _ in range(n)]
            if trial % 4 == 0:
                k = q.copy()
            if trial % 7 == 0:
                q, k = [0] * n, [1] + [0] * (n - 1)
            self.check_input(q, k, d, rng, "floating_inputs")

    def test_wide_symbols(self):
        rng = random.Random(2026101802)
        for d in (16, 31, 32, 63, 64):
            for trial in range(5):
                n = 5 + trial % 3
                q = [rng.getrandbits(d) for _ in range(n)]
                k = q.copy() if trial % 2 else [x ^ (1 << (d - 1)) for x in q]
                self.check_input(q, k, d, rng, "wide_inputs")

    def test_longer_structured_inputs(self):
        rng = random.Random(2026101803)
        for n in ((31, 48) if EXHAUSTIVE else (31,)):
            for d in (1, 4):
                q = [rng.randrange(1 << d) for _ in range(n)]
                period = [i % min(4, 1 << d) for i in range(n)]
                altered = period.copy()
                altered[n // 2] ^= 1
                cases = ((q, q.copy()), (period, altered),
                         ([0] * n, [1] + [0] * (n - 1)), ([0] * n, [0] * n))
                for qq, kk in cases:
                    self.check_input(qq, kk, d, rng, "structured_inputs")


if __name__ == "__main__":
    unittest.main()
