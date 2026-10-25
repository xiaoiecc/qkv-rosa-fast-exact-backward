"""The supplied diagonal loss model, checked by direct hard rerouting.

These tests do not estimate curvature or claim that a local quadratic model
equals an arbitrary downstream loss. Squared losses supply an exact special
case and a useful finite-step sign counterexample.
"""
import itertools
from math import fsum
import unittest

import numpy as np

from conv_rosa_transformer import rosa_core
from tests._support import COVERAGE
from tests.oracle import hard_outputs


def quadratic_oracle(q, k, v, grad, diagonal, emb0, emb1):
    q, k = list(map(int, q)), list(map(int, k))
    v = np.asarray(v, dtype=np.uint8)
    n, d = v.shape
    baseline, raw = hard_outputs(q, k, v, emb0, emb1)
    y = np.asarray(raw, dtype=np.float64).reshape(n, d)
    expected = [np.zeros((n, d), dtype=np.float64) for _ in range(3)]
    for stream in range(3):
        for owner in range(n):
            for bit in range(d):
                qq, kk, vv = q.copy(), k.copy(), v.copy()
                if stream == 0:
                    qq[owner] ^= 1 << bit
                elif stream == 1:
                    kk[owner] ^= 1 << bit
                else:
                    vv[owner, bit] ^= 1
                _, changed = hard_outputs(qq, kk, vv, emb0, emb1)
                delta = np.asarray(changed, dtype=np.float64).reshape(n, d) - y
                expected[stream][owner, bit] = fsum(
                    float(grad[t, j]) * float(delta[t, j])
                    + 0.5 * float(diagonal[t, j]) * float(delta[t, j]) ** 2
                    for t in range(n) for j in range(d))
    return (*expected, [x[0] for x in baseline], [x[1] for x in baseline])


class CurvatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rosa_core.build()
        cls.checks = 0
        cls.max_error = 0.0

    @classmethod
    def tearDownClass(cls):
        COVERAGE["diagonal_quadratic_credit"] = {
            "single_bit_checks": cls.checks, "max_abs_error": cls.max_error,
            "model": "supplied_nonnegative_output_diagonal",
            "training_claim": False,
        }

    def compare(self, q, k, v, g, h, e0, e1, integer=False):
        got = rosa_core.quadratic_credits_numpy(q, k, v, g, h, e0, e1)
        expected = quadratic_oracle(q, k, v, g, h, e0, e1)
        self.assertEqual(got[3:5], expected[3:5])
        for actual, target in zip(got[:3], expected[:3]):
            if actual.size:
                type(self).max_error = max(self.max_error,
                                           float(np.max(np.abs(actual - target))))
            if integer:
                np.testing.assert_array_equal(actual, target)
            else:
                np.testing.assert_allclose(actual, target, atol=2e-10, rtol=2e-10)
            type(self).checks += actual.size
        return got

    def test_small_binary_exhaustive(self):
        rng = np.random.default_rng(2026102501)
        for n in range(5):
            words = list(itertools.product((0, 1), repeat=n))
            for q in words:
                for k in words:
                    v = rng.integers(0, 2, (n, 1), dtype=np.uint8)
                    g = rng.integers(-3, 4, (n, 1)).astype(np.float64)
                    h = 2 * rng.integers(0, 4, (n, 1)).astype(np.float64)
                    e0, e1 = rng.integers(-3, 4, (2, 1)).astype(np.float64)
                    self.compare(q, k, v, g, h, e0, e1, integer=True)

    def test_random_and_wide_symbols(self):
        rng = np.random.default_rng(2026102502)
        for trial in range(55):
            d = int(rng.choice((1, 2, 3, 8, 64)))
            n = int(rng.integers(1, 13))
            q = rng.bit_generator.random_raw(n)
            if d < 64:
                q &= np.uint64((1 << d) - 1)
            k = q.copy()
            if trial % 2:
                k ^= np.uint64(1) << np.uint64(d - 1)
            v = rng.integers(0, 2, (n, d), dtype=np.uint8)
            g = rng.normal(size=(n, d))
            h = rng.uniform(0, 3, (n, d))
            e0, e1 = rng.normal(size=(2, d))
            self.compare(q, k, v, g, h, e0, e1)

    def test_caches_corridors_and_moving_owners(self):
        rng = np.random.default_rng(2026102503)
        for n, d, shadow in ((48, 8, True), (31, 3, False)):
            q = np.zeros(n, dtype=np.uint64)
            k = q.copy()
            if shadow:
                k[0] = 1
            v = rng.integers(0, 2, (n, d), dtype=np.uint8)
            g = rng.normal(size=(n, d))
            h = rng.uniform(0.1, 2, (n, d))
            e0, e1 = rng.normal(size=(2, d))
            got = self.compare(q, k, v, g, h, e0, e1)
            if shadow:
                self.assertGreater(got[5]["singleton_corridors"], 0)
                self.assertGreater(got[5]["singleton_dot_components"], 0)
                self.assertTrue(got[5]["curvature_history_materialized"])
                self.assertEqual(got[5]["curvature_working_bytes"], 16 * d * (n + 1))
            else:
                self.assertGreater(got[5]["deletion_owner_ranges"], 0)
                self.assertFalse(got[5]["curvature_history_materialized"])
                self.assertEqual(got[5]["curvature_working_bytes"], 16 * d)

    def test_zero_metric_preserves_first_order_bits(self):
        rng = np.random.default_rng(2026102504)
        n, d = 37, 4
        q = np.zeros(n, dtype=np.uint64)
        k = q.copy()
        k[0] = 1
        v = rng.integers(0, 2, (n, d), dtype=np.uint8)
        g = rng.normal(size=(n, d))
        e0, e1 = rng.normal(size=(2, d))
        original = rosa_core.credits_numpy(q, k, v, g, e0, e1)
        quadratic = rosa_core.quadratic_credits_numpy(q, k, v, g, np.zeros_like(g), e0, e1)
        for x, y in zip(original[:3], quadratic[:3]):
            np.testing.assert_array_equal(x, y)
        self.assertEqual(original[3:5], quadratic[3:5])
        for name, value in original[5].items():
            self.assertEqual(value, quadratic[5][name])
        self.assertEqual(quadratic[5]["curvature_working_bytes"], 0)

    def test_finite_step_sign_and_constant_outputs(self):
        q = k = [0, 0]
        v = np.zeros((2, 1), dtype=np.uint8)
        e0, e1 = np.array([0.0]), np.array([1.0])
        g = np.array([[0.0], [-0.5]])
        h = np.ones_like(g)
        first = rosa_core.credits_numpy(q, k, v, g, e0, e1)
        second = self.compare(q, k, v, g, h, e0, e1)
        self.assertEqual(first[2][1, 0], -1.0)
        self.assertEqual(second[2][1, 0], 1.0)
        # Any routing change is irrelevant if both empty and all signed values
        # are exactly zero. Curvature must not manufacture a Q/K signal.
        n, d = 20, 3
        zeros = np.zeros((n, d), dtype=np.uint8)
        got = rosa_core.quadratic_credits_numpy(
            [0] * n, [1] + [0] * (n - 1), zeros,
            np.full((n, d), 1e16), np.full((n, d), 1e16),
            np.zeros(d), np.zeros(d))
        for credit in got[:3]:
            self.assertFalse(credit.any())

    def test_metric_validation(self):
        args = ([0, 0], [0, 0], np.zeros((2, 1), np.uint8),
                np.ones((2, 1)), np.zeros(1), np.ones(1))
        q, k, v, g, e0, e1 = args
        for bad in (np.ones((2,)), -np.ones((2, 1)),
                    np.full((2, 1), np.inf), np.full((2, 1), np.nan),
                    np.ones((2, 1), dtype=np.complex128)):
            with self.subTest(metric=str(bad)):
                with self.assertRaises(ValueError):
                    rosa_core.quadratic_credits_numpy(q, k, v, g, bad, e0, e1)

    def test_small_jump_between_large_payloads(self):
        # The algebraic absolute score (g-hY)*P + hP^2/2 loses this 0.5
        # after subtracting ~1e16 baselines. Two-value relative costs retain it.
        q, k = [0, 1], [0, 0]
        v = np.ones((2, 1), dtype=np.uint8)
        g, h = np.zeros((2, 1)), np.ones((2, 1))
        got = self.compare(q, k, v, g, h, np.array([1e8]), np.array([1e8+1]))
        self.assertEqual(got[0][1, 0], 0.5)
        # The same semantic equality is independent of the absolute scale.
        for magnitude in (2.0**30, 2.0**40):
            got = rosa_core.quadratic_credits_numpy(
                q, k, v, g, h, np.array([magnitude]), np.array([magnitude+1]))
            self.assertEqual(got[0][1, 0], 0.5)


if __name__ == "__main__":
    unittest.main()
